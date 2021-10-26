# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

from pterotactyl.utility import utils

# DDQN Q network which makes use of a pertrained latent space of predicted objects
class Latent_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        latent_size = utils.load_model_config(self.args.auto_location)[0].encoding_size

        # for embedding previously performed actions
        layers = []
        layers.append(nn.Sequential(nn.Linear(self.args.num_actions, 200), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(200, 100), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(100, latent_size)))
        self.action_model = nn.Sequential(*layers)

        # MLP taking as input embedding of actions, a latent embedding of first prediction, and current prediction
        # and predicts a value for every action
        hidden_sizes = (
            [latent_size * 3]
            + [args.hidden_dim for _ in range(args.layers - 1)]
            + [self.args.num_actions]
        )
        layers = []
        for i in range(args.layers):
            if i < args.layers - 1:
                layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()
                    )
                )
            else:
                layers.append(
                    nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                )
        self.model = nn.Sequential(*layers)
        self.args = args

    def forward(self, obs, next=False):
        if next:
            action_input = self.action_model(obs["mask_n"].float().cuda())
            shape_input_1 = obs["latent_n"].float().cuda()
        else:
            action_input = self.action_model(obs["mask"].float().cuda())
            shape_input_1 = obs["latent"].float().cuda()
        shape_input_2 = obs["first_latent"].float().cuda()
        full_input = torch.cat((action_input, shape_input_1, shape_input_2), dim=-1)
        value = self.model(full_input)
        return value


# DDQN Q network which makes use of full mesh prediction
class Graph_Model(nn.Module):
    def __init__(self, args, adj):
        super().__init__()
        self.adj = adj["adj"].data.cpu().cuda()
        self.args = args
        self.num_layers = args.layers
        input_size = 100

        # for embedding previously performed actions
        layers = []
        layers.append(nn.Sequential(nn.Linear(50, 200), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(200, 100), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(100, input_size)))
        self.action_model = nn.Sequential(*layers)

        # embedding of vertex positions and masks
        self.positional_embedding = Positional_Encoder(input_size)
        self.mask_embedding = Mask_Encoder(input_size)

        # GCN for predicting actions values from input mesh
        hidden_sizes = (
            [input_size * 3]
            + [args.hidden_dim for _ in range(args.layers - 1)]
            + [self.args.num_actions]
        )
        layers = []
        for i in range(args.layers):
            layers.append(
                GCN_layer(
                    hidden_sizes[i],
                    hidden_sizes[i + 1],
                    cut=self.args.cut,
                    do_cut=(i != self.num_layers - 1),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, obs, next=False):

        if next:
            action_embedding = self.action_model(obs["mask_n"].float().cuda())
            mesh = obs["mesh_n"][:, :, :3].float().cuda()
            mask = obs["mesh_n"][:, :, 3:].float().cuda()
        else:
            action_embedding = self.action_model(obs["mask"].float().cuda())
            mesh = obs["mesh"][:, :, :3].float().cuda()
            mask = obs["mesh"][:, :, 3:].float().cuda()

        positional_embedding = self.positional_embedding(mesh)
        mask_embedding = self.mask_embedding(mask)
        action_embedding = action_embedding.unsqueeze(1).repeat(1, mesh.shape[1], 1)
        vertex_features = torch.cat(
            (action_embedding, positional_embedding, mask_embedding), dim=-1
        )

        # iterate through GCN layers
        x = self.layers[0](vertex_features, self.adj, F.relu)
        for i in range(1, self.num_layers):
            x = self.layers[i](
                x, self.adj, F.relu if (i != self.num_layers - 1) else lambda x: x
            )
        value = torch.max(x, dim=1)[0]
        return value


# Graph convolutional network layer
class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, cut=0.33, do_cut=True):
        super(GCN_layer, self).__init__()
        self.weight = Parameter(torch.Tensor(1, in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.cut_size = cut
        self.do_cut = do_cut

    def reset_parameters(self):
        stdv = 6.0 / math.sqrt((self.weight.size(1) + self.weight.size(0)))
        stdv *= 0.3
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, features, adj, activation):
        features = torch.matmul(features, self.weight)
        # uf we want to only share a subset of features with neighbors
        if self.do_cut:
            length = round(features.shape[-1] * self.cut_size)
            output = torch.matmul(adj, features[:, :, :length])
            output = torch.cat((output, features[:, :, length:]), dim=-1)
            output[:, :, :length] += self.bias[:length]
        else:
            output = torch.matmul(adj, features)
            output = output + self.bias

        return activation(output)


# encode the positional information of vertices using Nerf Embeddings
class Positional_Encoder(nn.Module):
    def __init__(self, input_size):
        super(Positional_Encoder, self).__init__()
        layers = []
        layers.append(
            nn.Linear(63, input_size // 4)
        )  # 10 nerf layers + original positions
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_size // 4, input_size // 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_size // 2, input_size))
        self.model = nn.Sequential(*layers)

    # apply nerf embedding of the positional information
    def nerf_embedding(self, points):
        embeddings = []
        for i in range(10):
            if i == 0:
                embeddings.append(torch.sin(np.pi * points))
                embeddings.append(torch.cos(np.pi * points))
            else:
                embeddings.append(torch.sin(np.pi * 2 * i * points))
                embeddings.append(torch.cos(np.pi * 2 * i * points))
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings

    def forward(self, positions):
        shape = positions.shape
        positions = positions.contiguous().view(shape[0] * shape[1], -1)
        # combine nerf embedding with origional positions
        positions = torch.cat((self.nerf_embedding((positions)), positions), dim=-1)
        embeding = self.model(positions).view(shape[0], shape[1], -1)

        return embeding


# embedding network for vetex masks
class Mask_Encoder(nn.Module):
    def __init__(self, input_size):
        super(Mask_Encoder, self).__init__()
        layers_mask = []
        layers_mask.append(nn.Embedding(4, input_size))
        self.model = nn.Sequential(*layers_mask)

    def forward(self, mask):
        shape = mask.shape
        mask = mask.contiguous().view(-1, 1)
        embeding_mask = self.model(mask.long()).view(shape[0], shape[1], -1)
        return embeding_mask
