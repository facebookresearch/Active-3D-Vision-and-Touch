# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# class for the autoencoder
# for extracting latent vector from predicted shape
class AutoEncoder(nn.Module):
    def __init__(self, adj_info, inital_positions, args, only_encode=False):
        super(AutoEncoder, self).__init__()
        self.adj_info = adj_info
        self.initial_positions = inital_positions
        self.args = args
        # feature size passed to the GCN
        input_size = 50
        self.only_encode = only_encode

        self.positional_encoder = Positional_Encoder(input_size)
        self.mask_encoder = Mask_Encoder(input_size)
        self.encoder = Encoder(input_size, args)
        if not self.only_encode:
            self.decoder = Decoder(args).cuda()

    def forward(self, verts, mask, only_encode=False):
        positional_features = self.positional_encoder(verts)
        mask_features = self.mask_encoder(mask)
        # combine mesh features
        vertex_features = positional_features + mask_features
        latent = self.encoder(vertex_features, self.adj_info)
        if self.only_encode or only_encode:
            return latent
        pred_points = self.decoder(latent)
        return pred_points.permute(0, 2, 1), latent


# encoder for the auto encoder
class Encoder(nn.Module):
    def __init__(self, input_features, args):
        super(Encoder, self).__init__()

        self.num_layers = args.num_GCN_layers
        # define output sizes for each GCN layer
        hidden_values = [input_features] + [
            args.hidden_GCN_size for _ in range(self.num_layers)
        ]

        # define layers
        layers = []
        for i in range(self.num_layers):
            layers.append(
                GCN_layer(
                    hidden_values[i],
                    hidden_values[i + 1],
                    args.cut,
                    do_cut=i < self.num_layers - 1,
                )
            )
        self.layers = nn.ModuleList(layers)

        # MLP layers
        hidden_values = [args.hidden_GCN_size, 500, 400, 300, args.encoding_size]
        num_layers = len(hidden_values) - 1
        layers = []
        for i in range(num_layers):
            if i < num_layers - 1:
                layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_values[i], hidden_values[i + 1]), nn.ReLU()
                    )
                )
            else:
                layers.append(
                    nn.Sequential(nn.Linear(hidden_values[i], hidden_values[i + 1]))
                )
        self.mlp = nn.Sequential(*layers)

    def forward(self, features, adj_info):
        adj = adj_info["adj"]
        for i in range(self.num_layers):
            activation = F.relu if i < self.num_layers - 1 else lambda x: x
            features = self.layers[i](features, adj, activation)
        features = features.max(dim=1)[0]
        features = self.mlp(features)
        return features


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
        # if we want to only share a subset of features with neighbors
        if self.do_cut:
            length = round(features.shape[-1] * self.cut_size)
            output = torch.matmul(adj, features[:, :, :length])
            output = torch.cat((output, features[:, :, length:]), dim=-1)
            output[:, :, :length] += self.bias[:length]
        else:
            output = torch.matmul(adj, features)
            output = output + self.bias

        return activation(output)


# decoder for the autoencoder
# this is just Foldingnet
class Decoder(nn.Module):
    def __init__(self, args, rank=0):
        super(Decoder, self).__init__()
        self.model = FoldingNetDec(rank=rank)
        self.initial = nn.Linear(args.encoding_size, 512)

    def forward(self, features):
        features = self.initial(features)
        points = self.model(features)
        return points


# foldingnet definition
class FoldingNetDecFold1(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = nn.Conv1d(514, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


# foldingnet definition
def GridSamplingLayer(batch_size, meshgrid):
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros(
        (np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32
    )  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)

    return g


# foldingnet definition
class FoldingNetDecFold2(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = nn.Conv1d(515, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# foldingnet definition
class FoldingNetDec(nn.Module):
    def __init__(self, rank=0):
        super(FoldingNetDec, self).__init__()
        self.rank = rank
        self.fold1 = FoldingNetDecFold1()
        self.fold2 = FoldingNetDecFold2()

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 80 ** 2, 1)  # x = batch,45^2,512
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-0.5, 0.5, 80], [-0.5, 0.5, 80]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid).cuda(self.rank)

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2
        x = self.fold1(x)  # x = batch,3,45^2
        x = torch.cat((code, x), 1)  # x = batch,515,45^2
        x = self.fold2(x)  # x = batch,3,45^2

        return x


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


# make embedding token of the mask information for each vertex
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
