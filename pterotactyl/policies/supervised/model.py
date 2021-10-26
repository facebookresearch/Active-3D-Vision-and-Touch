# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch

from pterotactyl.utility import utils


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

    def forward(self, obs):

        action_input = self.action_model(obs["mask"].float().cuda())
        shape_input_1 = obs["latent"].float().cuda()
        shape_input_2 = obs["first_latent"].float().cuda()
        full_input = torch.cat((action_input, shape_input_1, shape_input_2), dim=-1)
        if self.args.normalize:
            value = torch.sigmoid(self.model(full_input)) * 2 - 1
        elif self.args.use_img:
            value = torch.sigmoid(self.model(full_input)) * 6 - 3
        else:
            value = torch.sigmoid(self.model(full_input)) * 200 - 100
        return value
