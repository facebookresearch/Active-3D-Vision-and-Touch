# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
import torch.nn as nn
import torch.optim as optim

from pterotactyl.policies.DDQN import model
from pterotactyl.policies.baselines import baselines

# DDQN training module
class DDQN(nn.Module):
    def __init__(self, args, adj_info, replay):
        super().__init__()
        self.args = args
        self.model = self.get_model(adj_info)
        self.replay = replay
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.args = args
        self.random_sampler = baselines.random_sampler(self.args)

    # set the value of perfromaed action to never be selected
    def penalise_actions(self, values, obs):
        values[obs["mask"] > 0] = -1e10
        return values

    # select the model type required
    def get_model(self, adj):
        if self.args.pretrained:
            if self.args.use_latent:
                if self.args.use_img:
                    if self.args.finger:
                        self.args.hidden_dim = 300
                        self.args.layers = 5
                    else:
                        self.args.hidden_dim = 300
                        self.args.layers = 5
                else:
                    if self.args.finger:
                        self.args.hidden_dim = 300
                        self.args.layers = 5
                    else:
                        self.args.hidden_dim = 300
                        self.args.layers = 2
            else:
                if self.args.use_img:
                    if self.args.finger:
                        self.args.hidden_dim = 100
                        self.args.layers = 5
                    else:
                        self.args.hidden_dim = 100
                        self.args.layers = 5
                else:
                    if self.args.finger:
                        self.args.hidden_dim = 100
                        self.args.layers = 5
                    else:
                        self.args.hidden_dim = 100
                        self.args.layers = 2

        if self.args.use_latent:
            return model.Latent_Model(self.args).cuda()
        elif self.args.use_recon:
            return model.Graph_Model(self.args, adj).cuda()
        else:
            print("No Model type selected")
            exit()

    # decrease the epsilon value
    def update_epsilon(self, epsilon, args):
        return max(args.epsilon_end, epsilon * args.epsilon_decay)

    # add the observed transition to the replay buffer
    def add_experience(self, action, observation, next_observation, reward):
        self.replay.push(action, observation, next_observation, reward)

    # update the parameters of the model using DDQN update rule
    def update_parameters(self, target_net):
        self.model.train()
        batch = self.replay.sample()
        if batch is None:
            return None

        # get observations
        not_done_mask = batch["mask"].cuda().sum(dim=1) < self.args.budget - 1
        actions = batch["actions"].cuda()
        rewards = batch["rewards"].cuda()
        cur_score = batch["score"].cuda()
        first_score = batch["first_score"].cuda()

        # normalize if needed
        if self.args.normalization == "first":
            rewards = rewards / first_score
        elif self.args.normalization == "current":
            rewards = rewards / cur_score

        # Standard DDQN update rule
        all_q_values_cur = self.forward(batch, penalize=False)
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1).long()).squeeze()

        with torch.no_grad():
            best_next_action = self.forward(batch, next=True).detach().max(1)[1]
            target_values = target_net.forward(
                batch, next=True, penalize=False
            ).detach()
            all_q_values_next = torch.zeros((q_values.shape[0])).cuda()
            for i in range(q_values.shape[0]):
                if not_done_mask[i]:
                    all_q_values_next[i] = target_values[i][best_next_action[i]]
            target_values = (self.args.gamma * all_q_values_next) + rewards

        loss = ((q_values - target_values) ** 2).mean()

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def forward(self, obs, next=False, penalize=True):
        value = self.model(obs, next=next)
        if penalize:
            value = self.penalise_actions(value, obs)
        return value

    def get_action(self, obs, eps_threshold, give_random=False):
        sample = random.random()
        if sample < eps_threshold or give_random:
            return self.random_sampler.get_action(obs["mask"])
        else:
            with torch.no_grad():
                self.model.eval()
                q_values = self(obs)

            actions = torch.argmax(q_values, dim=1).data.cpu().numpy()
            return actions
