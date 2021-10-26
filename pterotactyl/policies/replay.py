# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch

from pterotactyl.utility import utils

BASE_MESH_SIZE = 1824
BASE_CHART_SIZE = 25

# replay buffer used for learning RL models over the environment
class ReplayMemory:
    def __init__(self, args):
        self.args = args
        # basic info which might be used by a learning method
        # _n denotes observations occuring after the action is perfromed
        self.mask = torch.zeros((self.args.mem_capacity, self.args.num_actions))
        self.mask_n = torch.zeros((self.args.mem_capacity, self.args.num_actions))
        self.actions = torch.zeros((self.args.mem_capacity))
        self.rewards = torch.zeros(self.args.mem_capacity)
        self.score = torch.zeros(self.args.mem_capacity)
        self.score_n = torch.zeros(self.args.mem_capacity)
        self.first_score = torch.zeros(self.args.mem_capacity)
        if self.args.use_recon:
            num_fingers = 1 if self.args.finger else 4
            mesh_shape = BASE_MESH_SIZE + (
                BASE_CHART_SIZE * self.args.num_grasps * num_fingers
            )
            self.mesh = torch.zeros((self.args.mem_capacity, mesh_shape, 4))
            self.mesh_n = torch.zeros((self.args.mem_capacity, mesh_shape, 4))
        if self.args.use_latent:
            latent_size = utils.load_model_config(self.args.auto_location)[
                0
            ].encoding_size
            self.latent = torch.zeros((self.args.mem_capacity, latent_size))
            self.latent_n = torch.zeros((self.args.mem_capacity, latent_size))
            self.first_latent = torch.zeros((self.args.mem_capacity, latent_size))

        self.position = 0
        self.count_seen = 0

    # add a set of transitions to the replay buffer
    def push(self, action, observation, next_observation, reward):
        for i in range(len(action)):
            self.actions[self.position] = action[i]
            self.rewards[self.position] = reward[i]
            self.score[self.position] = observation["score"][i]
            self.score_n[self.position] = next_observation["score"][i]
            self.first_score[self.position] = observation["first_score"][i]
            self.mask[self.position] = observation["mask"][i]
            self.mask_n[self.position] = next_observation["mask"][i]

            if self.args.use_recon:
                self.mesh[self.position] = observation["mesh"][i]
                self.mesh_n[self.position] = next_observation["mesh"][i]
            if self.args.use_latent:
                self.latent[self.position] = observation["latent"][i]
                self.latent_n[self.position] = next_observation["latent"][i]
                self.first_latent[self.position] = observation["first_latent"][i]

            self.count_seen += 1
            self.position = (self.position + 1) % self.args.mem_capacity

    # sample a set of transitions from the replay buffer
    def sample(self):
        if (
            self.count_seen < self.args.burn_in
            or self.count_seen < self.args.train_batch_size
        ):
            return None
        indices = np.random.choice(
            min(self.count_seen, self.args.mem_capacity), self.args.train_batch_size
        )
        data = {
            "mask": self.mask[indices],
            "mask_n": self.mask_n[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "score": self.score[indices],
            "score_n": self.score_n[indices],
            "first_score": self.first_score[indices],
        }
        if self.args.use_recon:
            data["mesh"] = self.mesh[indices]
            data["mesh_n"] = self.mesh_n[indices]
        if self.args.use_latent:
            data["latent"] = self.latent[indices]
            data["latent_n"] = self.latent_n[indices]
            data["first_latent"] = self.first_latent[indices]

        return data

    # save the replay buffer to disk
    def save(self, directory):
        data = {
            "mask": self.mask,
            "mask_n": self.mask_n,
            "actions": self.actions,
            "rewards": self.rewards,
            "score": self.score,
            "first_score": self.first_score,
            "position": self.position,
            "count_seen": self.count_seen,
        }
        if self.args.use_recon:
            data["mesh"] = self.mesh
            data["mesh_n"] = self.mesh_n
        if self.args.use_latent:
            data["latent"] = self.latent
            data["latent_n"] = self.latent_n
            data["first_latent"] = self.first_latent

        temp_path = directory + "_replay_buffer_temp.pt"
        full_path = directory + "_replay_buffer.pt"
        torch.save(data, temp_path)
        os.rename(temp_path, full_path)

    # load the replay buffer from the disk
    def load(self, directory):
        data = torch.load(directory + "_replay_buffer.pt")

        self.mask = data["mask"]
        self.mask_n = data["mask_n"]
        self.actions = data["actions"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.score = data["score"]
        self.first_score = data["first_score"]
        self.position = data["position"]
        self.count_seen = data["count_seen"]

        if self.args.use_recon:
            self.mesh = data["mesh"]
            self.mesh_n = data["mesh_n"]

        if self.args.use_latent:
            self.latent = data["latent"]
            self.latent_n = data["latent_n"]
            self.first_latent = data["first_latent"]
