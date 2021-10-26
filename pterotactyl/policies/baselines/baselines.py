# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random


# class for getting radnom samples from the space of action
class random_sampler:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def get_action(self, mask):
        batch_size = mask.shape[0]
        actions = []
        for b in range(batch_size):
            propositions = list(np.arange(self.args.num_actions))
            indexes = list(np.where(mask[b] > 0)[0])
            if len(indexes) > 0:
                for index in sorted(indexes, reverse=True):
                    del propositions[index]
            actions.append(random.choice(propositions))
        return np.array(actions)


# class for evenly spaced samples from the space of actions
class even_sampler:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.generate_points()

    # precompute the actions to be used in the trajectory
    def generate_points(self):
        self.angles = []
        for i in range(self.args.env_batch_size):
            spacing = self.args.num_actions // self.args.num_grasps
            set = [spacing * i for i in range(self.args.num_grasps)]
            update_num = random.choice(range(self.args.num_actions))
            choice = []
            for j in range(self.args.num_grasps):
                choice.append((set[j] + update_num) % self.args.num_actions)
            self.angles.append(choice)

    # reset the precomputed actions
    def reset(self):
        self.generate_points()

    def get_action(self, mask):
        batch_size = mask.shape[0]
        actions = []
        for b in range(batch_size):
            actions.append(self.angles[b][0])
            del self.angles[b][0]
        return np.array(actions)
