# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
from submitit.helpers import Checkpointable

from pterotactyl.policies import environment
from pterotactyl.utility import utils
from pterotactyl import pretrained


class Engine(Checkpointable):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # setup the environment, and data
        self.env = environment.ActiveTouch(self.args)
        data_loaders, valid_loaders = self.get_loaders()
        self.chosen_actions = []
        self.step = 0
        self.spot = 0
        self.counts = np.array([0.0 for i in range(self.args.num_actions)])

        # save location for the computed trajectory
        self.results_dir = os.path.join("results", self.args.exp_type)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.checkpoint_dir = os.path.join(
            "experiments/checkpoint/", "MFBA", self.args.exp_type
        )
        if not os.path.exists((self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint = self.checkpoint_dir + "actions.npy"

        with torch.no_grad():
            self.load()
            if self.args.eval:
                self.validate(valid_loaders)
            else:
                # find the best action at every step
                for i in range(self.step, self.args.num_grasps):
                    self.train(data_loaders)
                    self.save()

    # load data using pytorch dataloader
    def get_loaders(self):
        if not self.args.eval:
            train_loader = DataLoader(
                self.env.train_data,
                batch_size=self.args.env_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self.env.train_data.collate,
            )
        else:
            train_loader = []
        valid_loader = DataLoader(
            self.env.valid_data,
            batch_size=self.args.env_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.env.valid_data.collate,
        )
        return train_loader, valid_loader

    # compute the lowest error action for the current step
    def train(self, dataloader):
        print(f"Getting best action for step {len(self.chosen_actions)+1}")
        training_length = len(dataloader)
        random.seed(self.args.seed)
        training_instances = random.sample(
            range(training_length), int(training_length * 0.4)
        )
        for v, batch in enumerate(tqdm(dataloader)):
            if v < self.spot:
                continue
            if v not in training_instances:
                continue
            self.spot = v
            self.env.reset(batch)
            # check the accuracy of every action
            for action in self.chosen_actions:
                actions = np.array([action for _ in range(self.args.env_batch_size)])
                self.env.step(actions)

            actions, _, _, _ = self.env.best_step(greedy_checks=self.args.greedy_checks)

            # update the count for most successful action
            for a in actions:
                self.counts[a] += 1
            if v % 20 == 0:
                self.save()

        self.chosen_actions.append(np.argmax(self.counts))
        self.counts = np.array(
            [
                0 if i not in self.chosen_actions else -1e20
                for i in range(self.args.num_actions)
            ]
        )
        self.spot = 0
        self.step += 1

    # evaluate the policy
    def validate(self, dataloader):
        observations = []
        scores = []
        actions = []
        names = []
        for v, batch in enumerate(tqdm(dataloader)):
            names += batch["names"]
            obs = self.env.reset(batch)
            cur_scores = [obs["score"]]
            cur_actions = []

            for action in self.chosen_actions:
                best_actions = np.array(
                    [action for _ in range(self.args.env_batch_size)]
                )
                # perform actions
                with torch.no_grad():
                    next_obs, _, _ = self.env.step(best_actions)

                # record actions
                torch.cuda.empty_cache()
                obs = next_obs
                cur_scores.append(obs["score"])
                cur_actions.append(torch.FloatTensor(best_actions))

            observations.append(obs["mesh"])
            scores.append(torch.stack(cur_scores).permute(1, 0))
            actions.append(torch.stack(cur_actions).permute(1, 0))

            print_score = (scores[-1][:, -1] / scores[-1][:, 0]).mean()
            print_reward = (
                (scores[-1][:, 0] - scores[-1][:, -1]) / scores[-1][:, 0]
            ).mean()

            message = f"Valid || score: {print_score:.4f}, "
            message += f"reward = {print_reward:.4f}"
            tqdm.write(message)

            if self.args.visualize and v == 5:
                meshes = torch.cat(observations, dim=0)[:, :, :3]
                utils.visualize_prediction(
                    self.results_dir, meshes, self.env.mesh_info["faces"], names
                )
                self.env.reset_pybullet()

        scores = torch.cat(scores)
        rewards = ((scores[:, 0] - scores[:, -1]) / scores[:, 0]).mean()
        current_loss = (scores[:, -1] / scores[:, 0]).mean()

        if self.args.visualize:
            actions = torch.stack(actions).view(-1, self.args.budget)
            utils.visualize_actions(self.results_dir, actions, self.args)

        print("*" * 30)
        message = f"Total Valid || score: {current_loss:.4f}, "
        message += f"reward = {rewards.mean():.4f}"
        tqdm.write("*" * len(message))
        tqdm.write(message)
        tqdm.write("*" * len(message))

    def load(self):
        if self.args.pretrained:
            if self.args.use_img:
                if self.args.finger:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/dataset_specific/MFBA_v_t_p.npy"
                    )
                else:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/dataset_specific/MFBA_v_t_g.npy"
                    )
            else:
                if self.args.finger:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/dataset_specific/MFBA_t_p.npy"
                    )
                else:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/dataset_specific/MFBA_t_g.npy"
                    )
            data = np.load(location, allow_pickle=True).item()
            self.counts = data["counts"]
            self.chosen_actions = data["chosen_actions"]
            self.spot = data["spot"]
            self.step = data["step"]
        else:
            try:
                data = np.load(self.checkpoint, allow_pickle=True).item()
                self.counts = data["counts"]
                self.chosen_actions = data["chosen_actions"]
                self.spot = data["spot"]
                self.step = data["step"]
            except:
                return

    def save(self):
        data = {
            "counts": self.counts,
            "chosen_actions": self.chosen_actions,
            "step": self.step,
            "spot": self.spot,
        }
        np.save(self.checkpoint, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit_data",
        action="store_true",
        default=False,
        help="use less data, for debugging.",
    )
    parser.add_argument(
        "--finger", action="store_true", default=False, help="use only one finger."
    )
    parser.add_argument(
        "--touch_location",
        type=str,
        default=os.path.dirname(pretrained.__file__) + "/reconstruction/touch/best/",
        help="the location of the touch part prediction.",
    )
    parser.add_argument(
        "--vision_location",
        type=str,
        default=os.path.dirname(pretrained.__file__) + "/reconstruction/vision/t_p/",
        help="the location of the touch part prediction.",
    )
    parser.add_argument(
        "--number_points",
        type=int,
        default=30000,
        help="number of points sampled for the chamfer distance.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Setting for the random seed."
    )
    parser.add_argument(
        "--env_batch_size", type=int, default=3, help="Size of the batch."
    )
    parser.add_argument(
        "--use_img", action="store_true", default=False, help="To use the image."
    )
    parser.add_argument(
        "--loss_coeff", type=float, default=9000.0, help="Coefficient for loss term."
    )

    parser.add_argument(
        "--num_grasps", type=int, default=5, help="Number of grasps to train with. "
    )
    parser.add_argument("--num_actions", type=int, default=50)
    parser.add_argument("--use_latent", action="store_true", default=False)
    parser.add_argument("--use_recon", action="store_true", default=False)
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Evaluate the trained model on the test set.",
    )
    parser.add_argument(
        "--budget", type=int, default=5, help="number of graspsp to perform"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="visualize predictions and actions while evaluating",
    )
    parser.add_argument(
        "--exp_type", type=str, default="test", help="The experiment group."
    )
    parser.add_argument(
        "--greedy_checks",
        type=int,
        default=50,
        help="Number of actions to check at each time step",
    )
    parser.add_argument(
        "--pretrained_recon",
        action="store_true",
        default=False,
        help="use the pretrained reconstruction models to train",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="use the pretrained policy",
    )

    args = parser.parse_args()

    trainer = Engine(args)
    trainer()
