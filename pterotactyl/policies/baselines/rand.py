# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
import argparse

from pterotactyl.policies import environment
from pterotactyl.policies.baselines import baselines
from pterotactyl.utility import utils
from pterotactyl import pretrained


class Engine:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # set up environment and policy and data
        utils.set_seeds(self.args.seed)
        self.env = environment.ActiveTouch(self.args)
        self.policy = baselines.random_sampler(self.args)
        valid_loaders = self.get_loaders()
        self.results_dir = os.path.join("results", self.args.exp_type)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        # compute accuracy
        with torch.no_grad():
            self.validate(valid_loaders)

    # load data with pytorch dataloader
    def get_loaders(self):
        valid_loader = DataLoader(
            self.env.valid_data,
            batch_size=self.args.env_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.env.valid_data.collate,
        )
        return valid_loader

    # perform the random policy
    def validate(self, dataloader):
        observations = []
        scores = []
        actions = []
        names = []
        for v, batch in enumerate(tqdm(dataloader)):
            names += batch["names"]
            obs = self.env.reset(batch)
            all_done = False
            cur_scores = [obs["score"]]
            cur_actions = []
            while not all_done:
                # select actions
                action = self.policy.get_action(obs["mask"])
                # perform actions
                with torch.no_grad():
                    next_obs, reward, all_done = self.env.step(action)
                # record observations
                torch.cuda.empty_cache()
                obs = next_obs
                cur_scores.append(obs["score"])
                cur_actions.append(torch.FloatTensor(action))

            observations.append(obs["mesh"])
            scores.append(torch.stack(cur_scores).permute(1, 0))
            actions.append(torch.stack(cur_actions).permute(1, 0))

            print_score = (scores[-1][:, -1] / scores[-1][:, 0]).mean()
            print_reward = (
                (scores[-1][:, 0] - scores[-1][:, -1]) / scores[-1][:, 0]
            ).mean()

            temp_scored = torch.cat(scores)
            current_loss = (temp_scored[:, -1] / temp_scored[:, 0]).mean()
            message = f"Valid || score: {print_score:.4f} "
            message += f"reward = {print_reward:.4f} ave: {100 * current_loss:.4f} %"
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
            print("visualizing")
            actions = torch.stack(actions).view(-1, self.args.budget)
            utils.visualize_actions(self.results_dir, actions, self.args)

        print("*" * 30)
        message = f"Total Valid || score: {current_loss:.4f}, "
        message += f"reward = {rewards.mean():.4f}"
        tqdm.write("*" * len(message))
        tqdm.write(message)
        tqdm.write("*" * len(message))


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
        default=os.path.dirname(pretrained.__file__) + "/reconstruction/touch/best",
        help="the location of the touch part prediction.",
    )
    parser.add_argument(
        "--vision_location",
        type=str,
        default=os.path.dirname(pretrained.__file__) + "/reconstruction/vision/t_p/",
        help="the location of the vision part prediction.",
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
        "--eval", type=bool, default=True, help="for evaluating on test set"
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
        "--pretrained_recon",
        action="store_true",
        default=False,
        help="use the pretrained reconstruction models to train",
    )

    args = parser.parse_args()

    trainer = Engine(args)
    trainer()
