# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import argparse
from collections import namedtuple

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from pterotactyl.policies import environment
from pterotactyl.policies.baselines import baselines
from pterotactyl.policies.supervised import model as learning_model
from pterotactyl.utility import utils
from pterotactyl import pretrained


class Engine:
    def __init__(self, args):
        self.args = args
        self.results_dir = os.path.join("results", args.exp_type, args.exp_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.checkpoint_dir = os.path.join(
            "experiments/checkpoint/", args.exp_type, args.exp_id
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not self.args.eval:
            utils.save_config(self.checkpoint_dir, args)

    def __call__(self):
        # setup the environment, policy and data
        self.env = environment.ActiveTouch(self.args)
        self.policy = baselines.even_sampler(self.args)
        train_loaders, valid_loaders = self.get_loaders()
        self.step = 0
        self.models = [
            learning_model.Latent_Model(self.args).cuda()
            for i in range(self.args.budget)
        ]

        # logging information
        writer = SummaryWriter(
            os.path.join("experiments/tensorboard/", self.args.exp_type)
        )

        # evaluate the policy
        if self.args.eval:
            with torch.no_grad():
                self.load(train=False)
                self.step = self.args.budget - 1
                self.validate(valid_loaders, writer)
                return

        else:
            for i in range(self.args.budget):
                params = list(self.models[i].parameters())
                self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)
                self.load(train=True)
                for model in self.models:
                    model.eval()
                self.epoch = 0
                self.best_loss = 10000
                self.last_improvement = 0

                for j in range(self.args.epoch):
                    self.train(train_loaders, writer)
                    with torch.no_grad():
                        self.validate(valid_loaders, writer)
                    if self.check_values():
                        break
                    self.epoch += 1
                self.step += 1

    # load data using pytorch dataloader
    def get_loaders(self):
        if not self.args.eval:
            train_loader = DataLoader(
                self.env.train_data,
                batch_size=self.args.env_batch_size,
                shuffle=True,
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

    def train(self, dataloader, writer):
        total_loss = 0
        iterations = 0.0
        self.models[self.step].train()
        for v, batch in enumerate(
            tqdm(dataloader, total=min(self.args.train_steps, len(dataloader)))
        ):
            if v >= self.args.train_steps:
                break
            try:
                obs = self.env.reset(batch)
            except:
                continue

            # move to the correct step
            with torch.no_grad():
                cur_actions = []
                for i in range(self.step):
                    values = self.models[i](obs)
                    for acts in cur_actions:
                        for e, act in enumerate(acts):
                            values[e][act] = 1e10
                    actions = torch.argmin(values, dim=1)
                    next_obs, reward, all_done = self.env.step(actions)
                    obs = next_obs
                    cur_actions.append(actions)

            # predict action values
            all_pred_values = self.models[self.step](obs)
            pred_values = []

            # sample some random actions and compute their value
            random_actions = np.random.randint(
                50, size=self.args.env_batch_size * 5
            ).reshape(5, self.args.env_batch_size)
            target = []
            for actions in random_actions:
                temp_obs = self.env.check_step(actions)
                if self.args.normalize:
                    score = (temp_obs["first_score"] - temp_obs["score"]) / temp_obs[
                        "first_score"
                    ]
                else:
                    score = temp_obs["first_score"] - temp_obs["score"]

                cur_pred_values = []
                for j, a in enumerate(actions):
                    cur_pred_values.append(all_pred_values[j, a])
                pred_values.append(torch.stack(cur_pred_values))
                target.append(score)
            target = torch.stack(target).cuda()
            pred_values = torch.stack(pred_values)
            loss = ((target - pred_values) ** 2).mean()

            # backprop
            loss.backward()
            self.optimizer.step()

            # log
            message = f"Train || step {self.step + 1 } || Epoch: {self.epoch}, loss: {loss.item():.3f}, b_ptp:  {self.best_loss:.3f}"
            tqdm.write(message)
            total_loss += loss.item()
            iterations += 1.0
        self.train_loss = total_loss / iterations
        writer.add_scalars(
            f"train_loss_{self.step}",
            {self.args.exp_id: total_loss / iterations},
            self.epoch,
        )

    # perfrom the validation
    def validate(self, dataloader, writer):
        observations = []
        scores = []
        actions = []
        names = []
        self.models[self.step].eval()
        valid_length = int(len(dataloader) * 0.2)
        for v, batch in enumerate(tqdm(dataloader)):
            names += batch["names"]
            try:
                obs = self.env.reset(batch)
            except:
                continue
            self.policy.reset()
            cur_scores = [obs["score"]]
            cur_actions = []
            for i in range(self.step + 1):
                action_values = self.models[i](obs)
                for acts in cur_actions:
                    for e, act in enumerate(acts):
                        action_values[e][act] = 1e10
                action = torch.argmin(action_values, dim=1)
                next_obs, _, _ = self.env.step(action)

                # record observation
                obs = next_obs
                cur_scores.append(obs["score"])
                cur_actions.append(action.data.cpu())

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

            if self.args.visualize and v == 5 and self.args.eval:
                meshes = torch.cat(observations, dim=0)[:, :, :3]
                utils.visualize_prediction(
                    self.results_dir, meshes, self.env.mesh_info["faces"], names
                )
                self.env.reset_pybullet()

        scores = torch.cat(scores)
        rewards = ((scores[:, 0] - scores[:, -1]) / scores[:, 0]).mean()
        current_loss = (scores[:, -1] / scores[:, 0]).mean()
        self.current_loss = current_loss

        print("*" * 30)
        message = f"Total Valid || step {self.step + 1 } || score: {current_loss:.4f}, "
        message += f"reward = {rewards.mean():.4f}"
        tqdm.write("*" * len(message))
        tqdm.write(message)
        tqdm.write("*" * len(message))

        if self.args.visualize and self.args.eval:
            actions = torch.stack(actions).view(-1, self.args.budget)
            utils.visualize_actions(self.results_dir, actions, self.args)

        if not self.args.eval:
            writer.add_scalars(
                f"valid_loss_{self.step}", {self.args.exp_id: current_loss}, self.epoch
            )

    def check_values(self):
        if self.best_loss >= self.current_loss:
            improvement = self.best_loss - self.current_loss
            print(f"Saving with {improvement:.3f} improvement on Validation Set ")
            self.best_loss = self.current_loss
            self.last_improvement = 0
            self.save()
            return False
        else:
            self.last_improvement += 1
            if self.last_improvement >= self.args.patience:
                print(f"Over {self.args.patience} steps since last imporvement")
                print("Moving to next step or exiting")
                return True

    def load(self, train=False):
        if self.args.pretrained:
            if self.args.use_img:
                if self.args.finger:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/supervised/v_t_p"
                    )
                else:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/supervised/v_t_g"
                    )
            else:
                if self.args.finger:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/supervised/t_p"
                    )
                else:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/policies/supervised/t_g"
                    )
            config_location = f"{location}/config.json"
            with open(config_location) as json_file:
                data = json.load(json_file)
                data["auto_location"] = self.args.auto_location
                data["eval"] = True
                data["visualize"] = self.args.visualize
            self.args = namedtuple("ObjectName", data.keys())(*data.values())
            self.models = [
                learning_model.Latent_Model(self.args).cuda()
                for i in range(self.args.budget)
            ]
            for i in range(self.args.budget):
                self.models[i].load_state_dict(torch.load(location + f"/model_{i}"))

        else:
            if train:
                for i in range(self.step):
                    self.models[i].load_state_dict(
                        torch.load(self.checkpoint_dir + f"/model_{i}")
                    )
            else:
                for i in range(self.args.budget):
                    self.models[i].load_state_dict(
                        torch.load(self.checkpoint_dir + f"/model_{i}")
                    )

    def save(self):
        torch.save(
            self.models[self.step].state_dict(),
            self.checkpoint_dir + f"/model_{self.step}",
        )


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
        "--auto_location",
        type=str,
        default=os.path.dirname(pretrained.__file__) + "/reconstruction/auto/t_p/",
        help="the location of the autoencoder part prediction.",
    )
    parser.add_argument(
        "--number_points",
        type=int,
        default=30000,
        help="number of points sampled for the chamfer distance.",
    )
    parser.add_argument(
        "--epoch", type=int, default=3000, help="number of epochs per step"
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
    parser.add_argument(
        "--num_actions", type=int, default=50, help="number of action options"
    )

    parser.add_argument(
        "--eval", action="store_true", default=False, help="for evaluating on test set"
    )
    parser.add_argument(
        "--budget", type=int, default=5, help="number of graspsp to perform"
    )
    parser.add_argument(
        "--exp_id", type=str, default="test", help="The experiment name."
    )
    parser.add_argument(
        "--exp_type", type=str, default="test", help="The experiment type."
    )
    parser.add_argument(
        "--layers", type=int, default=4, help="Number of layers in the q network"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="number of epochs without progress before stopping",
    )
    parser.add_argument(
        "--training_actions",
        type=int,
        default=5,
        help="number of action values learned for each object in each iteration",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=200,
        help="hidden dimension size in layers in the q network",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="number of training steps per epoch",
    )
    parser.add_argument(
        "--normalize", type=int, default=0, help="number of training steps per epoch"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="visualize predictions and actions while evaluating",
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
    args.use_recon = False
    args.use_latent = True

    trainer = Engine(args)
    trainer()
