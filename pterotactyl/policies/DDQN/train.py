# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import argparse
from submitit.helpers import Checkpointable
from tqdm import tqdm


from pterotactyl.policies.DDQN import ddqn
from pterotactyl.policies import environment
from pterotactyl.policies import replay
from pterotactyl.utility import utils
from pterotactyl import pretrained


# module for training the DDQN models
class Engine(Checkpointable):
    def __init__(self, args):
        self.args = args
        self.steps = 0
        self.episode = 0
        self.epoch = 0
        self.cur_loss = 10000
        self.best_loss = 10000
        self.epsilon = self.args.epsilon_start

        self.results_dir = os.path.join("results", args.exp_type, args.exp_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.checkpoint_dir = os.path.join(
            "experiments/checkpoint/", self.args.exp_type, self.args.exp_id
        )
        if not os.path.exists((self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        utils.save_config(self.checkpoint_dir, args)

    def __call__(self):
        # initialize the learning environment
        self.env = environment.ActiveTouch(self.args)
        self.replay_memory = replay.ReplayMemory(self.args)
        self.policy = ddqn.DDQN(self.args, self.env.mesh_info, self.replay_memory)
        self.target_net = ddqn.DDQN(self.args, self.env.mesh_info, None)

        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()

        self.writer = SummaryWriter(
            os.path.join("experiments/tensorboard/", self.args.exp_type)
        )
        self.window_size = 1000
        self.ave_reward = torch.zeros((self.window_size)).cuda()
        self.ave_recon = torch.zeros((self.window_size)).cuda()
        train_loader, valid_loaders = self.get_loaders()

        if self.args.eval:
            self.load(best=True)
            self.validate(valid_loaders)
            return

        self.resume()
        # training loop
        for epoch in range(self.epoch, self.args.epochs):
            self.train(train_loader)
            self.env.reset_pybullet()
            if self.steps >= self.args.burn_in:
                with torch.no_grad():
                    self.validate(valid_loaders)
                self.env.reset_pybullet()
                self.check_values_and_save()
            self.epoch += 1

    # load the environment data into pytorch dataloaders
    def get_loaders(self):
        if self.args.eval:
            train_loader = ""
        else:
            train_loader = DataLoader(
                self.env.train_data,
                batch_size=self.args.env_batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=self.env.train_data.collate,
            )

        valid_loader = DataLoader(
            self.env.valid_data,
            batch_size=self.args.env_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.env.valid_data.collate,
        )
        return train_loader, valid_loader

    # training iteration
    def train(self, dataloader):
        for v, batch in enumerate(tqdm(dataloader, total=self.args.train_steps)):
            if v > self.args.train_steps - 1:
                break

            obs = self.env.reset(batch)

            all_done = False
            total_reward = 0

            while not all_done:
                # update epsilon

                if self.steps >= self.args.burn_in:
                    self.epsilon = self.policy.update_epsilon(self.epsilon, self.args)

                # get action
                get_random_action = self.steps < self.args.burn_in
                action = self.policy.get_action(
                    obs, eps_threshold=self.epsilon, give_random=get_random_action
                )

                # perform action
                with torch.no_grad():
                    next_obs, reward, all_done = self.env.step(action)

                # save experiance
                self.policy.add_experience(action, obs, next_obs, reward)

                # update policy
                if self.steps >= self.args.burn_in:
                    self.policy.update_parameters(self.target_net)

                # update target network
                if (
                    self.steps % self.args.target_update == 0
                    and self.steps >= self.args.burn_in
                ):
                    print("+" * 5 + " updating target " "+" * 5)
                    self.target_net.load_state_dict(self.policy.state_dict())
                torch.cuda.empty_cache()
                obs = next_obs
                self.steps += 1

            # logs
            recon = float((obs["score"] / obs["first_score"]).mean().item())
            reward = float(
                ((obs["first_score"] - obs["score"]) / obs["first_score"]).mean().item()
            )
            self.ave_reward[self.episode % self.window_size] = reward
            self.ave_recon[self.episode % self.window_size] = float(
                (obs["score"] / obs["first_score"]).mean().item()
            )
            ave_reward = self.ave_reward[: self.episode + 1].mean()
            ave_recon = self.ave_recon[: self.episode + 1].mean()
            message = (
                f"T Epoch: {self.epoch} Ep: {self.episode}, recon: {recon:.2f}, "
                f"reward: {reward:.2f}, a_recon: {ave_recon:.2f}, a_reward: {ave_reward:.2f}, "
                f" eps: {self.epsilon:.3f}, best: {self.best_loss:.3f}"
            )
            tqdm.write(message)
            self.episode += 1

        # logs
        if self.steps >= self.args.burn_in:
            self.writer.add_scalars(
                "train_recon_|_", {self.args.exp_id: ave_recon}, self.steps
            )
            self.writer.add_scalars(
                "train_reward_|_", {self.args.exp_id: ave_reward}, self.steps
            )

    # validation iteration
    def validate(self, dataloader):
        observations = []
        scores = []
        actions = []
        names = []
        print("*" * 30)
        print("Doing Validation")
        total = self.args.train_steps if not self.args.eval else None
        for v, batch in enumerate(tqdm(dataloader, total=total)):
            names += batch["names"]
            if v > self.args.valid_steps - 1 and not self.args.eval:
                break

            obs = self.env.reset(batch)

            all_done = False
            cur_scores = [obs["score"]]
            cur_actions = []

            while not all_done:
                # select actions
                action = self.policy.get_action(
                    obs, eps_threshold=-1, give_random=False
                )

                # perform actions

                with torch.no_grad():
                    next_obs, reward, all_done = self.env.step(action)

                # record actions
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

            message = f"Valid || E: {self.epoch}, score: {print_score:.2f}, best score: {self.best_loss:.2f} "
            message += f"reward = {print_reward:.2f}"
            tqdm.write(message)
            if self.args.visualize and v == 5 and self.args.eval:
                meshes = torch.cat(observations, dim=0)[:, :, :3]
                utils.visualize_prediction(
                    self.results_dir, meshes, self.env.mesh_info["faces"], names
                )
                self.env.reset_pybullet()

        scores = torch.cat(scores)
        actions = torch.cat(actions)
        rewards = ((scores[:, 0] - scores[:, -1]) / scores[:, 0]).mean()
        variation = torch.std(actions, dim=0).mean()
        self.current_loss = (scores[:, -1] / scores[:, 0]).mean()

        print("*" * 30)
        message = f"Total Valid || E: {self.epoch}, score: {self.current_loss:.4f}, best score: {self.best_loss:.4f} "
        message += f"reward = {rewards.mean():.2f}"
        tqdm.write("*" * len(message))
        tqdm.write(message)
        tqdm.write("*" * len(message))

        if not self.args.eval:
            self.writer.add_scalars(
                f"Valid_recon_|_", {self.args.exp_id: self.current_loss}, self.steps
            )
            self.writer.add_scalars(
                f"Valid_reward_|_", {self.args.exp_id: rewards.mean()}, self.steps
            )
            self.writer.add_scalars(
                "epsilon_|_", {self.args.exp_id: self.epsilon}, self.steps
            )
            self.writer.add_scalars(
                f"Valid_variation_|_", {self.args.exp_id: variation}, self.steps
            )
        if self.args.visualize and self.args.eval:
            utils.visualize_actions(self.results_dir, actions, self.args)

    # check if the new validation score if better and save checkpoint
    def check_values_and_save(self):
        if self.best_loss >= self.current_loss:
            improvement = self.best_loss - self.current_loss
            print(
                f"Saving with {improvement:.3f} improvement in Chamfer Distance on Validation Set "
            )
            self.best_loss = self.current_loss
            self.save(best=True)

        print(f"Saving DQN checkpoint")
        self.save(best=False)
        print("Saving replay memory.")
        self.replay_memory.save(self.checkpoint_dir)

    # resume training
    def resume(self):
        path = self.checkpoint_dir + "/recent"
        if os.path.exists(path + "_model"):
            print(f"Loading DQN checkpoint")
            self.load(best=False)
            print("Loading replay memory.")
            self.replay_memory.load(path)

    # save current state of training
    def save(self, best=False):
        if best:
            path = self.checkpoint_dir + "/best"
        else:
            path = self.checkpoint_dir + "/recent"

        self.replay_memory.save(path)
        torch.save(
            {
                "dqn_weights": self.policy.state_dict(),
                "target_weights": self.target_net.state_dict(),
                "args": self.args,
                "episode": self.episode,
                "steps": self.steps,
                "ave_reward": self.ave_reward,
                "ave_recon": self.ave_recon,
                "epsilon": self.epsilon,
                "epoch": self.epoch,
            },
            path + "_model",
        )

    # load previous state of training
    def load(self, best=True):
        if self.args.pretrained:
            prefix = "l" if self.args.use_latent else "g"
            if self.args.use_img:
                if self.args.finger:
                    path = (
                        os.path.dirname(pretrained.__file__)
                        + f"/policies/DDQN/{prefix}_v_t_p"
                    )
                else:
                    path = (
                        os.path.dirname(pretrained.__file__)
                        + f"/policies/DDQN/{prefix}_v_t_g"
                    )
            else:
                if self.args.finger:
                    path = (
                        os.path.dirname(pretrained.__file__)
                        + f"/policies/DDQN/{prefix}_t_p"
                    )
                else:
                    path = (
                        os.path.dirname(pretrained.__file__)
                        + f"/policies/DDQN/{prefix}_t_g"
                    )
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint["dqn_weights"])
        else:
            if best:
                path = self.checkpoint_dir + "/best_model"
            else:
                path = self.checkpoint_dir + "/recent_model"
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint["dqn_weights"])
            self.episode = checkpoint["episode"] + 1
            if not self.args.eval:
                self.target_net.load_state_dict(checkpoint["target_weights"])
                self.steps = checkpoint["steps"]
                self.ave_reward = checkpoint["ave_reward"]
                self.ave_recon = checkpoint["ave_recon"]
                self.epsilon = checkpoint["epsilon"]
                self.epoch = checkpoint["epoch"] + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cut",
        type=float,
        default=0.33,
        help="The shared size of features in the GCN.",
    )
    parser.add_argument(
        "--layers", type=int, default=4, help="Number of layers in the q network"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=200,
        help="hidden dimension size in layers in the q network",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to use."
    )
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
        "--eval",
        action="store_true",
        default=False,
        help="Evaluate the trained model on the test set.",
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
        "--seed", type=int, default=0, help="Setting for the random seed."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="Initial learning rate."
    )
    parser.add_argument(
        "--env_batch_size",
        type=int,
        default=3,
        help="Size of the batch of objects sampled from the environment",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Size of the batch of transitions sampled for training the q network.",
    )
    parser.add_argument(
        "--exp_id", type=str, default="test", help="The experiment name."
    )
    parser.add_argument(
        "--exp_type", type=str, default="test", help="The experiment group."
    )
    parser.add_argument(
        "--use_img", action="store_true", default=False, help="To use the image."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=70,
        help="How many epochs without imporvement before training stops.",
    )
    parser.add_argument(
        "--loss_coeff", type=float, default=9000.0, help="Coefficient for loss term."
    )

    parser.add_argument(
        "--num_grasps", type=int, default=5, help="Number of grasps to train with. "
    )
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["first", "current", "none"],
        default="first",
        help="how to normalize the reward for the q network update ",
    )
    parser.add_argument(
        "--mem_capacity", type=int, default=300, help="the size of the replay buffer"
    )
    parser.add_argument("--burn_in", type=int, default=20, help="ddqn burn in time")
    parser.add_argument(
        "--num_actions", type=int, default=50, help=" number of possible actions"
    )
    parser.add_argument("--gamma", type=float, default=0, help="ddqn gamma value")
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="ddqn initial epsilon value"
    )
    parser.add_argument(
        "--epsilon_decay", type=float, default=0.9999, help="ddqn epsilon decay value"
    )
    parser.add_argument(
        "--epsilon_end", type=float, default=0.01, help="ddqn minimum epsilon value"
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=20,
        help="number of training iterations per epoch",
    )
    parser.add_argument(
        "--valid_steps",
        type=int,
        default=10,
        help="number of validation iterations per epoch",
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=3000,
        help="frequency of target network updates",
    )
    parser.add_argument(
        "--use_latent",
        action="store_true",
        default=False,
        help="if the latent embedding of objects is to be used",
    )
    parser.add_argument(
        "--use_recon",
        action="store_true",
        default=False,
        help="if the object prediction is to be directly  used",
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

    trainer = Engine(args)
    trainer()
