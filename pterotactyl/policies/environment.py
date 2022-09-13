# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import numpy as np
import torch
import torch.utils.data

from pterotactyl.utility import utils
from pterotactyl.utility import data_loaders
from pterotactyl.reconstruction.touch import model as touch_model
from pterotactyl.reconstruction.vision import model as vision_model
from pterotactyl.reconstruction.autoencoder import model as auto_model
import pterotactyl.objects as objects
from pterotactyl.simulator.scene import sampler
from pterotactyl.simulator.physics import grasping
from pterotactyl import pretrained


class ActiveTouch:
    def __init__(self, args):
        self.args = args
        self.seed(self.args.seed)
        self.current_information = {}
        self.steps = 0
        self.touch_chart_location = os.path.join(
            os.path.dirname(objects.__file__), "touch_chart.obj"
        )
        self.vision_chart_location = os.path.join(
            os.path.dirname(objects.__file__), "vision_charts.obj"
        )
        self.pretrained_recon_models()
        self.setup_recon()
        self.get_loaders()
        self.sampler = sampler.Sampler(
            grasping.Agnostic_Grasp, bs=self.args.env_batch_size, vision=False
        )

    # Fix seeds
    def seed(self, seed):
        self.seed = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # get dataloaders
    def get_loaders(self):
        if not self.args.eval:
            self.train_data = data_loaders.mesh_loader_active(
                self.args, set_type="RL_train"
            )
            set_type = "valid"
        else:
            set_type = "test"
        self.valid_data = data_loaders.mesh_loader_active(self.args, set_type=set_type)

    def pretrained_recon_models(self):
        if self.args.pretrained_recon:
            self.args.touch_location = (
                os.path.dirname(pretrained.__file__) + "/reconstruction/touch/best/"
            )
            if self.args.use_img:
                if self.args.finger:
                    self.args.vision_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/v_t_p/"
                    )
                    self.args.auto_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/v_t_p/"
                    )
                else:
                    self.args.vision_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/v_t_g/"
                    )
                    self.args.auto_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/v_t_g/"
                    )
            else:
                if self.args.finger:
                    self.args.vision_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/t_p/"
                    )
                    self.args.auto_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/t_p/"
                    )
                else:
                    self.args.vision_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/t_g/"
                    )
                    self.args.auto_location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/t_g/"
                    )

    # initialize and load the correct reconstruction models
    def setup_recon(self):
        self.touch_verts, _ = utils.load_mesh_touch(self.touch_chart_location)

        # load predtrained touch prediction model
        touch_args, _ = utils.load_model_config(self.args.touch_location)
        weights = self.args.touch_location + '/model'
        self.touch_prediction = touch_model.Encoder().cuda()
        self.touch_prediction.load_state_dict(torch.load(weights))
        self.touch_prediction.eval()

        # load predtrained vision prediction model
        vision_args, _ = utils.load_model_config(self.args.vision_location)
        weights = self.args.vision_location + '/model'
        self.mesh_info, self.initial_mesh = utils.load_mesh_vision(
            vision_args, self.vision_chart_location
        )
        self.initial_mesh = self.initial_mesh.cuda()
        self.n_vision_charts = self.initial_mesh.shape[0]
        self.deform = vision_model.Deformation(
            self.mesh_info, self.initial_mesh, vision_args
        ).cuda()
        self.deform.load_state_dict(torch.load(weights))
        self.deform.eval()

        # load predtrained autoencoder model
        if self.args.use_latent:
            auto_args, _ = utils.load_model_config(self.args.auto_location)
            weights = self.args.auto_location + '/model'

            self.auto_encoder = auto_model.AutoEncoder(
                self.mesh_info, self.initial_mesh, auto_args, only_encode=True
            ).cuda()
            self.auto_encoder.load_state_dict(torch.load(weights), strict=False)
            self.auto_encoder.eval()

    # reset the environment with new objects
    def reset(self, batch):
        self.current_data = {}
        self.steps = 0
        self.current_data["first_score"] = None
        self.current_data["batch"] = batch
        self.current_data["mask"] = torch.zeros(
            [self.args.env_batch_size, self.args.num_actions]
        )
        self.sampler.load_objects(batch["names"], from_dataset=True)
        obs = self.compute_obs()
        self.current_data["score"] = obs["score"]
        return obs

    # take a set in the environment with supplied actions
    def step(self, actions):
        self.update_masks(actions)
        obs = self.compute_obs(actions=actions)
        reward = self.current_data["score"] - obs["score"]
        self.current_data["score"] = obs["score"]
        self.steps += 1
        done = self.steps == self.args.budget
        return obs, reward, done

    # compute the best myopic greedy actions and perfrom them
    def best_step(self, greedy_checks=None):

        best_actions = [None for _ in range(self.args.env_batch_size)]
        best_score = [1000 for _ in range(self.args.env_batch_size)]
        if greedy_checks == None or (
            greedy_checks is not None and greedy_checks >= self.args.num_actions
        ):
            for i in range(self.args.num_actions):
                actions = [i for _ in range(self.args.env_batch_size)]
                obs = self.compute_obs(actions)
                for e, s in enumerate(obs["score"]):
                    if s < best_score[e] and self.current_data["mask"][e][i] == 0:
                        best_actions[e] = actions[e]
                        best_score[e] = s
        else:

            possible_actions = [
                list(range(self.args.num_actions))
                for _ in range(self.args.env_batch_size)
            ]

            for i in range(self.args.env_batch_size):
                seen = torch.where(self.current_data["mask"][i] != 0)[0]
                actions = list(seen.data.cpu().numpy())
                actions.sort()
                actions.reverse()
                for action in actions:
                    del possible_actions[i][action]
            checks = min(greedy_checks, len(possible_actions[0]))
            selected_actions = [
                random.sample(possible_actions[i], checks)
                for i in range(self.args.env_batch_size)
            ]
            for i in range(checks):
                actions = [
                    selected_actions[j][i] for j in range(self.args.env_batch_size)
                ]
                obs = self.compute_obs(actions)
                for e, s in enumerate(obs["score"]):
                    if s < best_score[e]:
                        best_actions[e] = actions[e]
                        best_score[e] = s

        actions = np.array(best_actions)
        obs, reward, done = self.step(actions)

        return actions, obs, reward, done

    # check the result of perfroming a specific action
    def check_step(self, actions):
        obs = self.compute_obs(actions=actions)
        return obs

    # perfrom a given action and compute the new state observations
    def compute_obs(self, actions=None):
        with torch.no_grad():
            charts = self.get_inputs(actions)
            img = self.current_data["batch"]["img"].cuda()
            verts, mask = self.deform(img, charts)
            if self.args.use_latent:
                latent = self.auto_encoder(verts.detach(), mask)
            score = self.get_score(
                verts, self.current_data["batch"]["gt_points"].cuda()
            )

        if self.current_data["first_score"] is None:
            self.current_data["first_score"] = score
            if self.args.use_latent:
                self.current_data["first_latent"] = latent.data.cpu()

        mesh = torch.cat((verts, mask), dim=-1).data.cpu()
        obs = {
            "score": score.data.cpu().clone(),
            "first_score": self.current_data["first_score"].clone(),
            "mask": self.current_data["mask"].data.cpu().clone(),
            "names": self.current_data["batch"]["names"],
            "mesh": mesh.data.cpu().clone(),
        }

        if self.args.use_latent:
            obs["first_latent"] = self.current_data["first_latent"]
            obs["latent"] = latent.data.cpu()
        return obs

    # compute the Chamfer distance of object predictions
    def get_score(self, verts, gt_points):
        loss = utils.chamfer_distance(
            verts, self.mesh_info["faces"], gt_points, num=self.args.number_points
        )
        loss = self.args.loss_coeff * loss
        return loss.cpu()

    # perform a given action and a convert the resulting signals into expected input for the reconstructor
    def get_inputs(self, actions=None):
        num_fingers = 1 if self.args.finger else 4
        # this occurs if a reset is being perfromed
        # here the input is defined with not touch information
        if actions is None:
            self.touch_charts = torch.zeros(
                (self.args.env_batch_size, num_fingers, self.args.num_grasps, 25, 3)
            ).cuda()
            self.touch_masks = torch.zeros(
                (self.args.env_batch_size, num_fingers, self.args.num_grasps, 25, 1)
            ).cuda()
            self.vision_charts = self.initial_mesh.unsqueeze(0).repeat(
                self.args.env_batch_size, 1, 1
            )
            self.vision_masks = 3 * torch.ones(
                self.vision_charts.shape[:-1]
            ).cuda().unsqueeze(-1)
        else:
            # perfrom the action
            signals = self.sampler.sample(actions, touch_point_cloud=True)

            if self.args.finger:
                touch = (
                    torch.FloatTensor(
                        signals["touch_signal"].data.numpy().astype(np.uint8)
                    )[:, 1]
                    .permute(0, 3, 1, 2)
                    .cuda()
                    / 255.0
                )
                pos = signals["finger_transfrom_pos"][:, 1].cuda()
                rot = signals["finger_transform_rot_M"][:, 1].cuda()

                ref_frame = {"pos": pos, "rot": rot}
                # convert the touch signals to charts
                touch_verts = (
                    self.touch_verts.unsqueeze(0)
                    .repeat(self.args.env_batch_size, 1, 1)
                    .cuda()
                )
                pred_touch_charts = self.touch_prediction(
                    touch, ref_frame, touch_verts
                ).contiguous()
                # define the touch charts in the input mesh to the reconstructor
                for i in range(self.args.env_batch_size):
                    if signals["touch_status"][i][1] == "touch":
                        self.touch_charts[i, 0, self.steps] = pred_touch_charts[i]
                        self.touch_masks[i, 0, self.steps] = 2
                    elif signals["touch_status"][i][1] == "no_touch":
                        self.touch_charts[i, 0, self.steps] = (
                            pos[i].view(1, 1, 3).repeat(1, 25, 1)
                        )
                        self.touch_masks[i, 0, self.steps] = 1
                    else:
                        self.touch_charts[i, 0, self.steps] = 0
                        self.touch_masks[i, 0, self.steps] = 0

            else:

                touch = (
                    signals["touch_signal"]
                    .view(-1, 121, 121, 3)
                    .permute(0, 3, 1, 2)
                    .cuda()
                    / 255.0
                )
                pos = signals["finger_transfrom_pos"].view(-1, 3).cuda()
                rot = signals["finger_transform_rot_M"].view(-1, 3, 3).cuda()
                ref_frame = {"pos": pos, "rot": rot}
                # convert the touch signals to charts
                touch_verts = (
                    self.touch_verts.unsqueeze(0)
                    .repeat(self.args.env_batch_size * 4, 1, 1)
                    .cuda()
                )
                pred_touch_charts = self.touch_prediction(
                    touch, ref_frame, touch_verts
                ).contiguous()
                # define the touch charts in the input mesh to the reconstructor
                for i in range(self.args.env_batch_size):
                    for j in range(4):
                        if signals["touch_status"][i][j] == "touch":
                            self.touch_charts[i, j, self.steps] = pred_touch_charts[
                                i * 4 + j
                            ]
                            self.touch_masks[i, j, self.steps] = 2
                        elif signals["touch_status"][i][j] == "no_touch":
                            self.touch_charts[i, j, self.steps] = (
                                pos[i * 4 + j].view(1, 1, 3).repeat(1, 25, 1)
                            )
                            self.touch_masks[i, j, self.steps] = 1
                        else:
                            self.touch_charts[i, j, self.steps] = 0
                            self.touch_masks[i, j, self.steps] = 0

        charts = {
            "touch_charts": self.touch_charts.view(
                self.args.env_batch_size, num_fingers * 5 * 25, 3
            ).clone(),
            "vision_charts": self.vision_charts.clone(),
            "touch_masks": self.touch_masks.view(
                self.args.env_batch_size, num_fingers * 5 * 25, 1
            ).clone(),
            "vision_masks": self.vision_masks.clone(),
        }
        return charts

    # this is perfromed due to a meoery leak in pybullet where loaded meshes are not properly deleted
    def reset_pybullet(self):
        self.sampler.disconnect()
        del self.sampler
        self.sampler = sampler.Sampler(
            grasping.Agnostic_Grasp, bs=self.args.env_batch_size, vision=True
        )

    # update the set of action which have been performed
    def update_masks(self, actions):
        for i in range(actions.shape[0]):
            self.current_data["mask"][i, actions[i]] = 1
