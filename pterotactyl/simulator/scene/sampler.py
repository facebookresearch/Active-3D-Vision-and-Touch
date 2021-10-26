#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.
import os

import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet as pb
import pybullet_data
import torch

from pterotactyl.simulator.scene import instance
from pterotactyl.utility import utils


class Sampler:
    def __init__(
        self,
        grasp_class,
        bs=1,
        vision=True,
        max_depth=0.025,
        object_colours=[228, 217, 111, 255],
        resolution=[256, 256],
        TACTO=False,
    ):
        self.pybullet_connections = []
        self.pybullet_scenes = []
        self.bs = bs
        self.vision = vision

        # make a connection for every element in the batch
        for i in range(bs):
            self.pybullet_connections.append(bc.BulletClient(connection_mode=pb.DIRECT))
            self.pybullet_connections[i].setAdditionalSearchPath(
                pybullet_data.getDataPath()
            )

            if np.array(object_colours).shape == (4,):
                colour = object_colours
            else:
                colour = object_colours[i]
            self.pybullet_scenes.append(
                instance.Scene(
                    grasp_class,
                    max_depth=max_depth,
                    conn=self.pybullet_connections[i],
                    vision=self.vision,
                    object_colour=colour,
                    resolution=resolution,
                    TACTO=TACTO,
                )
            )

    # disconnets the pybullet threads
    def disconnect(self):
        for i in range(self.bs):
            self.pybullet_connections[i].disconnect()

    # loads the objects into each pybullet thread
    def load_objects(self, batch, from_dataset=True, scale=3.1):
        self.remove_objects()
        assert len(batch) == self.bs
        for i in range(self.bs):
            obj_location = batch[i]
            # if the object information has already been extracted
            if from_dataset:
                verts = np.load(obj_location + "_verts.npy")
                faces = np.load(obj_location + "_faces.npy")
                faces = utils.add_faces(faces)
                urdf_location = obj_location + ".urdf"
            # extract and record the object information
            else:
                obj_location = obj_location + ".obj"
                urdf_location = obj_location + ".urdf"
                verts, faces = utils.get_obj_data(obj_location, scale=scale)
                utils.make_urdf(verts, faces, urdf_location)

            self.pybullet_scenes[i].load_obj(verts, faces, urdf_location)

    def remove_objects(self):
        for i in range(self.bs):
            self.pybullet_scenes[i].remove_obj()

    def grasp(self, i, actions):
        return self.pybullet_scenes[i].grasp(actions[i])

    # perfrom the grasp and extracted the requested information
    def sample(
        self,
        actions,
        touch=True,
        touch_point_cloud=False,
        vision=False,
        vision_occluded=False,
        parameters=None,
    ):
        success = []
        poses = []
        dict = {}

        # check if the grasps are feasible
        for i in range(self.bs):
            # perfrom the grasps
            success.append(self.grasp(i, actions))
            if success[-1]:
                poses.append(self.pybullet_scenes[i].get_hand_pose())
            else:
                poses.append(None)
        dict["hand_pose"] = poses

        # get touch signal from grasp
        if touch:
            touch_status = [
                ["no_intersection" for _ in range(4)] for _ in range(self.bs)
            ]
            touch_signal = torch.zeros((self.bs, 4, 121, 121, 3))
            depths = torch.zeros((self.bs, 4, 121, 121))
            finger_transform_pos = torch.zeros((self.bs, 4, 3))
            finger_transform_rot_M = torch.zeros((self.bs, 4, 3, 3))

            for i in range(self.bs):
                if success[i]:
                    # depth from camera
                    touch_status[i] = self.pybullet_scenes[i].render_depth()
                    # simulated touch from depth
                    touch = self.pybullet_scenes[i].render_touch()
                    ref_frame = self.pybullet_scenes[i].get_finger_frame()
                    touch_signal[i] = torch.FloatTensor(touch)
                    depths[i] = torch.FloatTensor(self.pybullet_scenes[i].depths)
                    finger_transform_pos[i] = torch.FloatTensor(ref_frame["pos"])
                    finger_transform_rot_M[i] = torch.FloatTensor(ref_frame["rot_M"])
            dict["touch_status"] = touch_status
            dict["touch_signal"] = touch_signal
            dict["depths"] = depths
            dict["finger_transfrom_pos"] = finger_transform_pos
            dict["finger_transform_rot_M"] = finger_transform_rot_M

            # get pointcloud of touch site in the object frame of reference
            if touch_point_cloud:
                point_clouds = []
                for i in range(self.bs):
                    point_clouds.append(self.pybullet_scenes[i].depth_to_points())
                dict["touch_point_cloud"] = point_clouds

        # get image of the grasp
        if vision_occluded:
            vision_occluded_imgs = []
            for i in range(self.bs):
                if parameters is not None:
                    param = parameters[i]
                else:
                    param = None
                img = self.pybullet_scenes[i].scene_render(
                    occluded=True, parameters=param
                )
                vision_occluded_imgs.append(img)
                dict["vision_occluded"] = vision_occluded_imgs

        # get image of the object
        if vision:
            vision_imgs = []
            for i in range(self.bs):
                if parameters is not None:
                    param = parameters[i]
                else:
                    param = None
                img = self.pybullet_scenes[i].scene_render(
                    occluded=False, parameters=param
                )
                vision_imgs.append(img)
                dict["vision"] = vision_imgs

        return dict
