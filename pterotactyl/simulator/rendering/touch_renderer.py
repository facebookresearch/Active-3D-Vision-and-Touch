# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import pyrender

from pterotactyl.utility import utils


class Renderer:
    def __init__(self, cameraResolution=[120, 160]):
        self.scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
        self.object_nodes = []
        self.initialize_camera()
        self.r = pyrender.OffscreenRenderer(cameraResolution[0], cameraResolution[1])

    def initialize_camera(self):
        camera = pyrender.PerspectiveCamera(
            yfov=40.0 / 180.0 * np.pi, znear=0.0001, zfar=10.0
        )
        self.camera_pose = utils.euler2matrix(
            xyz="xyz", angles=[0, 0, 0], translation=[0, 0, 0], degrees=True
        )

        # Add camera node into scene
        camera_node = pyrender.Node(camera=camera, matrix=self.camera_pose)
        self.scene.add_node(camera_node)
        self.camera = camera_node

    def add_object(self, objTrimesh, position=[0, 0, 0], orientation=[0, 0, 0]):
        mesh = pyrender.Mesh.from_trimesh(objTrimesh)
        pose = utils.euler2matrix(angles=orientation, translation=position)
        objNode = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(objNode)
        self.object_nodes.append(objNode)

    def update_objects_pose(self, position, orientation):
        pose = utils.euler2matrix(angles=orientation, translation=position)
        for obj in self.object_nodes:
            self.scene.set_pose(obj, pose=pose)

    def remove_objects(self):
        for obj in self.object_nodes:
            self.scene.remove_node(obj)
        self.object_nodes = []

    def update_camera_pose(self, position, orientation):
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = orientation
        self.camera.matrix = pose.dot(self.camera_pose)

    def render(self):
        self.scene.main_camera_node = self.camera
        _, depth = self.r.render(self.scene)
        return depth
