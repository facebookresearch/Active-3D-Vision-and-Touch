# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrender
import trimesh

import pterotactyl.objects as objects
from pterotactyl.utility import utils
from random import randrange

HAND_COLOUR = [119, 136, 153, 255]
DIGIT_COLOUR = [119, 225, 153, 175]


class Renderer:
    def __init__(self, hand, pb, cameraResolution=[256, 256]):
        self.scene = self.init_scene()
        self.hand = hand
        self.pb = pb
        self.hand_nodes = []
        self.object_nodes = []

        self.init_camera()
        self.init_hand()
        self.update_hand()
        self.r = pyrender.OffscreenRenderer(cameraResolution[0], cameraResolution[1])

    # scene is initialized with fixed lights, this can be easily changed to match the desired environment
    def init_scene(self):
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[0, -0.8, 0.3], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
        scene.add(light, pose=light_pose)

        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[0, 0.8, 0.3], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
        scene.add(light, pose=light_pose)

        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[-1, 0, 1], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
        scene.add(light, pose=light_pose)

        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[1, 0, 1], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
        scene.add(light, pose=light_pose)
        return scene

    def init_camera(self):
        # initializes the camera parameters
        camera = pyrender.PerspectiveCamera(
            yfov=60.0 / 180.0 * np.pi, znear=0.01, zfar=10.0, aspectRatio=1.0
        )
        camera_pose = utils.euler2matrix(
            xyz="xyz", angles=[0, 0, 0], translation=[0, 0, 0], degrees=True
        )
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(camera_node)
        self.scene.main_camera_node = camera_node
        self.camera = camera_node
        # this viewpoint is used in the paper
        # if you change this, you will need to update the camaera parameter matrix in the reconstruction model as well
        initial_matrix = R.from_euler("xyz", [45.0, 0, 270.0], degrees=True).as_matrix()
        self.update_camera_pose([-0.3, 0, 0.3], initial_matrix)

    def add_object(
        self,
        mesh,
        position=[0, 0, 0],
        orientation=[0, 0, 0],
        colour=[228, 217, 111, 255],
    ):
        mesh.visual.vertex_colors = colour
        mesh = pyrender.Mesh.from_trimesh(mesh)
        pose = utils.euler2matrix(angles=orientation, translation=position)
        obj_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(obj_node)
        self.object_nodes.append(obj_node)

    # defines the hand in the scene
    def init_hand(self):
        hand_location = os.path.join(
            os.path.dirname(objects.__file__), "hand/meshes_obj/"
        )
        base_obj = trimesh.load(hand_location + "0_base.obj")
        base_obj = trimesh.Trimesh(vertices=base_obj.vertices, faces=base_obj.faces)
        base_obj.visual.vertex_colors = HAND_COLOUR
        self.add_hand_obj(base_obj)
        for _ in range(3):
            for i in range(1, 5):
                element = trimesh.load(hand_location + f"{i}_finger.obj")
                element = trimesh.Trimesh(
                    vertices=element.vertices, faces=element.faces
                )
                element.visual.vertex_colors = HAND_COLOUR
                self.add_hand_obj(element)
            element = trimesh.load(hand_location + "5_digit.obj")
            element = trimesh.Trimesh(vertices=element.vertices, faces=element.faces)
            element.visual.vertex_colors = DIGIT_COLOUR
            self.add_hand_obj(element)

        for i in range(6, 10):
            element = trimesh.load(hand_location + f"{i}_thumb.obj")
            element = trimesh.Trimesh(vertices=element.vertices, faces=element.faces)
            element.visual.vertex_colors = HAND_COLOUR
            self.add_hand_obj(element)
        element = trimesh.load(hand_location + "5_digit.obj")
        element = trimesh.Trimesh(vertices=element.vertices, faces=element.faces)
        element.visual.vertex_colors = DIGIT_COLOUR
        self.add_hand_obj(element)

    def add_hand_obj(self, obj_location):
        mesh = pyrender.Mesh.from_trimesh(obj_location)
        pose = utils.euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0])
        obj_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(obj_node)
        self.hand_nodes.append(obj_node)

    # gets the various hand element's position and orientation and uses them to update the hand in the scene
    def update_hand(self):
        # base of the hand
        position, orientation = self.pb.getBasePositionAndOrientation(self.hand)
        orientation = self.pb.getEulerFromQuaternion(orientation)
        pose = utils.euler2matrix(angles=orientation, translation=position)
        self.scene.set_pose(self.hand_nodes[0], pose=pose)

        indices = [
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            9,
            10,
            11,
            14,
            15,
            16,
            17,
            18,
            21,
            22,
            23,
            24,
            25,
        ]
        # all other elements
        for node, index in zip(self.hand_nodes[1:], indices):
            position, orientation = self.pb.getLinkState(self.hand, index)[:2]
            orientation = self.pb.getEulerFromQuaternion(orientation)
            pose = utils.euler2matrix(angles=orientation, translation=position)
            self.scene.set_pose(node, pose=pose)

    # moves the hand our of the perspective of the camera
    def remove_hand(self):
        for node in self.hand_nodes:
            pose = utils.euler2matrix(angles=[0, 0, 0], translation=[0, 0, -10.0])
            self.scene.set_pose(node, pose=pose)

    def remove_objects(self):
        for obj in self.object_nodes:
            self.scene.remove_node(obj)
        self.object_nodes = []

    def update_camera_pose(self, position, orientation):
        pose = np.eye(4)
        if np.array(orientation).shape == (3,):
            orientation = R.from_euler("xyz", orientation, degrees=True).as_matrix()
        pose[:3, 3] = position
        pose[:3, :3] = orientation
        self.camera.matrix = pose

    def render(self, get_depth=False):
        colour, depth = self.r.render(self.scene)
        if get_depth:
            return colour, depth
        return colour
