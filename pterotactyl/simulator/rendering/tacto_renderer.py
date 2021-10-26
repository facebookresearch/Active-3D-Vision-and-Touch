# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import cv2
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R

from pterotactyl.utility import utils


class Renderer:
    def __init__(self, cameraResolution=[120, 160]):
        """
		:param width: scalar
		:param height: scalar
		"""
        self.width = cameraResolution[0]
        self.height = cameraResolution[1]

        self._background_real = None
        self.force_enabled = False
        self._init_pyrender()

    def _init_pyrender(self):
        """
		Initialize pyrender
		"""
        # Create scene for pybullet sync
        self.scene = pyrender.Scene()
        self.object_nodes = []

        self.current_light_nodes = []
        self.cam_light_ids = None

        self._init_gel()
        self._init_camera()
        self._init_light()

        self.r = pyrender.OffscreenRenderer(self.width, self.height)

        colors, depths = self.render(noise=False, calibration=False)

        self._background_sim = colors

    def _init_gel(self):
        """
		Add gel surface in the scene
		"""
        # Create gel surface (flat/curve surface based on config file)
        gel_trimesh = self._generate_gel_trimesh()

        mesh_gel = pyrender.Mesh.from_trimesh(gel_trimesh, smooth=False)
        self.gel_pose0 = np.eye(4)
        self.gel_node = pyrender.Node(mesh=mesh_gel, matrix=self.gel_pose0)
        self.scene.add_node(self.gel_node)

    def _generate_gel_trimesh(self):

        # Load config
        origin = [0.022, 0, 0.015]

        X0, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = 0.02, 0.03

        # Curved gel surface
        N = 100
        M = int(N * H / W)
        R = 0.1
        zrange = 0.005

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        h = R - np.maximum(0, R ** 2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
        xx = X0 - zrange * h / h.max()

        gel_trimesh = self._generate_trimesh_from_depth(xx)

        return gel_trimesh

    def _generate_trimesh_from_depth(self, depth):
        # Load config
        origin = [0.022, 0, 0.015]

        _, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = 0.02, 0.03

        N = depth.shape[1]
        M = depth.shape[0]

        # Create grid mesh
        vertices = []
        faces = []

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        # Vertex format: [x, y, z]
        vertices = np.zeros([N * M, 3])

        # Add x, y, z position to vertex
        vertices[:, 0] = depth.reshape([-1])
        vertices[:, 1] = yy.reshape([-1])
        vertices[:, 2] = zz.reshape([-1])

        # Create faces

        faces = np.zeros([(N - 1) * (M - 1) * 6], dtype=np.uint)

        # calculate id for each vertex: (i, j) => i * m + j
        xid = np.arange(N)
        yid = np.arange(M)
        yyid, xxid = np.meshgrid(xid, yid)
        ids = yyid[:-1, :-1].reshape([-1]) + xxid[:-1, :-1].reshape([-1]) * N

        # create upper triangle
        faces[::6] = ids  # (i, j)
        faces[1::6] = ids + N  # (i+1, j)
        faces[2::6] = ids + 1  # (i, j+1)

        # create lower triangle
        faces[3::6] = ids + 1  # (i, j+1)
        faces[4::6] = ids + N  # (i+1, j)
        faces[5::6] = ids + N + 1  # (i+1, j+1)

        faces = faces.reshape([-1, 3])

        # camera_pose = utils.euler2matrix(
        # 	angles=np.deg2rad([90, 0, -90]), translation=[0, 0, 0.015],
        # )
        vertices = vertices - np.array([0, 0, 0.015]).reshape(1, 3)
        orientation = R.from_euler("xyz", [90, 0, -90], degrees=True).as_matrix()
        vertices = vertices.dot(orientation)

        # position = [0, 0, 0.015]

        gel_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return gel_trimesh

    def _init_camera(self):
        """
		Set up camera
		"""

        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), znear=0.001)
        camera_pose = utils.euler2matrix(
            angles=np.deg2rad([0, 0, 0]), translation=[0, 0, -0.0035]
        )
        self.camera_pose = camera_pose

        # Add camera node into scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(camera_node)
        self.camera = camera_node

        self.cam_light_ids = list([0, 1, 2])

    def _init_light(self):
        """
		Set up light
		"""

        # Load light from config file
        origin = np.array([0.005, 0, 0.015])

        xyz = []
        # Apply polar coordinates
        thetas = [30, 150, 270]
        rs = [0.02, 0.02, 0.02]
        xs = [0, 0, 0]
        for i in range(len(thetas)):
            theta = np.pi / 180 * thetas[i]
            xyz.append([xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])

        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        intensities = [1, 1, 1]

        # Save light nodes
        self.light_nodes = []
        self.light_poses0 = []

        for i in range(len(colors)):
            color = colors[i]

            position = xyz[i] + origin - np.array([0, 0, 0.015])
            orientation = R.from_euler("xyz", [90, 0, -90], degrees=True).as_matrix()
            position = position.dot(orientation)

            orientation = np.deg2rad([90, 0, -90])

            light_pose_0 = utils.euler2matrix(angles=orientation, translation=position)

            light = pyrender.PointLight(color=color, intensity=intensities[i])
            light_node = pyrender.Node(light=light, matrix=light_pose_0)

            self.scene.add_node(light_node)
            self.light_nodes.append(light_node)
            self.light_poses0.append(light_pose_0)
            self.current_light_nodes.append(light_node)

    def add_object(self, objTrimesh, position=[0, 0, 0], orientation=[0, 0, 0]):

        mesh = trimesh.Trimesh(
            vertices=objTrimesh.vertices, faces=objTrimesh.faces, process=False
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        pose = utils.euler2matrix(angles=orientation, translation=position)

        objNode = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(objNode)
        self.object_nodes.append(objNode)

    def update_camera_pose(self, position, orientation):
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = orientation
        self.camera.matrix = pose.dot(self.camera_pose)

        # Update gel
        gel_pose = pose.dot(self.gel_pose0)
        self.gel_node.matrix = gel_pose

        # Update light
        for i in range(len(self.light_nodes)):
            light_pose = pose.dot(self.light_poses0[i])
            light_node = self.light_nodes[i]
            light_node.matrix = light_pose

    def update_objects_pose(self, position, orientation):
        pose = utils.euler2matrix(angles=orientation, translation=position)
        for obj in self.object_nodes:
            self.scene.set_pose(obj, pose=pose)

    def remove_objects(self):
        for obj in self.object_nodes:
            self.scene.remove_node(obj)
        self.object_nodes = []

    def update_light(self, lightIDList):
        """
		Update the light node based on lightIDList, remove the previous light
		"""
        # Remove previous light nodes
        for node in self.current_light_nodes:
            self.scene.remove_node(node)

        # Add light nodes
        self.current_light_nodes = []
        for i in lightIDList:
            light_node = self.light_nodes[i]
            self.scene.add_node(light_node)
            self.current_light_nodes.append(light_node)

    def _add_noise(self, color):
        """
		Add Gaussian noise to the RGB image
		:param color:
		:return:
		"""
        # Add noise to the RGB image
        mean = 0
        std = 7

        noise = np.random.normal(mean, std, color.shape)  # Gaussian noise
        color = np.clip(color + noise, 0, 255).astype(np.uint8)  # Add noise and clip

        return color

    def _calibrate(self, color):
        if self._background_real is not None:
            # Simulated difference image, with scaling factor 0.5
            diff = (color.astype(np.float) - self._background_sim) * 0.5

            # Add low-pass filter to match real readings
            diff = cv2.GaussianBlur(diff, (7, 7), 0)

            # Combine the simulated difference image with real background image
            color = np.clip((diff[:, :, :3] + self._background_real), 0, 255).astype(
                np.uint8
            )

        return color

    def _post_process(self, color, depth, noise=True, calibration=True):
        if calibration:
            color = self._calibrate(color)
        if noise:
            color = self._add_noise(color)
        return color, depth

    def render(self, noise=True, calibration=True):

        self.scene.main_camera_node = self.camera
        self.update_light(self.cam_light_ids)

        color, depth = self.r.render(self.scene)
        color, depth = self._post_process(color, depth, noise, calibration)

        return color, depth
