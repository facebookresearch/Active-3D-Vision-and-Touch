#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import pyrender
from PIL import Image
import torch
from tqdm.contrib import tzip

from pterotactyl.utility import utils


class CameraRenderer:
    def __init__(self, cameraResolution=[512, 512]):
        self.W = cameraResolution[0]
        self.H = cameraResolution[1]
        self._init_pyrender()

    def _init_pyrender(self):
        self.scene = self._init_scene()
        self.objectNodes = []
        self.handNodes = []
        self._init_camera()
        self.r = pyrender.OffscreenRenderer(self.W, self.H)

    def _init_scene(self):
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[0, -0.8, 0.3], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.8)
        scene.add(light, pose=light_pose)

        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[0, 0.8, 0.3], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.8)
        scene.add(light, pose=light_pose)

        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[-1, 0, 1], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.8)
        scene.add(light, pose=light_pose)

        light_pose = utils.euler2matrix(
            angles=[0, 0, 0], translation=[1, 0, 1], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=1.8)
        scene.add(light, pose=light_pose)
        return scene

    def _init_camera(self):
        camera = pyrender.PerspectiveCamera(
            yfov=60.0 / 180.0 * np.pi, znear=0.01, zfar=10.0, aspectRatio=1.0
        )

        cameraPose0 = utils.euler2matrix(
            xyz="xyz", angles=[0, 0, 0], translation=[0, 0, 0], degrees=True
        )

        # Add camera node into scene
        cameraNode = pyrender.Node(camera=camera, matrix=cameraPose0)
        self.scene.add_node(cameraNode)
        self.scene.main_camera_node = cameraNode
        self.camera = cameraNode
        initial_matrix = R.from_euler("xyz", [45.0, 0, 180.0], degrees=True).as_matrix()
        self.update_camera_pose([0, 0.6, 0.6], initial_matrix)

    def update_camera_pose(self, position, orientation):
        """
		Update digit pose (including camera, lighting, and gel surface)
		"""
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = orientation

        self.camera.matrix = pose

    def add_object(self, objTrimesh, position=[0, 0, 0], orientation=[0, 0, 0]):
        mesh = pyrender.Mesh.from_trimesh(objTrimesh)
        pose = utils.euler2matrix(angles=orientation, translation=position)
        objNode = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(objNode)
        self.objectNodes.append(objNode)

    def add_points(self, points, radius, colour=[0, 0, 0]):
        sm = trimesh.creation.uv_sphere(radius=radius)
        sm.visual.vertex_colors = colour
        tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
        tfs[:, :3, 3] = points
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        objNode = pyrender.Node(mesh=m)
        self.scene.add_node(objNode)
        self.objectNodes.append(objNode)

    def remove_objects(self):
        for obj in self.objectNodes:
            self.scene.remove_node(obj)
        self.objectNodes = []

    def render(self):
        colour, depth = self.r.render(self.scene)
        colour = np.clip((np.array(colour)), 0, 255).astype(np.uint8)
        colour = Image.fromarray(colour)
        return colour



# renders the predicted mesh along with the ground truth mesh
def render_representations(locations, names, meshes, faces):
    recon_face = utils.add_faces(faces)
    scene = CameraRenderer()
    message = "rendering the predicted objects"
    print("*" * len(message))
    print(message)
    print("*" * len(message))
    for verts, name, location in tzip(meshes, names, locations):
        ###### render mesh #######
        mesh = trimesh.Trimesh(verts, recon_face)
        mesh.visual.vertex_colors = [228, 217, 111, 255]
        scene.add_object(mesh)
        img = scene.render()
        img.save(f"{location}/mesh.png")
        scene.remove_objects()

        ##### render point clouds #######
        verts = torch.FloatTensor(verts).cuda()
        faces = torch.LongTensor(recon_face).cuda()
        points = (
            utils.batch_sample(verts.unsqueeze(0), faces, num=100000)[0]
            .data.cpu()
            .numpy()
        )
        scene.add_points(points, 0.01, [228, 217, 111])
        img = scene.render()
        img.save(f"{location}/points.png")
        scene.remove_objects()

        ######## render real object #########
        verts = np.load(name + "_verts.npy")
        faces = np.load(name + "_faces.npy")
        faces = utils.add_faces(faces)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.vertex_colors = [228, 217, 111, 255]
        scene.add_object(mesh)
        img = scene.render()
        img.save(f"{location}/ground_truth.png")
        scene.remove_objects()
