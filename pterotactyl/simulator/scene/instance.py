#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import os

import pybullet as pb
import numpy as np
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

from pterotactyl.simulator.rendering import touch_renderer
from pterotactyl.simulator.rendering import tacto_renderer
from pterotactyl.simulator.rendering import vision_renderer
from pterotactyl.utility import utils
import pterotactyl.objects as objects


class Scene:
    def __init__(
        self,
        grasp_class,
        max_depth=0.025,
        conn=pb,
        vision=True,
        resolution=[256, 256],
        object_colour=[228, 217, 111, 255],
        TACTO=False,
    ):
        hand_location = os.path.join(
            os.path.dirname(objects.__file__), "hand/allegro_hand.urdf"
        )
        self.hand = conn.loadURDF(
            hand_location,
            [0, 0, 0],
            conn.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=1,
        )
        # the indices of the hand definition which correspond to the finger's perspective
        self.touch_cameras = [6, 13, 20, 27]
        # furthest distance from the fingers which is obseravble by the touch sensors

        self.max_depth = max_depth
        if TACTO:
            self.max_depth = min(self.max_depth, 0.015)
        self.pb = conn
        self.obj = None
        self.grasper = grasp_class(self.pb, self.hand)

        self.depths = None
        self.TACTO = TACTO
        # if vision signals are desired
        self.vision = vision
        if self.vision:
            self.object_colour = object_colour
            self.camera_renderer = vision_renderer.Renderer(
                self.hand, pb, cameraResolution=resolution
            )
        if self.TACTO:
            self.touch_renderer = tacto_renderer.Renderer(cameraResolution=[121, 121])
        else:
            self.touch_renderer = touch_renderer.Renderer(cameraResolution=[121, 121])

    def grasp(self, action):
        return self.grasper.grasp(action)

    def get_hand_pose(self):
        poses = []
        for i in range(28):
            poses.append(self.get_pose(self.hand, i))
        return poses

    def get_pose(self, objID, linkID):
        if linkID <= 0:
            position, orientation = self.pb.getBasePositionAndOrientation(objID)
        else:
            position, orientation = self.pb.getLinkState(
                objID, linkID, computeLinkVelocity=False, computeForwardKinematics=True
            )[:2]
        orientation = self.pb.getEulerFromQuaternion(orientation)
        return position, orientation

    def load_obj(self, verts, faces, urdf_location):
        # adding repeating faces to ensure they are observed
        faces = utils.add_faces(faces)
        # loading into pybullet
        self.obj = self.pb.loadURDF(
            urdf_location, [0, 0, 0], [0, 0, 0, 1], useFixedBase=1
        )

        # loading into pyrender
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        self.touch_renderer.add_object(mesh, position=[0, 0, 0], orientation=[0, 0, 0])
        if self.vision:
            self.camera_renderer.add_object(
                mesh,
                position=[0, 0, 0],
                orientation=[0, 0, 0],
                colour=self.object_colour,
            )

        # loading into grasp function
        self.obj_verts = torch.FloatTensor(verts)
        self.obj_faces = torch.LongTensor(faces)
        self.grasper.set_object(self.obj_verts, self.obj_faces)

    def remove_obj(self):
        if self.obj is not None:
            self.pb.removeBody(self.obj)
        self.touch_renderer.remove_objects()
        self.obj = None
        self.hull_faces = None
        if self.vision:
            self.camera_renderer.remove_objects()
        self.grasper.remove_object()

    # render depth from the perspective of each finger
    def render_depth(self):
        statuses = []
        depths = []
        colours = []
        for i in range(4):
            # update position of the scene camera
            position, orientation = self.get_pose(self.hand, self.touch_cameras[i])
            rot_off_finger = R.from_euler("xyz", [0, -90, 0], degrees=True).as_matrix()
            rot_finger = R.from_euler("xyz", orientation, degrees=False).as_matrix()
            orientation_update = np.matmul(rot_finger, rot_off_finger)
            self.touch_renderer.update_camera_pose(
                position=position, orientation=orientation_update
            )

            # render depth
            if self.TACTO:
                colour, depth = self.touch_renderer.render()
                colours.append(colour)
            else:
                depth = self.touch_renderer.render()
            # check if object is close enough to register on touch sensor
            if (depth <= self.max_depth).sum() - (depth == 0).sum() > 0:
                statuses.append("touch")
            else:
                statuses.append("no_touch")
            depths.append(depth)
        self.depths = depths
        self.statuses = statuses
        if self.TACTO:
            self.colours = colours
        return statuses

    # converts depth map into point cloud in the reference frame of the object
    def depth_to_points(self):
        if self.TACTO:
            fov = 60.0 / 180.0 * np.pi  # intrinsic camera parameter
        else:
            fov = 40.0 / 180.0 * np.pi  # intrinsic camera parameter
        points = []
        depths = np.array(self.depths)

        out_of_range = depths > self.max_depth
        # sets depth beyond touch sensor to 1
        depths[out_of_range] = 1.0
        # sets infinite depth to 1 instead of 0
        depths[depths == 0] = 1

        for i in range(4):
            if self.statuses[i] == "touch":
                depth = depths[i]

                # creates grid of points
                ys = np.arange(0, 121)
                ys = np.tile(ys, (121, 1)) - 60
                ys = ys.transpose()
                xs = ys.transpose()

                # updates grid  with depth
                point_cloud = np.zeros((121, 121, 3))
                angle = np.arctan((np.abs(xs) / 60.0) * np.tan(fov / 2.0))
                point_cloud[:, :, 0] = depth * np.tan(angle) * np.sign(xs)
                angle = np.arctan((np.abs(ys) / 60.0) * np.tan(fov / 2.0))
                point_cloud[:, :, 1] = depth * np.tan(angle) * -np.sign(ys)
                point_cloud[:, :, 2] = -depth

                # removes depth beyond sensor range
                point_cloud = point_cloud[depth < 1.0]
                point_cloud = point_cloud.reshape((-1, 3))

                # transforms points to reference frame of the finger
                position, orientation = self.get_pose(self.hand, self.touch_cameras[i])

                rot_z = np.array([0, -90.0, 0])
                r1 = R.from_euler("xyz", rot_z, degrees=True).as_matrix()
                r2 = R.from_euler("xyz", orientation, degrees=False).as_matrix()
                orientation = np.matmul(r2, r1)
                if self.TACTO:

                    point_cloud[:, -1] = point_cloud[:, -1] - 0.0035
                point_cloud = orientation.dot(point_cloud.T).T + position
                points.append(point_cloud)
            else:
                points.append(np.array([]))
        return points

    # simulates touch signal from depth
    def depth_to_touch(self, depth):
        # set depth which werent obsevred to 1 instead of zero
        out_of_range = depth > self.max_depth
        depth[out_of_range] = 1.0
        depth[depth == 0] = 1

        dim = depth.shape[-1]
        zeros = depth >= self.max_depth
        depth = -(depth - self.max_depth)
        depth[zeros] = 0
        gel_depths = depth * 6 / self.max_depth

        # smooth depth values
        depth = gel_depths / (30.0) + 0.4
        filter_size = 7

        k = np.ones((filter_size, filter_size)) / (filter_size ** 2)
        depth_smoothed = ndimage.convolve(depth, k, mode="reflect")
        # fix "infinite" depths to zeros
        depth[zeros] = depth_smoothed[zeros]

        # add rgb and ambient lights
        light_positions = np.array(
            [[-0.5, 0.5, 1.0], [1.3, -0.4, 1.0], [1.3, 1.4, 1.0]]
        )
        # set to zero, qulitativly better
        ambient_intensity = np.array([0.0, 0.0, 0.0])
        diffuse_constant = 2.0
        touch = np.zeros((dim, dim, 3))
        touch[:, :] += ambient_intensity

        # calculate normal of surface
        zy, zx = np.gradient(depth)
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))
        normal = utils.normalize_vector(normal)

        # calc depth positions
        depth_positions = np.arange(dim).repeat(dim).reshape(dim, dim) / float(dim)
        depth_positions = np.stack(
            (depth_positions, depth_positions.transpose(), depth)
        ).transpose((1, 2, 0))

        # compute intensity from light normal using phong model, assuming no specularity
        for i in range(3):
            light_direction = light_positions[i] - depth_positions
            light_direction = utils.normalize_vector(light_direction)
            touch[:, :, i] += np.clip(
                diffuse_constant * np.multiply(normal, light_direction).sum(-1), 0, 1
            )

        touch = np.clip(touch * 255.0, 0, 255)  # clip within reasonable range
        return touch

    def render_touch(self):
        touches = []
        depths = np.array(self.depths)
        if self.TACTO:
            return self.colours
        else:
            for depth in depths:
                touches.append(self.depth_to_touch(depth))
        return touches

    def get_finger_frame(self):
        positions = []
        rots = []
        for i in range(4):
            position, orientation = self.get_pose(self.hand, self.touch_cameras[i])
            rot = R.from_euler("xyz", orientation, degrees=False).as_matrix()
            positions.append(position)
            rots.append(rot)
        frame = {"pos": torch.FloatTensor(positions), "rot_M": torch.FloatTensor(rots)}
        return frame

    def scene_render(self, occluded=True, parameters=None):
        if occluded:
            self.camera_renderer.update_hand()
        else:
            self.camera_renderer.remove_hand()
        if parameters is not None:
            self.camera_renderer.update_camera_pose(parameters[0], parameters[1])
        image = self.camera_renderer.render()
        return image
