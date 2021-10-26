#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

from pterotactyl.utility import utils


class Agnostic_Grasp:
    def __init__(self, pb, hand):
        self.pb = pb
        self.hand = hand
        self.directions = -utils.get_circle(50).points.data.numpy()
        self.convex_mesh = None
        self.verts = None

    def set_object(self, verts, faces):
        hull = ConvexHull(verts.data.numpy())
        self.convex_mesh = trimesh.Trimesh(
            vertices=verts, faces=hull.simplices, process=False
        )
        self.verts = verts.data.numpy()

    def remove_object(self):
        self.convex_mesh = None
        self.verts = None

    # converts selected action into the corresponding hand rotation
    def action_to_params(
        self, action
    ):  # converts action selection into hand parameters
        direction = self.directions[action]
        rotation = 0
        return direction, rotation

    def grasp(self, action):
        self.reset_hand()
        direction, rotation = self.action_to_params(
            action
        )  # convert action into grasping parameters
        success = self.set_hand_hull(
            direction, rotation
        )  # identify point on convex hull which intersection the chosen hand direction

        # if no intersection is found
        if not success:
            return False
        else:
            # set all joint angles to maximum to perfrom grasp
            joint_angles = [10 for _ in range(28)]
            self.pb.setJointMotorControlArray(
                self.hand,
                range(28),
                self.pb.POSITION_CONTROL,
                targetPositions=joint_angles,
            )
            for i in range(5):
                self.pb.stepSimulation()

            return True

    def set_hand_hull(self, direction, rotation, hand_distance=0.013):
        # define ray from the center of the object to outwards in the chosen direction
        ray_origins = np.array([[0, 0, 0]])
        ray_directions = np.array([direction])

        # find intersection with ray and convex hull
        locations, index_ray, index_tri = self.convex_mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        # if no intersection were found
        if len(locations) == 0:
            return False

        else:
            # find furtherest interesection from the ceneter of the object
            test_locations = np.array(locations)
            test_locations = (test_locations ** 2).sum(axis=-1)
            max_location = np.argmax(test_locations)
            point = locations[max_location]
            face = self.convex_mesh.faces[index_tri[0]]

            # place the hand above the convex hull at the intersection point
            hand_position, surface_normal = self.get_position_on_hull(
                self.verts, face, point, hand_distance
            )
            hand_orientation = self.pb.getQuaternionFromEuler([rotation, 0, 0])
            surface_normal -= 0.001
            handUpdateOrientation = utils.quats_from_vectors([-1, 0, 0], surface_normal)
            hand_orientation = utils.combine_quats(
                handUpdateOrientation, hand_orientation
            )

            # place the middle finger tip on the point instead of the hand center
            # displacement of the fingertip from the center of the hand
            v = [0, 0, 0.133]
            matrix = (R.from_quat(hand_orientation)).as_matrix()
            hand_position -= matrix.dot(v)

            # transfrom the hand
            self.pb.resetBasePositionAndOrientation(
                self.hand, hand_position, hand_orientation
            )

            return True

    # find the normal face which the ray intersections with, and a point just above the siurface in this direction
    def get_position_on_hull(self, verts, face, point, distance):
        p1, p2, p3 = verts[face[0]], verts[face[1]], verts[face[2]]
        normal = utils.normal_from_triangle(p1, p2, p3)

        p1 = np.array([0, 0, 0])
        p2 = point
        p3 = point + normal * 0.0001

        # check the normal is pointing away from the mesh
        if ((p1 - p2) ** 2).sum() > ((p1 - p3) ** 2).sum():
            normal = normal * -1
        # move position of the finger to slightly above the mesh
        point = point + normal * distance

        return point, normal

    def reset_hand(self):
        # moves hand away from the object to avoid intersections
        self.pb.resetBasePositionAndOrientation(self.hand, [20, 0, 0], [1, 0, 0, 0])
        # sets all joints to the initial angles
        joint_angles = [0 for _ in range(28)]
        # sets thumb as oppositng fingers
        joint_angles[20] = 1.2
        joint_angles[22] = 0.7
        for i in range(28):
            self.pb.resetJointState(self.hand, i, joint_angles[i])
