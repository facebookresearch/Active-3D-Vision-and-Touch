#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import os
import numpy as np
from tqdm import tqdm
from glob import glob
import random
from pathlib import Path

import torch

import pterotactyl.object_data as object_data
import pterotactyl.objects as objects
from pterotactyl.utility import utils
from pterotactyl.simulator.scene import sampler
from pterotactyl.simulator.physics import grasping


def make_data_split():
    data_location = os.path.join(
        os.path.dirname(object_data.__file__), "initial_objects/*"
    )
    split_destination = os.path.join(
        os.path.dirname(objects.__file__), "data_split.npy"
    )
    object_files = glob(data_location)
    object_files = [o.split("/")[-1].split(".")[0] for o in object_files]

    object_files.sort()
    random.Random(0).shuffle(object_files)
    recon_train = object_files[:7700]
    auto_train = object_files[7700 : 2 * 7700]
    RL_train = object_files[2 * 7700 : 3 * 7700]
    valid = object_files[3 * 7700 : 3 * 7700 + 2000]
    test = object_files[3 * 7700 + 2000 : 3 * 7700 + 3000]
    dict = {
        "recon_train": recon_train,
        "auto_train": auto_train,
        "RL_train": RL_train,
        "valid": valid,
        "test": test,
    }
    np.save(split_destination, dict)


# produces a pointcloud from the surface of an object
def extract_points(verts, faces, dim=128, num_points=30000):
    verts = torch.FloatTensor(verts).cuda()
    faces = torch.LongTensor(faces).cuda()

    # converts the mesh to a voxel grid
    voxel = utils.mesh_to_voxel(verts, faces, dim)
    if voxel is None:
        return None
    # extracts orthographic depth maps from the voxel grid
    ODMs = utils.extract_ODMs(voxel)
    # reprojects the depth maps to a voxel grid to remove internal structure
    voxel = utils.apply_ODMs(ODMs, dim)

    # extracts a point cloud from the voxel grid
    points = utils.voxel_to_pointcloud(voxel)
    # aligns the pointcloud to the origional mesh
    points = utils.realign_points(points, verts.clone())
    # make the point cloud of uniform size
    while points.shape[0] < num_points:
        points = torch.cat((points, points))
    choices = np.random.choice(points.shape[0], num_points, replace=False)
    points = points[choices]
    return points


# extract the object information from mesh
def save_object_info():
    data_location = os.path.join(
        os.path.dirname(object_data.__file__), "initial_objects/*"
    )
    data_destination = os.path.join(
        os.path.dirname(object_data.__file__), "object_info/"
    )
    if not os.path.exists(data_destination):
        os.makedirs(data_destination)
    object_files = glob(data_location)
    pbar = tqdm(object_files, smoothing=0.0)
    pbar.set_description(f"Saving object information for quick loading")
    for file in pbar:
        file_destination = data_destination + file.split("/")[-1].split(".")[0]
        # scale meshes and extract vertices and faces
        verts, faces = utils.get_obj_data(file, scale=3.1)
        np.save(file_destination + "_verts.npy", verts)
        np.save(file_destination + "_faces.npy", faces)
        # save the new object as a mesh and reference it in a urdf file for pybullet
        utils.make_urdf(verts, faces, file_destination + ".urdf")


# extracts a point cloud from the object and saves it
def save_point_info():
    data_location = os.path.join(
        os.path.dirname(object_data.__file__), "object_info/*.obj"
    )
    data_destination = os.path.join(os.path.dirname(object_data.__file__), "/")
    if not os.path.exists(data_destination):
        os.makedirs(data_destination)

    object_files = glob(data_location)
    pbar = tqdm(object_files, smoothing=0.0)
    pbar.set_description(f"Extracting surface point cloud")
    for file in pbar:
        destination = data_destination + file.split("/")[-1].split(".")[0] + ".npy"
        verts = np.load(file.replace(".obj", "_verts.npy"))
        faces = np.load(file.replace(".obj", "_faces.npy"))
        # extract the point cloud
        points = extract_points(verts, faces)
        if points is None:
            continue
        np.save(destination, points.data.cpu().numpy())


# simulates the graps of an object for all possible actions
def save_simulation():
    data_location = os.path.join(
        os.path.dirname(object_data.__file__), "object_info/*.obj"
    )
    grasp_destination_dir = os.path.join(
        os.path.dirname(object_data.__file__), "grasp_info/"
    )
    image_destination_dir = os.path.join(
        os.path.dirname(object_data.__file__), "images_colourful/"
    )
    if not os.path.exists(grasp_destination_dir):
        os.makedirs(grasp_destination_dir)
    if not os.path.exists(image_destination_dir):
        os.makedirs(image_destination_dir)

    object_files = glob(data_location)
    simulation_infomation = {}
    # defines the sampling function for simulation
    s = sampler.Sampler(grasping.Agnostic_Grasp, bs=1, vision=True)

    pbar = tqdm(object_files, smoothing=0.0)
    pbar.set_description(f"Extracting grasp information")
    set = [0, 0, 0, 0]
    file_num = 0
    for file in pbar:
        file_number = file.split("/")[-1].split(".")[0]
        grasp_destination = grasp_destination_dir + file_number + "/"
        image_destination = image_destination_dir + file_number + ".npy"
        batch = [file.replace(".obj", "")]
        statuses = []
        try:
            s.load_objects(batch, from_dataset=True)
        except:
            continue

        # save an image of the object
        signals = s.sample(
            [0],
            touch=False,
            touch_point_cloud=False,
            vision=True,
            vision_occluded=False,
        )
        img = signals["vision"][0]
        np.save(image_destination, img)

        for i in range(50):
            # simulate the object
            signals = s.sample(
                [i],
                touch=True,
                touch_point_cloud=True,
                vision=False,
                vision_occluded=False,
            )
            status = signals["touch_status"][0]
            good = 0
            for k in range(4):
                if status[k] == "touch":
                    good += 1
            for k in range(good):
                set[k] += 1
            statuses.append(status)

            # extracts the touch information for each of the 4 fingers
            for j in range(4):
                instance_grasp_destination = os.path.join(
                    grasp_destination, str(i), str(j)
                )
                Path(instance_grasp_destination).mkdir(parents=True, exist_ok=True)
                if status[j] == "touch":
                    touch_signal = (
                        signals["touch_signal"][0][j].data.numpy().astype(np.uint8)
                    )
                    touch_points = signals["touch_point_cloud"][0][j]

                    np.save(instance_grasp_destination + "_touch.npy", touch_signal)
                    np.save(instance_grasp_destination + "_points.npy", touch_points)
                if status[j] != "no_intersection":
                    ref_frame_pos = signals["finger_transfrom_pos"][0][j].data.numpy()
                    ref_frame_rot_M = signals["finger_transform_rot_M"][0][
                        j
                    ].data.numpy()
                    ref_frame = {"pos": ref_frame_pos, "rot": ref_frame_rot_M}
                    np.save(instance_grasp_destination + "_ref_frame.npy", ref_frame)
        s.remove_objects()
        file_num += 0.5
        simulation_infomation[file_number] = statuses


if __name__ == "__main__":
    save_object_info()
    save_point_info()
    save_simulation()
    make_data_split()
