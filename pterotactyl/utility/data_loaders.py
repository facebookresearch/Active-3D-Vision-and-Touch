# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms

import pterotactyl.objects as objects
import pterotactyl.object_data as object_data


POINT_CLOUD_LOCATION = os.path.join(
    os.path.dirname(object_data.__file__), "point_cloud_info/"
)
GRASP_LOCATION = os.path.join(os.path.dirname(object_data.__file__), "grasp_info/")
TOUCH_LOCATION = os.path.join(os.path.dirname(object_data.__file__), "touch_charts/")
IMAGE_LOCATION = os.path.join(
    os.path.dirname(object_data.__file__), "images_colourful/"
)
DATA_SPLIT = np.load(
    os.path.join(os.path.dirname(objects.__file__), "data_split.npy"), allow_pickle=True
).item()
OBJ_LOCATION = os.path.join(os.path.dirname(object_data.__file__), "object_info/")


preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def get_finger_transforms(obj, grasp, finger):
    ref_location = os.path.join(
        GRASP_LOCATION, obj, str(grasp), f"{finger}_ref_frame.npy"
    )
    touch_info = np.load(ref_location, allow_pickle=True).item()
    rot = touch_info["rot"]
    pos = touch_info["pos"]
    return torch.FloatTensor(rot), torch.FloatTensor(pos)


# class used for obtaining an instance of the dataset for training touch chart prediction
# to be passed to a pytorch dataloader
class mesh_loader_touch(object):
    def __init__(self, args, set_type="train"):
        # initialization of data locations
        self.args = args
        self.set_type = set_type
        object_names = [
            f.split("/")[-1].split(".")[0] for f in glob(f"{IMAGE_LOCATION}/*.npy")
        ]
        self.object_names = []

        if self.args.limit_data:
            random.shuffle(object_names)
            object_names = object_names[:3000]
        for n in tqdm(object_names):
            if os.path.exists(POINT_CLOUD_LOCATION + n + ".npy"):
                if os.path.exists(GRASP_LOCATION + n):
                    if n in DATA_SPLIT[self.set_type]:
                        successful_touches = glob(
                            os.path.join(GRASP_LOCATION, n, "*", "*_touch.npy")
                        )
                        if self.args.limit_data:
                            random.shuffle(successful_touches)
                            successful_touches = successful_touches[:7]
                        for touch in successful_touches:
                            grasp_number = touch.split("/")[-2]
                            finger_number = touch.split("/")[-1].split("_")[0]
                            self.object_names.append([n, grasp_number, finger_number])

        print(f"The number of {set_type} set objects found : {len(self.object_names)}")

    def __len__(self):
        return len(self.object_names)

    def standerdize_point_size(self, points):
        np.random.shuffle(points)
        points = torch.FloatTensor(points)
        while points.shape[0] < self.args.num_samples:
            points = torch.cat((points, points, points, points))
        perm = torch.randperm(points.shape[0])
        idx = perm[: self.args.num_samples]
        return points[idx]

    def __getitem__(self, index):
        object_name, grasp, finger = self.object_names[index]

        # meta data
        data = {}
        data["names"] = object_name, grasp, finger

        # hand infomation
        data["rot"], data["pos"] = get_finger_transforms(object_name, grasp, finger)

        # simulated touch information
        touch = np.load(
            os.path.join(GRASP_LOCATION, object_name, grasp, f"{finger}_touch.npy")
        )
        data["sim_touch"] = (
            torch.FloatTensor(touch).permute(2, 0, 1).contiguous().view(3, 121, 121)
            / 255.0
        )

        # point cloud information
        points = np.load(
            os.path.join(GRASP_LOCATION, object_name, grasp, f"{finger}_points.npy")
        )
        data["samples"] = self.standerdize_point_size(points)

        return data

    def collate(self, batch):
        data = {}
        data["names"] = [item["names"] for item in batch]
        data["samples"] = torch.cat([item["samples"].unsqueeze(0) for item in batch])
        data["sim_touch"] = torch.cat(
            [item["sim_touch"].unsqueeze(0) for item in batch]
        )
        data["ref"] = {}
        data["ref"]["rot"] = torch.cat([item["rot"].unsqueeze(0) for item in batch])
        data["ref"]["pos"] = torch.cat([item["pos"].unsqueeze(0) for item in batch])

        return data


# class used for obtaining an instance of the dataset for training chart deformation
# to be passed to a pytorch dataloader
class mesh_loader_vision(object):
    def __init__(self, args, set_type="train"):
        # initialization of data locations
        self.args = args
        self.set_type = set_type
        object_names = [
            f.split("/")[-1].split(".")[0] for f in glob(f"{IMAGE_LOCATION}/*.npy")
        ]

        if self.set_type == "recon_train" or self.set_type == "auto_train":
            self.get_instance = self.get_training_instance
        else:
            self.get_instance = self.get_validation_instance
        self.object_names = []
        # for debuggin use less data
        if args.limit_data:
            random.Random(0).shuffle(object_names)
            object_names = object_names[:2000]
        seed = 0
        for n in tqdm(object_names):
            if os.path.exists(POINT_CLOUD_LOCATION + n + ".npy"):
                if os.path.exists(TOUCH_LOCATION + n):
                    if n in DATA_SPLIT[self.set_type]:
                        iters = (
                            1
                            if (
                                self.set_type == "recon_train"
                                or self.set_type == "auto_train"
                            )
                            else 5
                        )
                        for _ in range(iters):
                            self.object_names.append([n, seed])
                            seed += 1

        print(f"The number of {set_type} set objects found : {len(self.object_names)}")

    def __len__(self):
        return len(self.object_names)

    def get_training_instance(self, index):
        obj, seed = random.choice(self.object_names)
        num_grasps_choice = random.choice(range(0, self.args.num_grasps + 1))
        grasp_choices = [i for i in range(50)]
        random.shuffle(grasp_choices)
        grasps = grasp_choices[:num_grasps_choice]
        return obj, grasps

    def get_validation_instance(self, index):
        obj, seed = self.object_names[index]
        grasp_choices = [i for i in range(50)]

        if self.args.val_grasps >= 0 and self.args.eval:
            num_grasps_choice = self.args.val_grasps
        else:
            num_grasps_choice = random.Random(seed).choice(
                range(0, self.args.num_grasps + 1)
            )

        random.Random(seed).shuffle(grasp_choices)
        grasps = grasp_choices[:num_grasps_choice]
        return obj, grasps

    # load object point cloud
    def get_points(self, obj):
        point_location = os.path.join(POINT_CLOUD_LOCATION, obj + ".npy")
        samples = np.load(point_location)
        np.random.shuffle(samples)
        gt_points = torch.FloatTensor(samples[: self.args.number_points])
        return gt_points

    # load image of object
    def get_image(self, obj):
        img = torch.empty((1))
        if self.args.use_img:
            img_location = os.path.join(IMAGE_LOCATION, obj + ".npy")
            img = torch.FloatTensor(np.load(img_location)).permute(2, 0, 1) / 255.0
        return torch.FloatTensor(img)

    # load touch infomation from the object
    def get_touch_info(self, obj, grasps):
        touch_charts = torch.ones((1))
        if self.args.use_touch:
            remaining = self.args.num_grasps - len(grasps)
            all_touch_charts = torch.FloatTensor(
                np.load(TOUCH_LOCATION + obj + "/touch_charts.npy")
            ).view(50, 4, 25, 4)
            if self.args.finger:
                touch_charts = all_touch_charts[grasps][:, 1]
                touch_charts = torch.cat((touch_charts, torch.zeros(remaining, 25, 4)))
            else:
                touch_charts = all_touch_charts[grasps]
                touch_charts = torch.cat(
                    (touch_charts, torch.zeros(remaining, 4, 25, 4))
                )

        return touch_charts

    def __getitem__(self, index):
        obj, grasps = self.get_instance(index)

        data = {}

        # meta data
        data["names"] = OBJ_LOCATION + obj, grasps

        # load sampled ground truth points
        data["gt_points"] = self.get_points(obj)

        # load images
        data["img"] = self.get_image(obj)

        # get touch information
        data["touch_charts"] = self.get_touch_info(obj, grasps)
        return data

    def collate(self, batch):
        data = {}
        data["names"] = [item["names"] for item in batch]
        data["gt_points"] = torch.cat(
            [item["gt_points"].unsqueeze(0) for item in batch]
        )
        data["img"] = torch.cat([item["img"].unsqueeze(0) for item in batch])
        data["touch_charts"] = torch.cat(
            [item["touch_charts"].unsqueeze(0) for item in batch]
        )
        return data


# class used for obtaining an instance of the dataset for training chart deformation
# to be passed to a pytorch dataloader
class mesh_loader_active(object):
    def __init__(self, args, set_type="RL_train"):
        # initialization of data locations
        self.args = args
        self.set_type = set_type
        object_names = [
            f.split("/")[-1].split(".")[0] for f in glob(f"{IMAGE_LOCATION}/*.npy")
        ]

        self.object_names = []
        # for debuggin use less data
        if args.limit_data:
            random.Random(0).shuffle(object_names)
            object_names = object_names[:400]

        for n in tqdm(object_names):
            if os.path.exists(POINT_CLOUD_LOCATION + n + ".npy"):
                if n in DATA_SPLIT[self.set_type]:
                    self.object_names.append(n)

        print(f"The number of {set_type} set objects found : {len(self.object_names)}")

    def __len__(self):
        return (
            len(self.object_names) // self.args.env_batch_size
        ) * self.args.env_batch_size

    def get_instance(self, index):
        obj = self.object_names[index]
        num_grasps_choice = random.choice(range(0, self.args.num_grasps + 1))
        grasp_choices = [i for i in range(50)]
        random.shuffle(grasp_choices)
        grasps = grasp_choices[:num_grasps_choice]
        return obj, grasps

    # load object point cloud
    def get_points(self, obj):
        point_location = os.path.join(POINT_CLOUD_LOCATION, obj + ".npy")
        samples = np.load(point_location)
        np.random.shuffle(samples)
        gt_points = torch.FloatTensor(samples[: self.args.number_points])
        return gt_points

    # load image of object
    def get_image(self, obj):
        img = torch.empty((1))
        if self.args.use_img:
            img_location = os.path.join(IMAGE_LOCATION, obj + ".npy")
            img = torch.FloatTensor(np.load(img_location)).permute(2, 0, 1) / 255.0
        return torch.FloatTensor(img)

    def __getitem__(self, index):
        obj = self.object_names[index]
        data = {}

        # meta data
        data["names"] = OBJ_LOCATION + obj

        # load sampled ground truth points
        data["gt_points"] = self.get_points(obj)

        # load images
        data["img"] = self.get_image(obj)

        return data

    def collate(self, batch):
        data = {}
        data["names"] = [item["names"] for item in batch]
        data["gt_points"] = torch.cat(
            [item["gt_points"].unsqueeze(0) for item in batch]
        )
        data["img"] = torch.cat([item["img"].unsqueeze(0) for item in batch])
        return data
