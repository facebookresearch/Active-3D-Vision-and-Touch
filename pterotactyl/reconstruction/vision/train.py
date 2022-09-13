# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse


from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from submitit.helpers import Checkpointable

from pterotactyl.reconstruction.vision import model
from pterotactyl.utility import utils
from pterotactyl.utility import data_loaders
import pterotactyl.objects as objects
from pterotactyl import pretrained


class Engine(Checkpointable):
    def __init__(self, args):

        # set seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        # set initial data values
        self.epoch = 0
        self.best_loss = 10000
        self.args = args
        self.last_improvement = 0
        self.vision_chart_location = os.path.join(
            os.path.dirname(objects.__file__), "vision_charts.obj"
        )
        self.results_dir = os.path.join("results", args.exp_type, args.exp_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.checkpoint_dir = os.path.join(
            "experiments/checkpoint/", args.exp_type, args.exp_id
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not self.args.eval:
            utils.save_config(self.checkpoint_dir, args)

    def __call__(self) -> float:
        # compute mesh statistics
        self.mesh_info, self.initial_mesh = utils.load_mesh_vision(
            self.args, self.vision_chart_location
        )
        self.initial_mesh = self.initial_mesh.cuda()
        self.n_vision_charts = self.initial_mesh.shape[0]

        # define the model and optimizer
        self.encoder = model.Deformation(self.mesh_info, self.initial_mesh, self.args)
        self.encoder.cuda()
        if not self.args.eval:
            params = list(self.encoder.parameters())
            self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)

        # logging information
        writer = SummaryWriter(
            os.path.join("experiments/tensorboard/", self.args.exp_type)
        )

        # get data
        train_loader, valid_loaders = self.get_loaders()

        # evaluate of the test set
        if self.args.eval:
            self.load()
            with torch.no_grad():
                self.validate(valid_loaders, writer)
                return

        # train and validate
        else:
            self.load()
            for epoch in range(self.epoch, self.args.epochs):
                self.epoch = epoch
                self.train(train_loader, writer)
                with torch.no_grad():
                    self.validate(valid_loaders, writer)
                self.check_values()

    # get dataloaders
    def get_loaders(self):
        train_loader, valid_loader = "", ""
        if not self.args.eval:
            # training dataloader
            train_data = data_loaders.mesh_loader_vision(
                self.args, set_type="recon_train"
            )
            train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=16,
                collate_fn=train_data.collate,
            )

        # evaluation dataloder
        set_type = "test" if self.args.eval else "valid"
        valid_data = data_loaders.mesh_loader_vision(self.args, set_type=set_type)
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=16,
            collate_fn=valid_data.collate,
        )

        return train_loader, valid_loader

    def train(self, data, writer):
        total_loss = 0
        iterations = 0
        self.encoder.train()

        for k, batch in enumerate(tqdm(data, smoothing=0)):
            self.optimizer.zero_grad()

            # initialize data
            img = batch["img"].cuda()
            gt_points = batch["gt_points"].cuda()

            # for debugging , if you want to change the camera view, reach out to edward.smith@mail.mcgill.ca
            # self.encoder.img_encoder_global.debug_pooling(img, gt_points)
            # self.encoder.img_encoder_global.debug_pooling(img, self.initial_mesh.unsqueeze(0).repeat(img.shape[0], 1, 1))

            # inference
            with torch.no_grad():
                charts = model.prepare_mesh(batch, self.initial_mesh, self.args)
            verts = self.encoder(img, charts)[0]

            loss = utils.chamfer_distance(
                verts, self.mesh_info["faces"], gt_points, num=self.args.number_points
            )
            loss = self.args.loss_coeff * loss.mean()

            # backprop
            loss.backward()
            self.optimizer.step()

            # log
            message = f"Train || Epoch: {self.epoch}, loss: {loss.item():.2f}, b_ptp:  {self.best_loss:.2f}"
            tqdm.write(message)
            total_loss += loss.item()
            iterations += 1.0
        writer.add_scalars(
            "train_loss", {self.args.exp_id: total_loss / iterations}, self.epoch
        )

    def validate(self, valid_loader, writer):
        total_loss = 0
        self.encoder.eval()
        num_examples = 0
        observations = []
        names = []

        for v, batch in enumerate(tqdm(valid_loader)):
            # initialize data
            names += batch["names"]
            img = batch["img"].cuda()
            gt_points = batch["gt_points"].cuda()
            batch_size = img.shape[0]

            # inference
            charts = model.prepare_mesh(batch, self.initial_mesh, self.args)
            ps = list(self.encoder.parameters())
            ps = torch.cat([p.flatten() for p in ps])

            verts = self.encoder(img, charts)[0]
            observations.append(verts)
            loss = utils.chamfer_distance(
                verts, self.mesh_info["faces"], gt_points, num=self.args.number_points
            )

            loss = self.args.loss_coeff * loss.sum()

            # logs
            num_examples += float(batch_size)
            total_loss += loss

            message = f"Valid || Epoch: {self.epoch}, ave: {total_loss / num_examples:.4f}, b_ptp:  {self.best_loss:.2f}"
            tqdm.write(message)

            if self.args.visualize and v == 5 and self.args.eval:
                meshes = torch.cat(observations, dim=0)[:, :, :3]
                names = [n[0] for n in names]
                utils.visualize_prediction(
                    self.results_dir, meshes, self.mesh_info["faces"], names
                )

        total_loss = total_loss / num_examples

        print("*******************************************************")
        print(f"Validation Accuracy: {total_loss}")
        print("*******************************************************")

        if not self.args.eval:
            writer.add_scalars("valid_ptp", {self.args.exp_id: total_loss}, self.epoch)
        self.current_loss = total_loss

    # save the model
    def save(self):
        torch.save(self.encoder.state_dict(), self.checkpoint_dir + "/model")
        torch.save(self.optimizer.state_dict(), self.checkpoint_dir + "/optim")
        np.save(self.checkpoint_dir + "/epoch.npy", np.array([self.epoch + 1]))

    # load the model
    def load(self):
        if self.args.eval and self.args.pretrained:
            if self.args.use_img:
                if self.args.finger:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/v_t_p/"
                    )
                else:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/v_t_g/"
                    )
            else:
                if self.args.finger:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/t_p/"
                    )
                else:
                    location = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/t_g/"
                    )

            vision_args, _ = utils.load_model_config(location)
            weights = location + 'model'

            self.mesh_info, self.initial_mesh = utils.load_mesh_vision(
                vision_args, self.vision_chart_location
            )

            self.initial_mesh = self.initial_mesh.cuda()
            self.n_vision_charts = self.initial_mesh.shape[0]

            # define the model and optimizer
            self.encoder = model.Deformation(
                self.mesh_info, self.initial_mesh, vision_args
            )
            self.encoder.cuda()
            self.encoder.load_state_dict(torch.load(weights))

        else:
            try:
                self.encoder.load_state_dict(torch.load(self.checkpoint_dir + "/model"))
                self.optimizer.load_state_dict(
                    torch.load(self.checkpoint_dir + "/optim")
                )
                self.epoch = np.load(self.checkpoint_dir + "/epoch.npy")[0]
            except:
                return

    # check if the latest validation beats the previous, and save model if so
    def check_values(self):
        if self.best_loss >= self.current_loss:
            improvement = self.best_loss - self.current_loss
            print(
                f"Saving with {improvement:.3f} improvement in Chamfer Distance on Validation Set "
            )
            self.best_loss = self.current_loss
            self.last_improvement = 0
            self.save()
        else:
            self.last_improvement += 1
            if self.last_improvement >= self.args.patience:
                print(f"Over {self.args.patience} steps since last imporvement")
                print("Exiting now")
                exit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cut",
        type=float,
        default=0.33,
        help="The shared size of features in the GCN.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to use."
    )
    parser.add_argument(
        "--limit_data",
        action="store_true",
        default=False,
        help="use less data, for debugging.",
    )
    parser.add_argument(
        "--finger", action="store_true", default=False, help="use only one finger."
    )
    parser.add_argument(
        "--number_points",
        type=int,
        default=30000,
        help="number of points sampled for the chamfer distance.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Setting for the random seed."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="Initial learning rate."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Evaluate the trained model on the test set.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batch.")
    parser.add_argument(
        "--val_grasps",
        type=int,
        default=-1,
        help="number of grasps to use during validation.",
    )
    parser.add_argument(
        "--exp_id", type=str, default="test", help="The experiment name."
    )
    parser.add_argument(
        "--exp_type", type=str, default="test", help="The experiment group."
    )
    parser.add_argument(
        "--use_img", action="store_true", default=False, help="To use the image."
    )
    parser.add_argument(
        "--use_touch",
        action="store_true",
        default=False,
        help="To use the touch information.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=70,
        help="How many epochs without imporvement before training stops.",
    )
    parser.add_argument(
        "--loss_coeff", type=float, default=9000.0, help="Coefficient for loss term."
    )
    parser.add_argument(
        "--num_CNN_blocks",
        type=int,
        default=6,
        help="Number of image blocks in the CNN.",
    )
    parser.add_argument(
        "--layers_per_block",
        type=int,
        default=3,
        help="Number of image layers in each block in the CNN.",
    )
    parser.add_argument(
        "--CNN_ker_size",
        type=int,
        default=5,
        help="Size of the image kernel in each CNN layer.",
    )
    parser.add_argument(
        "--num_GCN_layers",
        type=int,
        default=20,
        help="Number of GCN layers in the mesh deformation network.",
    )
    parser.add_argument(
        "--hidden_GCN_size",
        type=int,
        default=300,
        help="Size of the feature vector for each GCN layer in the mesh deformation network.",
    )
    parser.add_argument(
        "--num_grasps", type=int, default=5, help="Number of grasps to train with. "
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="visualize predictions and actions while evaluating",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="load the pretrained model",
    )

    args = parser.parse_args()
    trainer = Engine(args)
    trainer()
