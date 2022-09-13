# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import random
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader

from pterotactyl.reconstruction.autoencoder import model
from pterotactyl.utility import utils
from pterotactyl.utility import data_loaders
from pterotactyl.reconstruction.vision import model as vision_model
import pterotactyl.objects as objects
from pterotactyl import pretrained
import pterotactyl.object_data as object_data

IMAGE_LOCATION = os.path.join(os.path.dirname(object_data.__file__), "images_colourful/")


class Engine:
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
        self.checkpoint_dir = os.path.join(
            "experiments/checkpoint/", args.exp_type, args.exp_id
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.results_dir = os.path.join("results", self.args.exp_type, self.args.exp_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        utils.save_config(self.checkpoint_dir, args)

    def __call__(self) -> float:

        # define the model and optimizer
        vision_args, weights = utils.load_model_config(self.args.vision_location)
        self.mesh_info, self.initial_mesh = utils.load_mesh_vision(
            vision_args, self.vision_chart_location
        )
        self.initial_mesh = self.initial_mesh.cuda()
        self.n_vision_charts = self.initial_mesh.shape[0]
        self.deform = vision_model.Deformation(
            self.mesh_info, self.initial_mesh, vision_args
        ).cuda()
        self.deform.load_state_dict(torch.load(weights))
        self.auto_encoder = model.AutoEncoder(
            self.mesh_info, self.initial_mesh, self.args
        )
        params = list(self.auto_encoder.parameters())
        self.auto_encoder.cuda()
        self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)
        self.load()

        # logging information
        writer = SummaryWriter(
            os.path.join("experiments/tensorboard/", self.args.exp_type)
        )
        self.train_loss = 0

        # get data
        train_loader, valid_loaders = self.get_loaders()

        # evaluate on the test set
        if self.args.eval:
            self.load()
            with torch.no_grad():
                self.validate(valid_loaders, writer)
            return

        # train and validate
        for epoch in range(0, self.args.epochs):
            self.epoch = epoch
            self.train(train_loader, writer)
            with torch.no_grad():
                self.validate(valid_loaders, writer)
            self.check_values()

    # get dataloaders
    def get_loaders(self):
        train_loader, valid_loader = "", ""
        # training loader
        if not self.args.eval:
            train_data = data_loaders.mesh_loader_vision(
                self.args, set_type="auto_train"
            )
            train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=16,
                collate_fn=train_data.collate,
            )

        # evaluation loaders
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
        self.auto_encoder.train()

        for k, batch in enumerate(tqdm(data, smoothing=0)):
            self.optimizer.zero_grad()

            # initialize data
            img = batch["img"].cuda()

            # inference
            with torch.no_grad():
                charts = vision_model.prepare_mesh(batch, self.initial_mesh, self.args)
                verts, mask = self.deform(img, charts)
            pred_points, latent = self.auto_encoder(verts.detach(), mask)

            loss = utils.chamfer_distance(
                verts.detach(),
                self.mesh_info["faces"],
                pred_points,
                num=self.args.number_points,
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
        self.train_loss = total_loss / iterations
        writer.add_scalars(
            "train_loss", {self.args.exp_id: total_loss / iterations}, self.epoch
        )

    def validate(self, valid_loader, writer):
        total_loss = 0
        self.auto_encoder.eval()
        num_examples = 0

        latents = []
        names = []

        for v, batch in enumerate(tqdm(valid_loader)):
            self.optimizer.zero_grad()
            # initialize data
            img = batch["img"].cuda()
            batch_size = img.shape[0]

            # inference
            charts = vision_model.prepare_mesh(batch, self.initial_mesh, self.args)
            verts, mask = self.deform(img, charts)
            pred_points, latent = self.auto_encoder(verts.detach(), mask)
            names += batch["names"]
            latents.append(latent)
            loss = utils.chamfer_distance(
                verts.detach(),
                self.mesh_info["faces"],
                pred_points,
                num=self.args.number_points,
            )
            loss = self.args.loss_coeff * loss.mean() * batch_size

            # logs
            num_examples += float(batch_size)
            total_loss += loss

        total_loss = total_loss / num_examples
        message = f"Valid || Epoch: {self.epoch}, train loss: {self.train_loss:.4f}, val loss: {total_loss:.4f}, b_ptp:  {self.best_loss:.4f}"
        tqdm.write(message)

        print("*******************************************************")
        print(f"Validation Accuracy: {total_loss}")
        print("*******************************************************")

        if not self.args.eval:
            writer.add_scalars("valid_ptp", {self.args.exp_id: total_loss}, self.epoch)
        self.current_loss = total_loss
        if self.args.eval:
            latents = torch.cat(latents)
            self.cluster(latents, names)

    # save the model
    def save(self):
        torch.save(self.auto_encoder.state_dict(), self.checkpoint_dir + "/model")
        torch.save(self.optimizer.state_dict(), self.checkpoint_dir + "/optim")

    # load the model
    def load(self):
        if self.args.eval and self.args.pretrained:
            if self.args.use_img:
                if self.args.finger:
                    location_vision = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/v_t_p/"
                    )
                    location_auto = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/v_t_p/"
                    )
                else:
                    location_vision = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/v_t_g/"
                    )
                    location_auto = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/v_t_g/"
                    )
            else:
                if self.args.finger:
                    location_vision = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/t_p/"
                    )
                    location_auto = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/t_p/"
                    )
                else:
                    location_vision = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/vision/t_g/"
                    )
                    location_auto = (
                        os.path.dirname(pretrained.__file__)
                        + "/reconstruction/auto/t_g/"
                    )

            # define the vision model
            vision_args, _ = utils.load_model_config(location_vision)
            weights = location_vision + 'model'
            self.mesh_info, self.initial_mesh = utils.load_mesh_vision(
                vision_args, self.vision_chart_location
            )

            self.initial_mesh = self.initial_mesh.cuda()
            self.n_vision_charts = self.initial_mesh.shape[0]

            self.deform = vision_model.Deformation(
                self.mesh_info, self.initial_mesh, vision_args
            )
            self.deform.cuda()
            self.deform.load_state_dict(torch.load(weights))
            self.deform.eval()

            # define the autoencoder model

            auto_args, _ = utils.load_model_config(location_auto)
            weights = location_auto + '/model'
            self.auto_encoder = model.AutoEncoder(
                self.mesh_info, self.initial_mesh, auto_args
            )
            self.auto_encoder.cuda()
            self.auto_encoder.load_state_dict(torch.load(weights))

        else:
            try:
                self.auto_encoder.load_state_dict(
                    torch.load(self.checkpoint_dir + "/model")
                )
                self.optimizer.load_state_dict(
                    torch.load(self.checkpoint_dir + "/optim")
                )
            except:
                return

    # check if current validation is better, and if so save model
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

    def cluster(self, latents, names):
        example_nums = 20
        crop = 20
        img_dim = 256
        examples = random.choices(range(latents.shape[0]), k=example_nums)
        collage = Image.new(
            "RGB", ((img_dim - crop * 2) * 5, (img_dim - crop * 2) * example_nums)
        )

        for v, e in enumerate(examples):
            new_im = Image.new("RGB", (img_dim * 5, img_dim))
            l = latents[e]
            main_obj = names[e][0].split("/")[-1]

            imgs = [os.path.join(IMAGE_LOCATION, main_obj + ".npy")]
            seen = [main_obj]

            compare_latents = latents - l.unsqueeze(0)
            compare_latents = (compare_latents ** 2).sum(-1)
            closest = torch.topk(compare_latents, 25, largest=False)[1][1:]
            for c in closest:
                obj = names[c][0].split("/")[-1]
                if obj in seen:
                    continue
                seen.append(obj)
                imgs.append(os.path.join(IMAGE_LOCATION, obj + ".npy"))

            for i in range(5):
                im = Image.fromarray(np.load(imgs[i]))
                new_im.paste(im, (i * img_dim, 0))
            new_im.save(f"{self.results_dir}/valid_{v}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cut",
        type=float,
        default=0.33,
        help="The shared size of features in the GCN.",
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
        "--vision_location",
        type=str,
        default=os.path.dirname(pretrained.__file__) + "/reconstruction/vision/t_p/",
        help="the location of the deformation  prediction.",
    )
    parser.add_argument(
        "--number_points",
        type=int,
        default=30000,
        help="number of points sampled for the chamfer distance.",
    )
    parser.add_argument(
        "--encoding_size", type=int, default=200, help="size of the latent vector"
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
        "--epochs", type=int, default=1000, help="Number of epochs to use."
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
