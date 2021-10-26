# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader

from pterotactyl.reconstruction.touch import model
from pterotactyl.utility import utils
from pterotactyl.utility import data_loaders
import pterotactyl.objects as objects
from pterotactyl import pretrained


class Engine:
    def __init__(self, args):
        utils.set_seeds(args.seed)
        self.epoch = 0
        self.best_loss = 10000
        self.args = args
        self.last_improvement = 0
        self.checkpoint_dir = os.path.join(
            "experiments/checkpoint/", args.exp_type, args.exp_id
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        utils.save_config(self.checkpoint_dir, args)

        chart_location = os.path.join(
            os.path.dirname(objects.__file__), "touch_chart.obj"
        )
        self.verts, self.faces = utils.load_mesh_touch(chart_location)
        self.verts = self.verts.view(1, self.verts.shape[0], 3).repeat(
            args.batch_size, 1, 1
        )

    def __call__(self):
        self.encoder = model.Encoder()
        self.encoder.cuda()
        params = list(self.encoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.args.lr)
        writer = SummaryWriter(
            os.path.join("experiments/tensorboard/", self.args.exp_type)
        )
        train_loader, valid_loader = self.get_loaders()

        # evaluate
        if self.args.eval:
            self.load()
            with torch.no_grad():
                self.validate(valid_loader, writer)
                return
        # train and validate
        else:
            for epoch in range(self.args.epochs):
                self.epoch = epoch
                self.train(train_loader, writer)
                with torch.no_grad():
                    self.validate(valid_loader, writer)
                self.check_values()

    # get the dataloaders
    def get_loaders(self):
        train_loader, valid_loader = "", ""
        # dataloader for training
        if not self.args.eval:
            train_data = data_loaders.mesh_loader_touch(
                self.args, set_type="recon_train"
            )
            train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=16,
                collate_fn=train_data.collate,
            )
        # dataloader for evaluation
        set_type = "test" if self.args.eval else "valid"
        valid_data = data_loaders.mesh_loader_touch(self.args, set_type=set_type)
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
        for k, batch in enumerate(tqdm(data)):

            # initialize
            self.optimizer.zero_grad()
            sim_touch = batch["sim_touch"].cuda()
            ref_frame = batch["ref"]
            gt_points = batch["samples"].cuda()
            batch_size = gt_points.shape[0]

            # inference
            pred_verts = self.encoder(
                sim_touch, ref_frame, self.verts.clone()[:batch_size]
            )

            loss = self.args.loss_coeff * utils.chamfer_distance(
                pred_verts, self.faces, gt_points, self.args.num_samples
            )

            loss = loss.mean()
            total_loss += loss.data.cpu().numpy()

            # backprop
            loss.backward()
            self.optimizer.step()

            # log
            message = f"Train || Epoch: {self.epoch},  loss: {loss.item():.5f} "
            message += f"|| best_loss:  {self.best_loss :.5f}"
            tqdm.write(message)
            iterations += 1.0

        writer.add_scalars(
            "train", {self.args.exp_id: total_loss / iterations}, self.epoch
        )

    def validate(self, valid_loader, writer):
        total_loss = 0
        self.encoder.eval()
        num_examples = 0
        for k, batch in enumerate(tqdm(valid_loader)):

            # initialize data
            sim_touch = batch["sim_touch"].cuda()
            ref_frame = batch["ref"]
            gt_points = batch["samples"].cuda()
            batch_size = gt_points.shape[0]

            # inference
            pred_verts = self.encoder(
                sim_touch, ref_frame, self.verts.clone()[:batch_size]
            )

            # back prop
            loss = self.args.loss_coeff * utils.chamfer_distance(
                pred_verts, self.faces, gt_points, self.args.num_samples
            )

            loss = loss.mean()
            num_examples += float(batch_size)
            total_loss += loss * float(batch_size)

        total_loss = total_loss / float(num_examples)
        # log
        print("*******************************************************")
        print(f"Total validation loss: {total_loss}")
        print("*******************************************************")

        if not self.args.eval:
            writer.add_scalars("valid", {self.args.exp_id: total_loss}, self.epoch)
        self.current_loss = total_loss

    # save the model
    def save(self):
        torch.save(self.encoder.state_dict(), self.checkpoint_dir + "/model")
        torch.save(self.optimizer.state_dict(), self.checkpoint_dir + "/optim")

    # check if the latest validation is better, save if so
    def check_values(self):
        if self.best_loss >= self.current_loss:
            improvement = self.best_loss - self.current_loss
            self.best_loss = self.current_loss
            print(f"Saving Model with a {improvement} improvement in point loss")
            self.save()
            self.last_improvement = 0
        else:
            self.last_improvement += 1
            if self.last_improvement == self.args.patience:
                print(f"Over {self.args.patience} steps since last imporvement")
                print("Exiting now")
                exit()
        print("*******************************************************")

    # load the model
    def load(self):
        if self.args.eval and self.args.pretrained:
            location = (
                os.path.dirname(pretrained.__file__)
                + "/reconstruction/touch/best/model"
            )
            self.encoder.load_state_dict(torch.load(location))
        else:
            self.encoder.load_state_dict(torch.load(self.checkpoint_dir + "/model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0, help="Setting for the random seed."
    )
    parser.add_argument(
        "--limit_data",
        action="store_true",
        default=False,
        help="reduces the number of data examples",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to use."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Initial learning rate."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Evaluate the trained model on the test set.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batch.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4000,
        help="Number of points in the predicted point cloud.",
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
        "--exp_id", type=str, default="test", help="The experiment name"
    )
    parser.add_argument(
        "--exp_type", type=str, default="test", help="The experiment group"
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
