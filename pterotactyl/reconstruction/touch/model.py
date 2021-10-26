# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


# CNN block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.last = last
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
        )
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        if not self.last:
            x = self.activation(x)
        return x


# Model for predicting touch chart shape
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # CNN
        CNN_layers = []
        CNN_layers.append(DoubleConv(3, 16))
        CNN_layers.append(DoubleConv(16, 32))
        CNN_layers.append(DoubleConv(32, 32))
        CNN_layers.append(DoubleConv(32, 64))
        CNN_layers.append(DoubleConv(64, 128))
        CNN_layers.append(DoubleConv(128, 128, last=True))
        self.CNN_layers = nn.Sequential(*CNN_layers)

        # MLP
        layers = []
        layers.append(nn.Sequential(nn.Linear(512, 256), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(256, 128), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(128, 75)))
        self.fc = nn.Sequential(*layers)

    def predict_verts(self, touch):
        for layer in self.CNN_layers:
            touch = layer(touch)
        points = touch.contiguous().view(-1, 512)
        points = self.fc(points)
        return points

    # tranform the predicted shape into the reference frame of the sensro
    def transform_verts(self, verts, ref):
        pos = ref["pos"].cuda().view(-1, 1, 3).repeat(1, verts.shape[1], 1)
        rot = ref["rot"].cuda()
        verts = torch.bmm(rot, verts.permute(0, 2, 1)).permute(0, 2, 1)
        verts += pos
        return verts

    def forward(self, gel, ref_frame, verts):
        verts = verts + self.predict_verts(gel).view(-1, verts.shape[1], 3)
        verts = self.transform_verts(verts, ref_frame)
        return verts
