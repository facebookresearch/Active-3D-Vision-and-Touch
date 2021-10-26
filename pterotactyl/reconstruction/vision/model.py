# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from PIL import Image

# basic CNN layer template
def CNN_layer(f_in, f_out, k, stride=1, simple=False, padding=1):
    layers = []
    if not simple:
        layers.append(nn.BatchNorm2d(int(f_in)))
        layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(int(f_in), int(f_out), kernel_size=k, padding=padding, stride=stride)
    )
    return nn.Sequential(*layers)


# network for making image features for vertex feature vectors
class Image_Encoder(nn.Module):
    def __init__(self, args):
        super(Image_Encoder, self).__init__()

        # CNN sizes
        cur_size = 3
        next_size = 16

        # layers for the CNN
        layers = []
        layers.append(
            CNN_layer(cur_size, cur_size, args.CNN_ker_size, stride=1, simple=True)
        )
        for i in range(args.num_CNN_blocks):
            layers.append(CNN_layer(cur_size, next_size, args.CNN_ker_size, stride=2))
            cur_size = next_size
            next_size = next_size * 2
            for j in range(args.layers_per_block - 1):
                layers.append(CNN_layer(cur_size, cur_size, args.CNN_ker_size))

        self.args = args
        self.layers = nn.ModuleList(layers)
        # camera parameters
        f = 221.7025
        RT = np.array(
            [
                [
                    -7.587616579485257e-08,
                    -1.0000001192092896,
                    0.0,
                    -2.2762851159541242e-08,
                ],
                [-0.7071068286895752, 7.587616579485257e-08, -0.7071068286895752, 0.0],
                [0.7071068286895752, 0.0, -0.7071067690849304, 0.4242640733718872],
            ]
        )

        K = np.array([[f, 0, 128.0], [0, f, 128.0], [0, 0, 1]])

        # projection matrix
        self.matrix = torch.FloatTensor(K.dot(RT)).cuda()

    # defines image features over vertices from vertex positions, and feature mpas from vision
    def pooling(self, blocks, verts_pos):

        # convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension
        ext_verts_pos = torch.cat(
            (
                verts_pos,
                torch.FloatTensor(
                    np.ones([verts_pos.shape[0], verts_pos.shape[1], 1])
                ).cuda(),
            ),
            dim=-1,
        )
        ext_verts_pos = torch.matmul(ext_verts_pos, self.matrix.permute(1, 0))
        ext_verts_pos[:, :, 2][ext_verts_pos[:, :, 2] == 0] = 0.1
        xs = ext_verts_pos[:, :, 1] / ext_verts_pos[:, :, 2] / 256.0
        xs[torch.isinf(xs)] = 0.5
        ys = ext_verts_pos[:, :, 0] / ext_verts_pos[:, :, 2] / 256.0
        ys[torch.isinf(ys)] = 0.5

        full_features = None
        xs = xs.unsqueeze(2).unsqueeze(3)
        ys = ys.unsqueeze(2).unsqueeze(3)
        grid = torch.cat([ys, xs], 3)
        grid = grid * 2 - 1

        # extract image features based on vertex projected positions
        for block in blocks:
            features = torch.nn.functional.grid_sample(block, grid, align_corners=True)
            if full_features is None:
                full_features = features
            else:
                full_features = torch.cat((full_features, features), dim=1)
        vert_image_features = full_features[:, :, :, 0].permute(0, 2, 1)
        return vert_image_features

    # Examines the projection of points into image space and displayes the image
    # This is only for debugging purposes
    def debug_pooling(self, img, points):
        # convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension
        ext_verts_pos = torch.cat(
            (
                points,
                torch.FloatTensor(
                    np.ones([points.shape[0], points.shape[1], 1])
                ).cuda(),
            ),
            dim=-1,
        )
        ext_verts_pos = torch.matmul(ext_verts_pos, self.matrix.permute(1, 0))
        xs = ext_verts_pos[:, :, 1] / ext_verts_pos[:, :, 2] / 256.0
        ys = ext_verts_pos[:, :, 0] / ext_verts_pos[:, :, 2] / 256.0

        for xses, yses, i in zip(xs, ys, img):
            i = (255 * i.permute(1, 2, 0)).data.cpu().numpy().astype(np.uint8)
            for x, y in zip(xses, yses):
                x = int(x * 255)
                if x > 255:
                    x = 255
                if x < 0:
                    x = 0
                y = int(y * 255)
                if y > 255:
                    y = 255
                if y < 0:
                    y = 0
                i[x, y, 0] = 255.0
                i[x, y, 1] = 0
                i[x, y, 2] = 0

            Image.fromarray(i).save("debug_img.png")
            print("Image of point projection has been saved to debug_img.png")
            print("press enter to continue")
            input()
            print("*" * 15)
            print()
        exit()

    def forward(self, img):
        x = img
        features = []
        # layers to select image features from
        layer_selections = [
            len(self.layers) - 1 - (i + 1) * self.args.layers_per_block
            for i in range(3)
        ]
        for e, layer in enumerate(self.layers):
            # if too many layers are applied the map size will be lower then then kernel size
            if x.shape[-1] < self.args.CNN_ker_size:
                break
            x = layer(x)
            # collect feature maps
            if e in layer_selections:
                features.append(x)
        features.append(x)
        return features


# Class for defroming the charts into the traget shape
class Deformation(nn.Module):
    def __init__(
        self, adj_info, inital_positions, args, return_img=False, pass_img=False
    ):
        super(Deformation, self).__init__()
        self.adj_info = adj_info
        self.initial_positions = inital_positions
        self.args = args
        self.return_img = return_img
        self.pass_img = pass_img

        # add image encoder and get image feature size
        if args.use_img:
            self.img_encoder_global = Image_Encoder(args).cuda()
            self.img_encoder_local = Image_Encoder(args).cuda()
            with torch.no_grad():
                img_features = self.img_encoder_global(
                    torch.zeros(1, 3, 256, 256).cuda()
                )
                vert_positions = torch.zeros(1, 1, 3).cuda()
                input_size = self.img_encoder_global.pooling(
                    img_features, vert_positions
                ).shape[-1]
        else:
            # if no image features fix the feature size at 50
            input_size = 50

        # add positional and mask enocoder and GCN deformation networks
        self.positional_encoder = Positional_Encoder(input_size)
        self.mask_encoder = Mask_Encoder(input_size)
        self.mesh_deform_1 = GCN(
            input_size, args, ignore_touch_matrix=args.use_img
        ).cuda()
        self.mesh_deform_2 = GCN(input_size, args).cuda()

    def forward(self, img, charts, img_features=None):
        # number of vision charts
        vc_length = charts["vision_charts"].clone().shape[1]

        # get image features
        if self.pass_img and img_features is not None:
            global_img_features, local_img_features = img_features
        elif self.args.use_img:
            global_img_features = self.img_encoder_global(img)
            local_img_features = self.img_encoder_local(img)
        else:
            global_img_features, local_img_features = [], []

        ##### first iteration #####
        # if we are using only touch then we need to use touch information immediately
        if self.args.use_touch and not self.args.use_img:
            # use touch information

            vertices = torch.cat(
                (charts["vision_charts"].clone(), charts["touch_charts"].clone()), dim=1
            )

            mask = torch.cat(
                (charts["vision_masks"].clone(), charts["touch_masks"].clone()), dim=1
            )

            positional_features = self.positional_encoder(vertices)
            mask_features = self.mask_encoder(mask)

            vertex_features = positional_features + mask_features

        # in all other setting we only use vision
        else:
            vertices = charts["vision_charts"].clone()
            mask = charts["vision_masks"].clone()
            positional_features = self.positional_encoder(vertices)
            mask_features = self.mask_encoder(mask)
            vertex_features = positional_features + mask_features
            # use vision information
            if self.args.use_img:
                img_features = self.img_encoder_global.pooling(
                    global_img_features, vertices
                )
                vertex_features += img_features
        # perfrom the first deformation
        update = self.mesh_deform_1(vertex_features, self.adj_info)
        # update positions of vision charts only
        vertices[:, :vc_length] = vertices[:, :vc_length] + update[:, :vc_length]

        ##### second loop #####
        # add touch information if not already present
        if self.args.use_touch and self.args.use_img:
            vertices = torch.cat((vertices, charts["touch_charts"].clone()), dim=1)
            mask = torch.cat(
                (charts["vision_masks"].clone(), charts["touch_masks"].clone()), dim=1
            )
            mask_features = self.mask_encoder(mask)

        positional_features = self.positional_encoder(vertices)
        vertex_features = positional_features + mask_features
        # add image information
        if self.args.use_img:
            img_features = self.img_encoder_global.pooling(local_img_features, vertices)
            vertex_features += img_features
        # perfrom the second deformation
        update = self.mesh_deform_2(vertex_features, self.adj_info)
        # update positions of vision charts only
        vertices[:, :vc_length] = vertices[:, :vc_length] + update[:, :vc_length]

        ##### third loop #####
        positional_features = self.positional_encoder(vertices)
        mask_features = self.mask_encoder(mask)
        vertex_features = positional_features + mask_features
        if self.args.use_img:
            img_features = self.img_encoder_global.pooling(local_img_features, vertices)
            vertex_features += img_features

        # perfrom the third deformation
        update = self.mesh_deform_2(vertex_features, self.adj_info)
        # update positions of vision charts only
        vertices[:, :vc_length] = vertices[:, :vc_length] + update[:, :vc_length]
        if self.return_img:
            return vertices, mask, [global_img_features, local_img_features]
        return vertices, mask


# Graph convolutional network class for predicting mesh deformation
class GCN(nn.Module):
    def __init__(self, input_features, args, ignore_touch_matrix=False):
        super(GCN, self).__init__()
        #
        self.ignore_touch_matrix = ignore_touch_matrix
        self.num_layers = args.num_GCN_layers
        # define output sizes for each GCN layer
        hidden_values = (
            [input_features]
            + [args.hidden_GCN_size for _ in range(self.num_layers - 1)]
            + [3]
        )

        # define layers
        layers = []
        for i in range(self.num_layers):
            layers.append(
                GCN_layer(
                    hidden_values[i],
                    hidden_values[i + 1],
                    args.cut,
                    do_cut=i < self.num_layers - 1,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, features, adj_info):
        if self.ignore_touch_matrix:
            adj = adj_info["origional"]
        else:
            adj = adj_info["adj"]

        # iterate through GCN layers
        for i in range(self.num_layers):
            activation = F.relu if i < self.num_layers - 1 else lambda x: x
            features = self.layers[i](features, adj, activation)
            if torch.isnan(features).any():
                print(features)
                print("here", i, self.num_layers)
                input()

        return features


# Graph convolutional network layer
class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, cut=0.33, do_cut=True):
        super(GCN_layer, self).__init__()
        self.weight = Parameter(torch.Tensor(1, in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.cut_size = cut
        self.do_cut = do_cut

    def reset_parameters(self):
        stdv = 6.0 / math.sqrt((self.weight.size(1) + self.weight.size(0)))
        stdv *= 0.3
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, features, adj, activation):
        features = torch.matmul(features, self.weight)
        # uf we want to only share a subset of features with neighbors
        if self.do_cut:
            length = round(features.shape[-1] * self.cut_size)
            output = torch.matmul(adj, features[:, :, :length])
            output = torch.cat((output, features[:, :, length:]), dim=-1)
            output[:, :, :length] += self.bias[:length]
        else:
            output = torch.matmul(adj, features)
            output = output + self.bias

        return activation(output)


# encode the positional information of vertices using Nerf Embeddings
class Positional_Encoder(nn.Module):
    def __init__(self, input_size):
        super(Positional_Encoder, self).__init__()
        layers = []
        layers.append(
            nn.Linear(63, input_size // 4)
        )  # 10 nerf layers + original positions
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_size // 4, input_size // 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_size // 2, input_size))
        self.model = nn.Sequential(*layers)

    # apply nerf embedding of the positional information
    def nerf_embedding(self, points):
        embeddings = []
        for i in range(10):
            if i == 0:
                embeddings.append(torch.sin(np.pi * points))
                embeddings.append(torch.cos(np.pi * points))
            else:
                embeddings.append(torch.sin(np.pi * 2 * i * points))
                embeddings.append(torch.cos(np.pi * 2 * i * points))
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings

    def forward(self, positions):
        shape = positions.shape
        positions = positions.contiguous().view(shape[0] * shape[1], -1)
        # combine nerf embedding with origional positions
        positions = torch.cat((self.nerf_embedding((positions)), positions), dim=-1)
        embeding = self.model(positions).view(shape[0], shape[1], -1)
        return embeding


# make embedding token of the mask information for each vertex
class Mask_Encoder(nn.Module):
    def __init__(self, input_size):
        super(Mask_Encoder, self).__init__()
        layers_mask = []
        layers_mask.append(nn.Embedding(4, input_size))
        self.model = nn.Sequential(*layers_mask)

    def forward(self, mask):
        shape = mask.shape
        mask = mask.contiguous().view(-1, 1)
        embeding_mask = self.model(mask.long()).view(shape[0], shape[1], -1)
        return embeding_mask


# takes as input the touch information, and makes it a mart of the input mesh
def prepare_mesh(batch, vision_mesh, args):
    s1 = batch["img"].shape[0]
    if args.use_touch:

        touch_info = batch["touch_charts"].cuda().view(s1, -1, 4)
        touch_charts = touch_info[:, :, :3]
        touch_masks = touch_info[:, :, 3:]
        # combine vision charts into a single mesh
        vision_charts = vision_mesh.unsqueeze(0).repeat(s1, 1, 1)
        vision_masks = 3 * torch.ones(vision_charts.shape[:-1]).cuda().unsqueeze(-1)
        charts = {
            "touch_charts": touch_charts,
            "vision_charts": vision_charts,
            "touch_masks": touch_masks,
            "vision_masks": vision_masks,
        }
    else:
        # combine vision charts into a single mesh
        vision_charts = vision_mesh.unsqueeze(0).repeat(s1, 1, 1)
        vision_masks = 3 * torch.ones(vision_charts.shape[:-1]).cuda().unsqueeze(-1)
        charts = {"vision_charts": vision_charts, "vision_masks": vision_masks}
    return charts
