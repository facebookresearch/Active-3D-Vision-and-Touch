# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
from PIL import Image
import math
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from scipy import ndimage
from collections import namedtuple

from pytorch3d.loss import chamfer_distance as cuda_cd
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords
from pytorch3d.io.obj_io import load_obj, save_obj
from pterotactyl.utility import pretty_render
import pterotactyl.objects as objects




def load_mesh_vision(args, obj):
	# load obj file
	verts, faces = load_mesh_touch(obj)

	# get adjacency matrix infomation
	adj_info = adj_init(verts, faces, args)
	return adj_info, verts

# set seeds for consistency
def set_seeds(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	random.seed(seed)


# normalizes symetric, binary adj matrix such that sum of each row is 1
def normalize_adj(mx):
	rowsum = mx.sum(1)
	r_inv = (1. / rowsum).view(-1)
	r_inv[r_inv != r_inv] = 0.
	mx = torch.mm(torch.eye(r_inv.shape[0]).to(mx.device) * r_inv, mx)
	return mx


# defines the adjacecny matrix for an object
def adj_init(verts, faces, args):
	# get generic adjacency matrix for vision charts
	adj = calc_adj(faces)
	adj_info = {}
	adj_info['origional'] = normalize_adj(adj.clone())
	# this combines the adjacency information of touch and vision charts
	# the output adj matrix has the first k rows corresponding to vision charts, and the last |V| - k
	# corresponding to touch charts. Similarly the first l faces are correspond to vision charts, and the
	# remaining correspond to touch charts
	if args.use_touch:
		adj, faces = adj_fuse_touch(verts, faces, adj, args)

	adj = normalize_adj(adj)
	adj_info['adj'] = adj
	adj_info['faces'] = faces
	return adj_info


# combines graph for vision and touch charts to define a fused adjacency matrix
def adj_fuse_touch(verts, faces, adj, args):
	verts = verts.data.cpu().numpy()
	hash = {}
	number_of_grasps = args.num_grasps
	# find vertices which have the same 3D position
	for e, v in enumerate(verts):
		if v.tobytes() in hash:
			hash[v.tobytes()].append(e)
		else:
			hash[v.tobytes()] = [e]

	# load object information for generic touch chart
	if args.use_touch:
		chart_location = os.path.join(
			os.path.dirname(objects.__file__), "touch_chart.obj"
		)
		sheet_verts, sheet_faces = load_mesh_touch(chart_location)
		sheet_adj = calc_adj(sheet_faces)

		# central vertex for each touch chart that will communicate with all vision charts
		central_point = 4
		fingers = 1 if args.finger else 4
		central_points = [central_point + (i * sheet_adj.shape[0]) + adj.shape[0] for i in
						  range(fingers * number_of_grasps)]

		# define and fill new adjacency matrix with vision and touch charts
		new_dim = adj.shape[0] + (fingers * number_of_grasps * sheet_adj.shape[0])
		new_adj = torch.zeros((new_dim, new_dim)).cuda()
		new_adj[: adj.shape[0], :adj.shape[0]] = adj.clone()
		for i in range(fingers * number_of_grasps):
			start = adj.shape[0] + (sheet_adj.shape[0] * i)
			end = adj.shape[0] + (sheet_adj.shape[0] * (i + 1))
			new_adj[start: end, start:end] = sheet_adj.clone()
		adj = new_adj

		# define new faces with vision and touch charts
		all_faces = [faces]
		for i in range(fingers * number_of_grasps):
			temp_sheet_faces = sheet_faces.clone() + verts.shape[0]
			temp_sheet_faces += i * sheet_verts.shape[0]
			all_faces.append(temp_sheet_faces)
		faces = torch.cat(all_faces)

	# update adjacency matrix to allow communication between vision and touch charts
	for key in hash.keys():
		cur_verts = hash[key]
		if len(cur_verts) > 1:
			for v1 in cur_verts:
				for v2 in cur_verts:  # vertices on the boundary of vision charts can communicate
					adj[v1, v2] = 1
				if args.use_touch:
					for c in central_points:  # touch and vision charts can communicate
						adj[v1, c] = 1
						adj[c, v1] = 1

	return adj, faces


# computes adjacemcy matrix from face information
def calc_adj(faces):
	v1 = faces[:, 0]
	v2 = faces[:, 1]
	v3 = faces[:, 2]
	num_verts = int(faces.max())
	adj = torch.eye(num_verts + 1).to(faces.device)

	adj[(v1, v2)] = 1
	adj[(v1, v3)] = 1
	adj[(v2, v1)] = 1
	adj[(v2, v3)] = 1
	adj[(v3, v1)] = 1
	adj[(v3, v2)] = 1

	return adj


# sample points from a batch of meshes
def batch_sample(verts, faces, num=10000):
	# Pytorch3D based code
	bs = verts.shape[0]
	face_dim = faces.shape[0]
	vert_dim = verts.shape[1]
	# following pytorch3D convention shift faces to correctly index flatten vertices
	F = faces.unsqueeze(0).repeat(bs, 1, 1)
	F += vert_dim * torch.arange(0, bs).unsqueeze(-1).unsqueeze(-1).to(F.device)
	# flatten vertices and faces
	F = F.reshape(-1, 3)
	V = verts.reshape(-1, 3)
	with torch.no_grad():
		areas, _ = mesh_face_areas_normals(V, F)
		Ar = areas.reshape(bs, -1)
		Ar[Ar != Ar] = 0
		Ar = torch.abs(Ar / Ar.sum(1).unsqueeze(1))
		Ar[Ar != Ar] = 1

		sample_face_idxs = Ar.multinomial(num, replacement=True)
		sample_face_idxs += face_dim * torch.arange(0, bs).unsqueeze(-1).to(Ar.device)


	# Get the vertex coordinates of the sampled faces.
	face_verts = V[F]
	v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

	# Randomly generate barycentric coords.
	w0, w1, w2 = _rand_barycentric_coords(bs, num, V.dtype, V.device)

	# Use the barycentric coords to get a point on each sampled face.
	A = v0[sample_face_idxs]  # (N, num_samples, 3)
	B = v1[sample_face_idxs]
	C = v2[sample_face_idxs]
	samples = w0[:, :, None] * A + w1[:, :, None] * B + w2[:, :, None] * C

	return samples


# implemented from:
# https://github.com/EdwardSmith1884/GEOMetrics/blob/master/utils.py
# MIT License
# loads the initial mesh and returns vertex, and face information
def load_mesh_touch(obj):
	obj_info = load_obj(obj)
	verts = obj_info[0]
	faces = obj_info[1].verts_idx
	verts = torch.FloatTensor(verts).cuda()
	faces = torch.LongTensor(faces).cuda()
	return verts, faces


# returns the chamfer distance between a mesh and a point cloud
def chamfer_distance(verts, faces, gt_points, num=1000, repeat=3):
	pred_points= batch_sample(verts, faces, num=num)

	cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
	if repeat > 1:
		cds = [cd]
		for i in range(repeat - 1):
			pred_points = batch_sample(verts, faces, num=num)
			cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
			cds.append(cd)
		cds = torch.stack(cds)
		cd = cds.mean(dim=0)

	return cd

# saves a point cloud as a .obj file
def save_points(file, points):
	location = f'{file}.obj'
	try:
		write_obj(location, points.data.cpu().numpy(), [])
	except:
		write_obj(location, points, [])

# converts a voxel object to a point cloud
def extract_surface(voxel):
	conv_filter = torch.ones((1, 1, 3, 3, 3)).cuda()
	local_occupancy = F.conv3d(voxel.unsqueeze(
		0).unsqueeze(0), conv_filter, padding=1)
	local_occupancy = local_occupancy.squeeze(0).squeeze(0)
	# only elements with exposed faces
	surface_positions = (local_occupancy < 27) * (local_occupancy > 0)
	points = torch.where(surface_positions)
	points = torch.stack(points)
	points = points.permute(1, 0)
	return points.type(torch.cuda.FloatTensor)

# saves a mesh as an .obj file
def write_obj(filename, verts, faces):
	""" write the verts and faces on file."""
	with open(filename, 'w') as f:
		# write vertices
		f.write('g\n# %d vertex\n' % len(verts))
		for vert in verts:
			f.write('v %f %f %f\n' % tuple(vert))

		# write faces
		f.write('# %d faces\n' % len(faces))
		for face in faces:
			f.write('f %d %d %d\n' % tuple(face))



# makes the sphere of actions
class get_circle(object):
	def __init__(self, num_points, rank=0):
		action_position = []
		a = 4 * np.pi / float(num_points)
		d = math.sqrt(a)
		M_t = round(np.pi / d)
		d_t = np.pi / M_t
		d_phi = a / d_t
		sphere_positions = []
		for i in range(0, M_t):
			theta = np.pi * (i + .5) / M_t
			M_phi = round(2 * np.pi * math.sin(theta) / d_phi)
			for j in range(0, M_phi):
				phi = 2 * np.pi * j / M_phi
				point = self.get_point(theta, phi)
				sphere_positions.append([theta, phi])
				action_position.append(point)
		self.points = torch.stack(action_position)
		self.sphere_points = sphere_positions
		if num_points != self.points.shape[0]:
			print(f' we have {self.points.shape} points but want {num_points}')
			exit()

	def get_point(self, a, b):
		x = math.sin(a) * math.cos(b)
		y = math.sin(a) * math.sin(b)
		z = math.cos(a)
		return torch.FloatTensor([x, y, z])



# get the normal of a 3D traingle
def normal_from_triangle(a, b, c):
	A = b - a
	B = c - a
	normal = np.cross(A, B)
	normal = normalize_vector(normal.reshape(1, 1, 3))
	return normal.reshape(3)

# normalizes a vector
def normalize_vector(vector):
	n = np.linalg.norm(vector, axis=2)
	vector[:, :, 0] /= n
	vector[:, :, 1] /= n
	vector[:, :, 2] /= n
	return vector

# combines 2 3D rotations and converts to a quaternion
def quats_from_vectors(vec1, vec2):
	vec1 = np.array(vec1)
	vec2 = np.array(vec2)

	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	if s == 0:
		s = 1
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

	rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	quat = R.from_matrix(rotation_matrix).as_quat()

	return quat

# combines two quaternions
def combine_quats(q1, q2):
	r1 = R.from_quat(q1).as_matrix()
	r2 = R.from_quat(q2).as_matrix()
	new_q = R.from_matrix(np.matmul(r1, r2)).as_quat()
	return new_q

# converts a euler rotation to a rotation matrix
def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False):
	r = R.from_euler(xyz, angles, degrees=degrees)
	pose = np.eye(4)
	pose[:3, 3] = translation
	pose[:3, :3] = r.as_matrix()
	return pose

# adds redundent faces
def add_faces(faces):
	f1 = np.array(faces[:, 0]).reshape(-1, 1)
	f2 = np.array(faces[:, 1]).reshape(-1, 1)
	f3 = np.array(faces[:, 2]).reshape(-1, 1)
	faces_2 = np.concatenate((f1, f3, f2), axis=-1)
	faces_3 = np.concatenate((f3, f2, f1), axis=-1)
	faces = np.concatenate((faces, faces_2, faces_3), axis=0)
	return faces

# centers a pointcloud and scales to defined size
def scale_points(points, scale = 1.):
	for i in range(3):
		points[:,i] -= points[:,i].min()
	points = points / points.max()
	points = points / scale
	for i in range(3):
		verts_range = points[:, i].max()
		points[:, i] -= verts_range / 2.
	return points

# makes a urdf file pointing to a mesh
def make_urdf(verts, faces, urdf_location):
	obj_location = urdf_location.replace('.urdf', '.obj')
	faces = add_faces(faces)
	save_obj(obj_location, torch.FloatTensor(verts), torch.LongTensor(faces), 4)

	blank_location = os.path.join(os.path.dirname(objects.__file__), 'blank.urdf')
	tree = ET.parse(blank_location)
	root = tree.getroot()
	root.attrib['name'] = 'object.urdf'

	root[0][2][1][0].attrib['filename'] = obj_location
	root[0][3][1][0].attrib['filename'] = obj_location
	tree.write(urdf_location)

# loads a obj file and scales it
def get_obj_data(obj_location, scale = 1.):
	obj_info = load_obj(obj_location)
	verts = obj_info[0].data.numpy()
	verts = scale_points(verts, scale)
	faces = obj_info[1].verts_idx.data.numpy()
	return verts, faces

# converts a mesh to a voxel array by subdeviding the mesh
def mesh_to_voxel(verts, faces, resolution):
	# maximum side lentghs of the subdevided triangles
	smallest_side = (1. / resolution) ** 2

	# center the mesh and scales to unit
	verts_max = verts.max()
	verts_min = verts.min()
	verts = (verts - verts_min) / (verts_max - verts_min) - 0.5

	# get all of the mesh triangles
	faces = faces.clone()
	v1 = torch.index_select(verts, 0, faces[:, 0])
	v2 = torch.index_select(verts, 0, faces[:, 1])
	v3 = torch.index_select(verts, 0, faces[:, 2])
	# defined points as swt of all vertices
	points = torch.cat((v1, v2, v3))

	while True:
		# get maximum side length of all traingles
		side_1 = (torch.abs(v1 - v2) ** 2).sum(dim=1).unsqueeze(1)
		side_2 = (torch.abs(v2 - v3) ** 2).sum(dim=1).unsqueeze(1)
		side_3 = (torch.abs(v3 - v1) ** 2).sum(dim=1).unsqueeze(1)
		sides = torch.cat((side_1, side_2, side_3), dim=1)
		sides = sides.max(dim=1)[0]

		# identify triangles which are small enough
		keep = sides > smallest_side
		if keep.sum() == 0:
			break

		# remove triangles which are small enough
		v1 = v1[keep]
		v2 = v2[keep]
		v3 = v3[keep]
		v4 = (v1 + v3) / 2.
		v5 = (v1 + v2) / 2.
		v6 = (v2 + v3) / 2.
		del (side_1, side_2, side_3, keep, sides)

		# add new vertices to set of points
		points = torch.cat((points, v4, v5, v6))

		# add subdevided traingles to list of triagnles
		vertex_set = [v1, v2, v3, v4, v5, v6]
		new_traingles = [[0, 3, 4], [4, 1, 5], [4, 3, 5], [3, 2, 5]]
		new_verts = []
		for i in range(4):
			for j in range(3):
				if i == 0:
					new_verts.append(vertex_set[new_traingles[i][j]])
				else:
					new_verts[j] = torch.cat(
						(new_verts[j], vertex_set[new_traingles[i][j]]))

		v1, v2, v3 = new_verts
		del (v4, v5, v6, vertex_set, new_verts)
	del (v1, v2, v3)
	if points is None:
		return None

	# scales points
	points = ((points + .5) * (resolution - 1)).long()
	points = torch.split(points.permute(1, 0), 1, dim=0)
	points = [m.unsqueeze(0) for m in points]
	# set grid points to on if a point exists inside them
	voxel = torch.zeros((resolution, resolution, resolution)).cuda()
	voxel[points] = 1

	return voxel

#  converts a voxel grid to a pointcloud
def voxel_to_pointcloud(voxel):
	voxel = voxel.float()
	off_positions = voxel == 0
	conv_filter = torch.ones((1, 1, 3, 3, 3))
	surface_voxel = torch.zeros(voxel.shape).cuda()
	conv_filter = conv_filter.cuda()
	local_occupancy = F.conv3d(voxel.unsqueeze(0).unsqueeze(0), conv_filter, padding=1)
	local_occupancy = local_occupancy.squeeze(0).squeeze(0)
	surface_positions = (local_occupancy < 27) * (local_occupancy > 0)
	surface_voxel[surface_positions] = 1
	surface_voxel[off_positions] = 0
	points = torch.where(surface_voxel != 0)
	points = torch.stack(points).permute(1, 0).float()
	return points

# implemented from:
# https://github.com/EdwardSmith1884/GEOMetrics/blob/master/utils.py
# MIT License
def extract_ODMs(voxels):
	voxels = voxels.data.cpu().numpy()
	dim = voxels.shape[0]
	a, b, c = np.where(voxels == 1)
	large = int(dim * 1.5)
	big_list = [[[[-1, large] for j in range(dim)] for i in range(dim)] for k in range(3)]
	# over the whole object extract for each face the first and last occurance of a voxel at each pixel
	# we take highest for convinience
	for i, j, k in zip(a, b, c):
		big_list[0][i][j][0] = (max(k, big_list[0][i][j][0]))
		big_list[0][i][j][1] = (min(k, big_list[0][i][j][1]))
		big_list[1][i][k][0] = (max(j, big_list[1][i][k][0]))
		big_list[1][i][k][1] = (min(j, big_list[1][i][k][1]))
		big_list[2][j][k][0] = (max(i, big_list[2][j][k][0]))
		big_list[2][j][k][1] = (min(i, big_list[2][j][k][1]))
	ODMs = np.zeros((6, dim, dim))  # will hold odms
	for i in range(dim):
		for j in range(dim):
			ODMs[0, i, j] = dim - 1 - big_list[0][i][j][0] if big_list[0][i][j][0] > -1 else dim
			ODMs[1, i, j] = big_list[0][i][j][1] if big_list[0][i][j][1] < large else dim
			ODMs[2, i, j] = dim - 1 - big_list[1][i][j][0] if big_list[1][i][j][0] > -1 else dim
			ODMs[3, i, j] = big_list[1][i][j][1] if big_list[1][i][j][1] < large else dim
			ODMs[4, i, j] = dim - 1 - big_list[2][i][j][0] if big_list[2][i][j][0] > -1 else dim
			ODMs[5, i, j] = big_list[2][i][j][1] if big_list[2][i][j][1] < large else dim

	return ODMs

# implemented from:
# https://github.com/EdwardSmith1884/GEOMetrics/blob/master/utils.py
# MIT License
# use orthographic depth maps to do space carving
def apply_ODMs(ODMs, dim):
	voxel = np.ones((dim, dim, dim))
	a, b, c = np.where(ODMs > 0)
	for x, i, j in zip(a, b, c):
		pos = int(ODMs[x, i, j])
		if x == 0:
			voxel[i, j, -pos:] = 0
		if x == 1:
			voxel[i, j, :pos] = 0
		if x == 2:
			voxel[i, -pos:, j] = 0
		if x == 3:
			voxel[i, :pos, j] = 0
		if x == 4:
			voxel[-pos:, i, j] = 0
		if x == 5:
			voxel[:pos, i, j] = 0
	voxel[ndimage.binary_fill_holes(voxel)] = 1
	return torch.LongTensor(voxel).cuda()

# aligns a pointcloud to the size of a mesh
def realign_points(points, verts):
	points = points.float()
	verts = verts
	for i in range(3):
		points[:, i] = points[:, i] - ((points[:, i].max() + points[:, i].min()) / 2.)
		v_range = verts[:, i].max() - verts[:, i].min()
		p_range = points[:, i].max() + 1 - points[:, i].min()
		points[:, i] = points[:, i] * v_range / p_range

	return points

# saves arguments for a experiment
def save_config(location, args):
	abs_path = os.path.abspath(location)
	args = vars(args)
	args['check_point'] = abs_path

	config_location = f'{location}/config.json'
	with open(config_location, 'w') as fp:
		json.dump(args, fp, indent=4)

	return config_location

# loads arguments from an experiment and the model weights
def load_model_config(location):
	config_location = f'{location}/config.json'
	with open(config_location) as json_file:
		data = json.load(json_file)
	weight_location = data['check_point'] + '/model'
	args = namedtuple("ObjectName", data.keys())(*data.values())
	return args, weight_location

# for nicely visualizing dpeth images
def visualize_depth(depth, max_depth=0.025):
	depth[depth > max_depth] = 0
	depth = 255 * (depth / max_depth)
	depth = depth.astype(np.uint8)
	return depth

# visualize the actions used by the policy
def visualize_actions(location, actions, args):
	actions = actions.view(-1).long().data.cpu().numpy()
	circle = get_circle(args.num_actions)
	plt.hist(actions, bins=np.arange(0, args.num_actions+ 1 ))
	plt.title("actions histogram")
	plt.savefig(location + '/histogram.png')
	plt.close()

	array = np.zeros([args.num_actions * 2, args.num_actions * 4, 3])
	for i in range(args.num_actions):
		x, y, z = circle.points[i]
		x = math.atan2(-x, y);
		x = (x + np.pi / 2.0) / (np.pi * 2.0) + np.pi * (28.670 / 360.0);
		y = math.acos(z) / np.pi;

		x_co = int(y * args.num_actions * 12 / (2 * np.pi))
		y_co = int(x * args.num_actions * 24 / (2 * np.pi))
		for i in range(3):
			for j in range(3):
				array[x_co - 1 + i, y_co - 1 + j] += 1.
	for a in actions:
		x, y, z = circle.points[a]
		x = math.atan2(-x, y);
		x = (x + np.pi / 2.0) / (np.pi * 2.0) + np.pi * (28.670 / 360.0);
		y = math.acos(z) / np.pi;

		x_co = int(y * args.num_actions * 12 / (2 * np.pi))
		y_co = int(x * args.num_actions * 24 / (2 * np.pi))
		for i in range(3):
			for j in range(3):
				array[x_co - 1 + i, y_co - 1 + j] += 1.
	array = array * 255. / array.max()

	if args.use_img:
		visible_location = os.path.join(
			os.path.dirname(objects.__file__), "visible.obj"
		)
		seen_points = np.array(load_obj(visible_location)[0])
		seen_points = seen_points / np.sqrt(((seen_points ** 2).sum(axis=1))).reshape(-1, 1)
		for point in seen_points:
			x, y, z = point
			x = math.atan2(-x, y);
			x = (x + np.pi / 2.0) / (np.pi * 2.0) + np.pi * (28.670 / 360.0);
			y = math.acos(z) / np.pi;

			x_co = int(y * args.num_actions * 12 / (2 * np.pi))
			y_co = int(x * args.num_actions * 24 / (2 * np.pi))
			for i in range(5):
				for j in range(5):
					if array[x_co - 2 + i, y_co - 2 + j].sum() == 0:
						array[x_co - 2 + i, y_co - 2 + j] = (255, 127, 80)
		array[np.all(array == (0, 0, 0), axis=-1)] = (0, 204, 204)


		check_array = np.zeros([args.num_actions * 2, args.num_actions * 4])
		for point in seen_points:
			x, y, z = point
			x = math.atan2(-x, y);
			x = (x + np.pi / 2.0) / (np.pi * 2.0) + np.pi * (28.670 / 360.0);
			y = math.acos(z) / np.pi;

			x_co = int(y * args.num_actions * 12 / (2 * np.pi))
			y_co = int(x * args.num_actions * 24 / (2 * np.pi))
			for i in range(3):
				for j in range(3):
					check_array[x_co - 1 + i, y_co - 1 + j] = 100

		on = 0.
		off = 0.
		for a in actions:
			x, y, z = circle.points[a]
			x = math.atan2(-x, y);
			x = (x + np.pi / 2.0) / (np.pi * 2.0) + np.pi * (28.670 / 360.0);
			y = math.acos(z) / np.pi;

			x_co = int(y * args.num_actions * 12 / (2 * np.pi))
			y_co = int(x * args.num_actions * 24 / (2 * np.pi))
			if check_array[x_co, y_co] > 0:
				on += 1
			else:
				off += 1

		print(f'percentage in vision is {on * 100 / (on+off):.2f} % for policy')
	else:
		array[np.all(array == (0, 0, 0), axis=-1)] = (0, 204, 204)
	array = array.astype(np.uint8)
	Image.fromarray(array).save(location + '/sphere_projection.png')






# visualize the actions used by the policy
def visualize_prediction(location, meshes, faces, names):
	data = {}
	meshes = meshes.data.cpu().numpy()
	faces = faces.data.cpu().numpy()
	locations = []
	for n in names:
		n = '/'+ n.split('/')[-1] + '/'
		locations.append(location + n)
		if not os.path.exists(locations[-1]):
			os.makedirs(locations[-1])
	data['locations'] = locations
	pretty_render.render_representations(locations, names, meshes, faces)
