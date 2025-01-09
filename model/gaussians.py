import logging
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.graphics_utils import BasicPointCloud
from utils.math import inverse_sigmoid, build_covariance_from_scaling_rotation, ActivationType, \
    activation_functions
from utils.sh_utils import RGB2SH, rgb_str_to_tensor
from utils.system_utils import mkdir_p


class Gaussians:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    active_sh_degree: int
    max_sh_degree: int
    channel: int
    opacity_activation_type: ActivationType
    scaling_activation_type: ActivationType

    def __init__(self,
                 xyz, scaling, rotation, opacity, features_dc, features_rest,
                 active_sh_degree, max_sh_degree=3, channel=3,
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP
                 ):
        self.channel = channel

        self.xyz = xyz
        self.scaling = scaling
        self.rotation = rotation
        self.opacity = opacity
        self.features_dc = features_dc
        self.features_rest = features_rest

        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = max_sh_degree

        self.opacity_activation_type = opacity_activation_type
        self.scaling_activation_type = scaling_activation_type
        self._scaling_activation = activation_functions[self.scaling_activation_type][0]
        self._inverse_scaling_activation = activation_functions[self.scaling_activation_type][1]
        self._opacity_activation = activation_functions[self.opacity_activation_type][0]
        self._inverse_opacity_activation = activation_functions[self.opacity_activation_type][1]

    @property
    def get_num(self):
        return self.xyz.shape[0]

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_scaling(self):
        return self._scaling_activation(self.scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotation)

    def get_covariance(self, scaling_modifier=1):
        return build_covariance_from_scaling_rotation(self.get_scaling, scaling_modifier, self.rotation)

    @property
    def get_opacity(self):
        return self._opacity_activation(self.opacity)

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        if self.channel == 3:
            return torch.cat((features_dc, features_rest), dim=1)
        elif self.channel == 1:
            return torch.cat((features_dc, features_rest), dim=1).repeat(1, 1, 3)
        else:
            raise ValueError('channel must be 1 or 3')

    @property
    def is_available(self):
        return False if self.get_num == 0 else True

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1] * self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1] * self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        dtype_full = [(attribute, 'f4') for attribute in l]

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (self.features_dc.detach().transpose(1, 2).flatten(start_dim=1)
                .contiguous().cpu().numpy())
        f_rest = (self.features_rest.detach().transpose(1, 2).flatten(start_dim=1)
                  .contiguous().cpu().numpy())
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if self.channel == 3:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        elif self.channel == 1:
            features_dc = np.zeros((xyz.shape[0], 1, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == self.channel * (self.max_sh_degree + 1) ** 2 - self.channel
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], self.channel, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self.features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def create_from_pcd(self, pcd: BasicPointCloud, init_color=None):
        channel = self.channel
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if init_color is None:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors[:, :channel])).float().cuda())
        else:
            fused_color = RGB2SH(rgb_str_to_tensor(init_color)).unsqueeze(0).repeat(fused_point_cloud.shape[0], 1)
        features = torch.zeros((fused_color.shape[0], channel, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :, 0] = fused_color
        features[:, :, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = self._inverse_scaling_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.scaling = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))
        self.features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def add_gaussians(self, gs):
        gs: Gaussians
        assert self.opacity_activation_type == gs.opacity_activation_type
        assert self.scaling_activation_type == gs.scaling_activation_type
        assert self.max_sh_degree == gs.max_sh_degree
        if self.active_sh_degree != gs.active_sh_degree:
            self.active_sh_degree = max(gs.active_sh_degree, self.active_sh_degree)
            logging.warning('active_sh_degree mismatch, use {}'.format(self.active_sh_degree))
        self.xyz = torch.concat((self.xyz, gs.xyz), dim=0)
        self.features_dc = torch.concat((self.features_dc, gs.features_dc), dim=0)
        self.features_rest = torch.concat((self.features_rest, gs.features_rest), dim=0)
        self.scaling = torch.concat((self.scaling, gs.scaling), dim=0)
        self.rotation = torch.concat((self.rotation, gs.rotation), dim=0)
        self.opacity = torch.concat((self.opacity, gs.opacity), dim=0)
