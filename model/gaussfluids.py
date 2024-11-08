#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn

from model.gaussians import Gaussians
from model.transition import MLP
from utils.graphics_utils import BasicPointCloud
from utils.math import ActivationType
from utils.position_encoding import Embedder


class GaussfluidsModel(Gaussians):
    feats: torch.Tensor
    track_channel: int
    base_time: float
    multires: int
    mlp: torch.nn.Module

    def __init__(self,
                 xyz, scaling, rotation, opacity, features_dc, features_rest, feats,
                 active_sh_degree, max_sh_degree=3, channel=3,
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP,
                 base_time=0, track_channel=64, hidden_sizes=[256, 256, 256, 256], multires=4
                 ):
        super().__init__(xyz, scaling, rotation, opacity, features_dc, features_rest,
                         active_sh_degree, max_sh_degree, channel, opacity_activation_type, scaling_activation_type)
        self.feats = feats
        self.track_channel = track_channel
        self.base_time = base_time
        self.multires = multires
        self._embedder = Embedder(multires=self.multires, input_dims=1)
        self.mlp = MLP(track_channel + self._embedder.out_dim, hidden_sizes, 3 + 3 + 4)

    def create_from_pcd(self, pcd: BasicPointCloud, init_color=None):
        super().create_from_pcd(pcd, init_color)
        self.feats = nn.Parameter(torch.zeros((self.get_num, self.track_channel), device='cuda'))

    def get_static(self, time) -> Gaussians:
        dt_time = time - self.base_time
        output = self.mlp(
            torch.concat((self.feats, self._embedder(torch.full(self.feats[:, :1].size(), dt_time, device='cuda'))),
                         dim=-1))
        dt_xyz, dt_scaling, dt_rotation = torch.split(dt_time * output, [3, 3, 4], dim=-1)
        return Gaussians(self.xyz + dt_xyz, self.scaling, self.rotation,
                         self.opacity, self.features_dc, self.features_rest,
                         self.active_sh_degree, self.max_sh_degree, self.channel,
                         self.opacity_activation_type, self.scaling_activation_type)
