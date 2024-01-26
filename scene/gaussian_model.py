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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.trans_model import TransModel


class GaussianFrame:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def modified_sigmoid(x):
            return 0.01 * (torch.sigmoid(x) + 1)

        def modified_sigmoid_inverse(y):
            # 首先，将 y 从 [0.01, 0.02] 转换回 [0, 1]
            original_sigmoid_output = (y / 0.01) - 1

            # 使用 torch.logit 应用逆 sigmoid 变换
            return torch.logit(original_sigmoid_output)

        if self.use_sigmoid_scaling_activation:
            # 然后在你的类中使用这个函数
            self.scaling_activation = modified_sigmoid
            # 在类中使用
            self.scaling_inverse_activation = modified_sigmoid_inverse
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.cfd_activation = torch.sigmoid

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, active_sh_degree, max_sh_degree, _xyz, _vel, _features_dc, _features_rest, _scaling, _rotation,
                 _opacity, _cfd, use_sigmoid_scaling_activation=False):
        self._xyz = _xyz
        self._vel = _vel
        self._features_dc = _features_dc
        self._features_rest = _features_rest
        self._scaling = _scaling
        self._rotation = _rotation
        self._opacity = _opacity
        self._cfd = _cfd

        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = max_sh_degree

        self.use_sigmoid_scaling_activation = use_sigmoid_scaling_activation
        self.setup_functions()

    def add_extra_gaussians(self, other):
        other: GaussianFrame
        self._xyz = torch.concat((self._xyz, other._xyz), dim=0)
        self._vel = torch.concat((self._vel, other._vel), dim=0)
        self._features_dc = torch.concat((self._features_dc, other._features_dc), dim=0)
        self._features_rest = torch.concat((self._features_rest, other._features_rest), dim=0)
        self._scaling = torch.concat((self._scaling, other._scaling), dim=0)
        self._rotation = torch.concat((self._rotation, other._rotation), dim=0)
        self._opacity = torch.concat((self._opacity, other._opacity), dim=0)
        self._cfd = torch.concat((self._cfd, other._cfd), dim=0)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_vel(self):
        return self._vel

    @property
    def get_cfd(self):
        return self.cfd_activation(self._cfd)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_num(self):
        return self._xyz.shape[0]

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'vx', 'vy', 'vz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('cfd')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        vel = self._vel.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        cfd = self._cfd.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, vel, f_dc, f_rest, opacities, cfd, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def clone_detached(self):
        """
        Clones and detaches all tensor attributes of the GaussianFrame instance
        from the computation graph.
        """
        cloned_instance = GaussianFrame(self.active_sh_degree, self.max_sh_degree,
                                        self._xyz.clone().detach(), self._vel.clone().detach(),
                                        self._features_dc.clone().detach(), self._features_rest.clone().detach(),
                                        self._scaling.clone().detach(), self._rotation.clone().detach(),
                                        self._opacity.clone().detach(), self._cfd.clone().detach())
        return cloned_instance


class GaussianModel(GaussianFrame):

    def __init__(self, sh_degree: int, use_sigmoid_scaling_activation=False):
        super().__init__(active_sh_degree=0, max_sh_degree=sh_degree, _xyz=torch.empty(0), _vel=torch.empty(0),
                         _features_dc=torch.empty(0), _features_rest=torch.empty(0), _scaling=torch.empty(0),
                         _rotation=torch.empty(0), _opacity=torch.empty(0), _cfd=torch.empty(0),
                         use_sigmoid_scaling_activation=use_sigmoid_scaling_activation)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._vel,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._cfd,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._vel,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self._cfd,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def fixed_pose(self):
        self._xyz.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        # self._opacity.requires_grad = False

    def fixed_xyz(self):
        self._xyz.requires_grad = False

    def fixed_feature_rest(self):
        self._features_rest.requires_grad = False

    def fixed_feature_dc(self):
        self._features_dc.requires_grad = False

    def set_featrue_dc(self, mask, new_dc):
        new_features_dc = torch.where(mask.unsqueeze(-1), new_dc, self._features_dc.squeeze(1)).unsqueeze(1)
        optimizable_tensors = self.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]

    def reset_feature_rest(self):
        new_features_rest = torch.zeros_like(self._features_rest)
        optimizable_tensors = self.replace_tensor_to_optimizer(new_features_rest, "f_rest")
        self._features_rest = optimizable_tensors["f_rest"]

    def average_color(self):
        new_features_dc = torch.mean(self._features_dc, dim=0, keepdim=True).repeat(self._features_dc.shape[0], 1, 1)
        new_features_rest = torch.mean(self._features_rest, dim=0, keepdim=True).repeat(self._features_rest.shape[0], 1,
                                                                                        1)
        optimizable_tensors = self.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]
        optimizable_tensors = self.replace_tensor_to_optimizer(new_features_rest, "f_rest")
        self._features_rest = optimizable_tensors["f_rest"]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, init_color=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_point_cloud_vel = torch.zeros_like(fused_point_cloud, device="cuda")
        if init_color is None:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = RGB2SH(torch.tensor(np.asarray(init_color)).float().cuda()).unsqueeze(0).repeat(
                fused_point_cloud.shape[0], 1)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        cfd = inverse_sigmoid(0.9 * torch.ones_like(opacities, dtype=torch.float, device='cuda'))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._vel = nn.Parameter(fused_point_cloud_vel.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._cfd = nn.Parameter(cfd.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._vel], 'lr': training_args.velocity_lr_init * self.spatial_lr_scale, "name": "vel"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._cfd], 'lr': training_args.cfd_lr, "name": "cfd"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        vel = np.stack((np.asarray(plydata.elements[0]["vx"]),
                        np.asarray(plydata.elements[0]["vy"]),
                        np.asarray(plydata.elements[0]["vz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        cfd = np.asarray(plydata.elements[0]["cfd"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._vel = nn.Parameter(torch.tensor(vel, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._cfd = nn.Parameter(torch.tensor(cfd, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, trans: TransModel = None):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._vel = optimizable_tensors["vel"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._cfd = optimizable_tensors["cfd"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if trans is not None:
            trans.prune_points(valid_points_mask=valid_points_mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_vel, new_features_dc, new_features_rest, new_opacities, new_cfd,
                              new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "vel": new_vel,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "cfd": new_cfd,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._vel = optimizable_tensors["vel"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._cfd = optimizable_tensors["cfd"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def double_scaling(self, multiplier=2):
        new_scaling = self.scaling_inverse_activation(multiplier * self.get_scaling)
        optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def split_ellipsoids(self, N=2, trans=None, target_radius=None):
        # 计算目标半径
        target_radius = torch.min(self.get_scaling, dim=1).values.mean() if target_radius is None else target_radius
        selected_pts_mask = torch.any(self.get_scaling > 1.8 * target_radius, dim=1)
        # 检测并分裂
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_vel = self._vel[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_cfd = self._cfd[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_vel, new_features_dc, new_features_rest, new_opacity, new_cfd,
                                   new_scaling, new_rotation)

        trans.densify(selected_pts_mask, N)

        # 删除原始需要分裂的椭球（示例）
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, trans)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, trans: TransModel = None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_vel = self._vel[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_cfd = self._cfd[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_vel, new_features_dc, new_features_rest, new_opacity, new_cfd,
                                   new_scaling, new_rotation)

        trans.densify(selected_pts_mask, N)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, trans)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, trans: TransModel = None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_vel = self._vel[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_cfd = self._cfd[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_vel, new_features_dc, new_features_rest, new_opacities, new_cfd,
                                   new_scaling, new_rotation)

        trans.densify(selected_pts_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, trans: TransModel = None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, trans=trans)
        self.densify_and_split(grads, max_grad, extent, trans=trans)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        sh_mask = torch.all(torch.all(self.get_features < 1e-5, dim=-1), dim=-1)
        prune_mask = torch.logical_or(prune_mask, sh_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, trans=trans)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def move_0(self) -> GaussianFrame:
        return GaussianFrame(self.active_sh_degree, self.max_sh_degree, self._xyz, self._vel, self._features_dc,
                             self._features_rest, self._scaling, self._rotation, self._opacity, self._cfd)

    def move(self, dt_xyz, dt_scaling, dt_rotation) -> GaussianFrame:
        return GaussianFrame(self.active_sh_degree, self.max_sh_degree, self._xyz + dt_xyz, self._vel,
                             self._features_dc, self._features_rest, self._scaling,
                             self._rotation, self._opacity, self._cfd)

    def move_detach(self, dt_xyz, dt_scaling, dt_rotation) -> GaussianFrame:
        return GaussianFrame(self.active_sh_degree, self.max_sh_degree, self._xyz.clone().detach() + dt_xyz, self._vel,
                             self._features_dc, self._features_rest, self._scaling.clone().detach() + dt_scaling,
                             self._rotation.clone().detach() + dt_rotation, self._opacity, self._cfd)

    @property
    def is_available(self):
        return False if self.get_num == 0 else True
