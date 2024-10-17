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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, ActivationType
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_covariance_from_scaling_rotation, activation_functions
from scene.trans_model import TransModel
from utils.tools import generate_random_bool_tensor, classify_mask
from dataclasses import dataclass


@dataclass
class GaussfluidsModel:
    _xyz: np.ndarray
    _scaling: np.ndarray
    _rotation: np.ndarray
    _opacity: np.ndarray
    _features_dc: np.ndarray
    _features_rest: np.ndarray
    active_sh_degree: int
    max_sh_degree: int
    opacity_activation_type: ActivationType
    scaling_activation_type: ActivationType

    def __init__(self,
                 _xyz, _scaling, _rotation, _opacity, _features_dc, _features_rest,
                 active_sh_degree, max_sh_degree=3,
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP
                 ):
        self._xyz = _xyz
        self._scaling = _scaling
        self._rotation = _rotation
        self._opacity = _opacity
        self._features_dc = _features_dc
        self._features_rest = _features_rest

        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = max_sh_degree

        self.opacity_activation_type = opacity_activation_type
        self.scaling_activation_type = scaling_activation_type

    @property
    def scaling_activation(self):
        return activation_functions[self.scaling_activation_type][0]

    @property
    def inverse_scaling_activation(self):
        return activation_functions[self.scaling_activation_type][1]

    @property
    def opacity_activation(self):
        return activation_functions[self.opacity_activation_type][0]

    @property
    def inverse_opacity_activation(self):
        return activation_functions[self.opacity_activation_type][1]

    @property
    def get_num(self):
        return self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    def get_covariance(self, scaling_modifier=1):
        return build_covariance_from_scaling_rotation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def is_available(self):
        return False if self.get_num == 0 else True

    def create_from_pcd(self, pcd: BasicPointCloud, init_color=None):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
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

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        dtype_full = [(attribute, 'f4') for attribute in l]

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (self._features_dc.detach().transpose(1, 2).flatten(start_dim=1)
                .contiguous().cpu().numpy())
        f_rest = (self._features_rest.detach().transpose(1, 2).flatten(start_dim=1)
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
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def add_gaussians(self, gs):
        gs: GaussfluidsModel
        self._xyz = torch.concat((self._xyz, gs._xyz), dim=0)
        self._features_dc = torch.concat((self._features_dc, gs._features_dc), dim=0)
        self._features_rest = torch.concat((self._features_rest, gs._features_rest), dim=0)
        self._scaling = torch.concat((self._scaling, gs._scaling), dim=0)
        self._rotation = torch.concat((self._rotation, gs._rotation), dim=0)
        self._opacity = torch.concat((self._opacity, gs._opacity), dim=0)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


class Gaussfluids(GaussfluidsModel):

    def __init__(self, sh_degree: int,
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP):
        super().__init__(active_sh_degree=0, max_sh_degree=sh_degree, _xyz=torch.empty(0), _scaling=torch.empty(0),
                         _rotation=torch.empty(0), _opacity=torch.empty(0),
                         _features_dc=torch.empty(0), _features_rest=torch.empty(0),
                         opacity_activation_type=opacity_activation_type,
                         scaling_activation_type=scaling_activation_type)
        # grad controller
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.T_sum = torch.empty(0)
        self.T_count = torch.empty(0)

        # hyperparameter
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.optimizer = None
        self.xyz_scheduler_args = None

    def training_setup(self, training_args, spatial_lr_scale: float, position_lr_max_steps):
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.T_sum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.T_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.percent_dense = training_args.percent_dense
        self.spatial_lr_scale = spatial_lr_scale

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)

    def save(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._features_dc,
            self._features_rest,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.T_sum,
            self.T_count,
            self.spatial_lr_scale,
            self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args, position_lr_max_steps):
        (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._features_dc,
            self._features_rest,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            T_sum,
            T_count,
            spatial_lr_scale,
            opt_dict,
        ) = model_args
        self.training_setup(training_args, spatial_lr_scale, position_lr_max_steps)
        self.max_radii2D = max_radii2D
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.T_sum = T_sum
        self.T_count = T_count
        self.optimizer.load_state_dict(opt_dict)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter, T_sum=0, T_count=0):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        self.T_sum += T_sum
        self.T_count += T_count

    def prune_optimizer(self, mask):
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

    def prune_points(self, mask, trans: TransModel = None):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.T_sum = self.T_sum[valid_points_mask]
        self.T_count = self.T_count[valid_points_mask]

        if trans is not None:
            trans.prune_points(valid_points_mask=valid_points_mask)

    def densification_postfix(self, new_xyz, new_scaling, new_rotation,
                              new_opacities, new_features_dc, new_features_rest):
        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacities,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.T_sum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.T_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent, trans: TransModel = None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacities, new_features_dc, new_features_rest)

        if trans is not None:
            trans.densify(selected_pts_mask)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, trans: TransModel = None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        new_xyz = self.cal_split_xyz(selected_pts_mask, N)
        new_scaling = self.inverse_scaling_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacity, new_features_dc, new_features_rest)

        if trans is not None:
            trans.densify(selected_pts_mask, N)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, trans)

    def densify_and_prune(self, max_grad, min_opacity, extent,
                          max_screen_size=None, prune_min_iters=200, prune_min_T=None, trans: TransModel = None):
        if prune_min_T is None:
            prune_mask = (self.denom < prune_min_iters).squeeze()
            self.prune_points(prune_mask, trans=trans)
        else:
            mean_T = self.T_sum / self.T_count
            mean_T[mean_T.isnan()] = 0.0
            prune_mask = torch.logical_or((self.T_count < prune_min_iters), mean_T < prune_min_T).squeeze()
            self.prune_points(prune_mask, trans=trans)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, trans=trans)
        self.densify_and_split(grads, max_grad, extent, trans=trans)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # sh_mask = torch.all(torch.all(self.get_features < 1e-5, dim=-1), dim=-1)
        # prune_mask = torch.logical_or(prune_mask, sh_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, trans=trans)

        torch.cuda.empty_cache()

    def fixed_pose(self):
        self._xyz.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        # self._opacity.requires_grad = False

    def fixed_feature_rest(self):
        self._features_rest.requires_grad = False

    def fixed_feature_dc(self):
        self._features_dc.requires_grad = False

    def reset_opacity(self, value=0.01):
        value = np.clip(value, a_max=0.999, a_min=0.001)
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * value))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def set_opacity(self, value):
        opacities_new = inverse_sigmoid(value)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def double_scaling(self, multiplier=2):
        new_scaling = self.inverse_scaling_activation(multiplier * self.get_scaling)
        optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def prune_points_random(self, num, trans=None):
        mask = generate_random_bool_tensor(self.get_num, num)
        self.prune_points(mask, trans=trans)

    def split_ellipsoids(self, target_radius=None, max_num=200000, N=2, trans=None):
        threshold = target_radius
        selected_pts_mask = torch.any(self.get_scaling > threshold, dim=1)
        if selected_pts_mask.sum() < 1:
            return
        if self.get_num + selected_pts_mask.sum() * (N - 1) > max_num:
            self.prune_points_random(self.get_num + selected_pts_mask.sum() * (N - 1) - max_num, trans=trans)
            selected_pts_mask = torch.any(self.get_scaling > threshold, dim=1)
        new_xyz = self.cal_split_xyz(selected_pts_mask, N)
        new_scaling = self.inverse_scaling_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacity, new_features_dc, new_features_rest)

        if trans is not None:
            trans.densify(selected_pts_mask, N)

        # 删除原始需要分裂的椭球（示例）
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, trans)

    def move_0(self) -> GaussfluidsModel:
        return GaussfluidsModel(self._xyz, self._scaling, self._rotation,
                                self._opacity, self._features_dc, self._features_rest,
                                self.active_sh_degree, self.max_sh_degree)

    def move(self, dt_xyz, dt_scaling, dt_rotation) -> GaussfluidsModel:
        return GaussfluidsModel(self._xyz + dt_xyz, self._scaling, self._rotation,
                                self._opacity, self._features_dc, self._features_rest,
                                self.active_sh_degree, self.max_sh_degree)

    def prune_min_opacity(self, min_opacity, trans=None):
        opacity_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(opacity_mask, trans=trans)

    def prune_district(self, eps=0.075, min_samples=10, first_class=0, trans=None):
        xyz = self.get_xyz.detach().cpu().numpy()
        mask = classify_mask(xyz, eps=eps, min_samples=min_samples, first_class=first_class)
        self.prune_points(~torch.tensor(mask, dtype=torch.bool, device='cuda'), trans=trans)

    def cal_split_xyz(self, selected_pts_mask, N):
        # 构建旋转矩阵和标准差，这部分对两种情况都是通用的
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)

        # 生成均值为0的正态分布样本
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        # 计算新的坐标点
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        return new_xyz

    def split_ball(self, target_radius, max_num=200000, trans: TransModel = None):
        """
        将椭球切割为多个正球
        """
        if self.get_num > max_num:
            return
        scaling = self.get_scaling
        radius = torch.tensor(target_radius).float().cuda()
        split_num = torch.floor(torch.prod(torch.clip(scaling / radius, min=1), dim=1)).int()
        ratio = max_num / torch.sum(split_num)
        if ratio < 1:  # exceed
            split_num = torch.clip(torch.round(split_num * ratio), min=1).int()
        selected_pts_mask = split_num > 1
        if not selected_pts_mask.any():
            return  # 如果没有任何点需要切割，则直接返回
        split_num = split_num[selected_pts_mask]
        N = split_num

        rots = torch.repeat_interleave(build_rotation(self._rotation[selected_pts_mask]), N, dim=0)
        stds = torch.repeat_interleave(self.get_scaling[selected_pts_mask], N, dim=0)

        # 生成均值为0的正态分布样本
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        # 计算新的坐标点
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  torch.repeat_interleave(self.get_xyz[selected_pts_mask], N, dim=0)

        new_scaling = self.inverse_scaling_activation(radius.unsqueeze(0).repeat(torch.sum(split_num), 3))
        new_rotation = torch.repeat_interleave(self.get_rotation[selected_pts_mask], split_num, dim=0)
        new_features_dc = torch.repeat_interleave(self._features_dc[selected_pts_mask], split_num, dim=0)
        new_features_rest = torch.repeat_interleave(self._features_rest[selected_pts_mask], split_num, dim=0)
        new_opacity = torch.repeat_interleave(self._opacity[selected_pts_mask], split_num, dim=0)

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacity, new_features_dc, new_features_rest)

        if trans is not None:
            trans.densify(selected_pts_mask, split_num)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(split_num.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, trans)
