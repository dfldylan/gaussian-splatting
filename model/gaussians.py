import logging
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import get_expon_lr_func
from utils.graphics_utils import BasicPointCloud
from utils.math_utils import ActivationType, inverse_sigmoid, build_rotation
from utils.math_utils import build_covariance_from_scaling_rotation, activation_functions
from utils.optimizer import prune_optimizer, replace_tensor_to_optimizer, cat_tensors_to_optimizer
from utils.sh_utils import RGB2SH, rgb_str_to_tensor
from utils.system_utils import mkdir_p
from utils.tools import generate_random_bool_tensor, classify_mask


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
    is_shared_feature: bool
    is_shared_opacity: bool
    opacity_activation_type: ActivationType
    scaling_activation_type: ActivationType

    def __init__(self,
                 xyz=torch.empty(0), scaling=torch.empty(0), rotation=torch.empty(0),
                 opacity=torch.empty(0), features_dc=torch.empty(0), features_rest=torch.empty(0),
                 active_sh_degree=0, max_sh_degree=3, channel=3,
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP,
                 shared_feature=False, shared_opacity=False,
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

        self.is_shared_feature = shared_feature
        self.is_shared_opacity = shared_opacity

        self.opacity_activation_type = opacity_activation_type
        self.scaling_activation_type = scaling_activation_type
        self._scaling_activation = activation_functions[self.scaling_activation_type][0]
        self._inverse_scaling_activation = activation_functions[self.scaling_activation_type][1]
        self._opacity_activation = activation_functions[self.opacity_activation_type][0]
        self._inverse_opacity_activation = activation_functions[self.opacity_activation_type][1]

        # grad controller
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.T_sum = torch.empty(0)
        self.T_count = torch.empty(0)

        # optimizer_args
        self._percent_dense = 0
        self._xyz_scheduler_args = None

        self.optimizer = None

    def reset_gradient_accum(self):
        self.max_radii2D = torch.zeros((self.get_num), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_num, 1), device="cuda")
        self.denom = torch.zeros((self.get_num, 1), device="cuda")
        self.T_sum = torch.zeros((self.get_num, 1), device="cuda")
        self.T_count = torch.zeros((self.get_num, 1), device="cuda")

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
        rots[:, 0] = 1 # (w,x,y,z)

        opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.scaling = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))
        self.features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.reset_gradient_accum()

    def save(self):
        return (
            self.channel,
            self.xyz,
            self.scaling,
            self.rotation,
            self.opacity,
            self.features_dc,
            self.features_rest,
            self.active_sh_degree,
            self.max_sh_degree,
            self.opacity_activation_type,
            self.scaling_activation_type,
            self.is_shared_opacity,
            self.is_shared_feature,

            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.T_sum,
            self.T_count,

        )

    def restore(self, model_args):
        (
            channel,
            self.xyz,
            self.scaling,
            self.rotation,
            self.opacity,
            self.features_dc,
            self.features_rest,
            self.active_sh_degree,
            sh_degree,
            opacity_activation_type,
            scaling_activation_type,
            is_shared_opacity,
            is_shared_feature,

            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.T_sum,
            self.T_count,

        ) = model_args
        assert channel == self.channel
        assert sh_degree == self.max_sh_degree
        assert opacity_activation_type == self.opacity_activation_type
        assert scaling_activation_type == self.scaling_activation_type
        assert is_shared_opacity == self.is_shared_opacity
        assert is_shared_feature == self.is_shared_feature

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self._xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def build_optimizer_args(self, training_args, spatial_lr_scale: float, position_lr_max_steps: int):
        self._percent_dense = training_args.percent_dense
        self._xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * spatial_lr_scale,
                                                     lr_final=training_args.position_lr_final * spatial_lr_scale,
                                                     lr_delay_mult=training_args.position_lr_delay_mult,
                                                     max_steps=position_lr_max_steps)

        l = [
            {'params': [self.xyz], 'lr': training_args.position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [self.scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self.features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"}
        ]
        return l

    def setup(self, training_args, spatial_lr_scale: float, position_lr_max_steps: int, opt_dict=None):
        l = self.build_optimizer_args(training_args, spatial_lr_scale, position_lr_max_steps)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter, T_sum=0, T_count=0):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        self.T_sum += T_sum
        self.T_count += T_count

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(self.optimizer, valid_points_mask)

        self.xyz = optimizable_tensors["xyz"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        self.opacity = optimizable_tensors["opacity"] if not self.is_shared_opacity else self.opacity
        self.features_dc = optimizable_tensors["f_dc"] if not self.is_shared_feature else self.features_dc
        self.features_rest = optimizable_tensors["f_rest"] if not self.is_shared_feature else self.features_rest

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.T_sum = self.T_sum[valid_points_mask]
        self.T_count = self.T_count[valid_points_mask]

        return optimizable_tensors

    def densification_postfix(self, d):
        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, d)
        self.xyz = optimizable_tensors["xyz"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        self.opacity = optimizable_tensors["opacity"] if not self.is_shared_opacity else self.opacity
        self.features_dc = optimizable_tensors["f_dc"] if not self.is_shared_feature else self.features_dc
        self.features_rest = optimizable_tensors["f_rest"] if not self.is_shared_feature else self.features_rest

        self.reset_gradient_accum()

        return optimizable_tensors

    def build_clone_data(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self._percent_dense * scene_extent)

        new_xyz = self.xyz[selected_pts_mask]
        new_scaling = self.scaling[selected_pts_mask]
        new_rotation = self.rotation[selected_pts_mask]
        new_opacity = self.opacity[selected_pts_mask] if not self.is_shared_opacity else None
        new_features_dc = self.features_dc[selected_pts_mask] if not self.is_shared_feature else None
        new_features_rest = self.features_rest[selected_pts_mask] if not self.is_shared_feature else None

        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }

        return selected_pts_mask, d

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask, d = self.build_clone_data(grads, grad_threshold, scene_extent)

        add_num = d["xyz"].shape[0]
        logging.info("Add {} points, {} points left".format(add_num, self.get_num + add_num))

        self.densification_postfix(d)

    def build_split_grad_data(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask = torch.where(grads >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self._percent_dense * scene_extent)

        d, add_num = self.build_split_mask_data(selected_pts_mask, N)
        return selected_pts_mask, d, add_num

    def build_split_ellipsoids_data(self, target_radius, max_num=200000, N=2):
        threshold = target_radius
        selected_pts_mask = torch.any(self.get_scaling > threshold, dim=1)
        if selected_pts_mask.sum() < 1:
            return selected_pts_mask, None, 0
        if self.get_num + selected_pts_mask.sum() * (N - 1) > max_num:
            self.prune_points_random(self.get_num + selected_pts_mask.sum() * (N - 1) - max_num)
            selected_pts_mask = torch.any(self.get_scaling > threshold, dim=1)
        d, add_num = self.build_split_mask_data(selected_pts_mask, N)
        return selected_pts_mask, d, add_num

    def build_split_mask_data(self, selected_pts_mask, N=2):
        new_xyz = self.cal_split_xyz(selected_pts_mask, N)
        new_scaling = self._inverse_scaling_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N, 1) if not self.is_shared_opacity else None
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N, 1, 1) if not self.is_shared_feature else None
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N, 1,
                                                                         1) if not self.is_shared_feature else None

        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        add_num = new_xyz.shape[0]
        return d, add_num

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask, d, add_num = self.build_split_grad_data(grads, grad_threshold, scene_extent, N)
        self.split_with_mask(selected_pts_mask, d, add_num, N=2)

    def split_ellipsoids(self, target_radius, max_num=200000, N=2):
        selected_pts_mask, d, add_num = self.build_split_ellipsoids_data(target_radius, max_num, N)
        self.split_with_mask(selected_pts_mask, d, add_num, N)

    def split_with_mask(self, selected_pts_mask, d, add_num, N=2):
        if d is None:
            return
        logging.info("Add {} points, {} points left".format(add_num, self.get_num + add_num))
        self.densification_postfix(d)

        # 删除原始需要分裂的椭球（示例）
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent,
                          max_screen_size=None, prune_min_iters=10, prune_min_T=None):
        if prune_min_T is None:
            prune_mask = (self.denom < prune_min_iters).squeeze()
            self.prune_points(prune_mask)
        else:
            mean_T = self.T_sum / self.T_count
            mean_T[mean_T.isnan()] = 0.0
            prune_mask = torch.logical_or((self.T_count < prune_min_iters), mean_T < prune_min_T).squeeze()
            self.prune_points(prune_mask)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((self.get_num), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        self.densify_and_split(padded_grad, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # sh_mask = torch.all(torch.all(self.get_features < 1e-5, dim=-1), dim=-1)
        # prune_mask = torch.logical_or(prune_mask, sh_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

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
        out = self._opacity_activation(self.opacity)
        if self.is_shared_opacity:
            assert out.size(0) == 1
            out = out.expand(self.get_num, -1)
        return out

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        if self.channel == 3:
            out = torch.cat((features_dc, features_rest), dim=1)
        elif self.channel == 1:
            out = torch.cat((features_dc, features_rest), dim=1).repeat(1, 1, 3)
        else:
            raise ValueError('channel must be 1 or 3')
        if self.is_shared_feature:
            assert out.size(0) == 1
            out = out.expand(self.get_num, -1, -1)
        return out

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
        if self.is_shared_feature:
            f_dc = np.repeat(f_dc, xyz.shape[0], 0)
            f_rest = np.repeat(f_rest, xyz.shape[0], 0)
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

        self.is_shared_feature = False
        self.is_shared_opacity = False

    def add_gaussians(self, gs):
        gs: Gaussians
        assert self.opacity_activation_type == gs.opacity_activation_type
        assert self.scaling_activation_type == gs.scaling_activation_type
        assert self.max_sh_degree == gs.max_sh_degree
        assert self.channel == gs.channel
        if self.active_sh_degree != gs.active_sh_degree:
            self.active_sh_degree = max(gs.active_sh_degree, self.active_sh_degree)
            logging.warning('active_sh_degree mismatch, use {}'.format(self.active_sh_degree))
        if self.is_shared_opacity:
            self.opacity = self.opacity.expand(self.get_num, -1)
            self.is_shared_opacity = False
        if self.is_shared_feature:
            self.features_dc = self.features_dc.expand(self.get_num, -1, -1)
            self.features_rest = self.features_rest.expand(self.get_num, -1, -1)
            self.is_shared_feature = False
        self.xyz = torch.concat((self.xyz, gs.xyz), dim=0)
        self.features_dc = torch.concat((self.features_dc, gs.features_dc), dim=0)
        self.features_rest = torch.concat((self.features_rest, gs.features_rest), dim=0)
        self.scaling = torch.concat((self.scaling, gs.scaling), dim=0)
        self.rotation = torch.concat((self.rotation, gs.rotation), dim=0)
        self.opacity = torch.concat((self.opacity, gs.opacity), dim=0)

    def fixed_pose(self):
        self.xyz.requires_grad = False
        self.scaling.requires_grad = False
        self.rotation.requires_grad = False
        # self._opacity.requires_grad = False

    def fixed_feature_rest(self):
        self.features_rest.requires_grad = False

    def fixed_feature_dc(self):
        self.features_dc.requires_grad = False

    def reset_opacity(self, value=0.01):
        value = np.clip(value, a_max=0.999, a_min=0.001)
        opacities_new = self._inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * value))
        optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def set_opacity(self, value):
        opacities_new = inverse_sigmoid(value)
        optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def set_shared_opacity(self, shared_opacity=None):
        if shared_opacity is None:
            shared_opacity = torch.mean(self.opacity, dim=0, keepdim=True).detach()
        self.opacity = shared_opacity
        self.is_shared_opacity = True
        self.no_optimizer()

    def set_shared_feature(self, shared_feature_dc=None):
        if shared_feature_dc is None:
            shared_feature_dc = torch.mean(self.features_dc, dim=0, keepdim=True).detach()
        self.features_dc = shared_feature_dc
        self.features_rest = torch.zeros_like(self.features_rest[0:1])
        self.active_sh_degree = 0
        self.is_shared_feature = True
        self.no_optimizer()

    def double_scaling(self, multiplier=2):
        new_scaling = self._inverse_scaling_activation(multiplier * self.get_scaling)
        optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, new_scaling, "scaling")
        self.scaling = optimizable_tensors["scaling"]

    def prune_points_random(self, num):
        mask = generate_random_bool_tensor(self.get_num, num)
        self.prune_points(mask)

    def prune_min_opacity(self, min_opacity, trans=None):
        opacity_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(opacity_mask)

    def prune_district(self, eps=0.075, min_samples=10, first_class=0):
        xyz = self.get_xyz.detach().cpu().numpy()
        mask = classify_mask(xyz, eps=eps, min_samples=min_samples, first_class=first_class)
        self.prune_points(~torch.tensor(mask, dtype=torch.bool, device='cuda'))

    def cal_split_xyz(self, selected_pts_mask, N):
        # 构建旋转矩阵和标准差，这部分对两种情况都是通用的
        rots = build_rotation(self.rotation[selected_pts_mask]).repeat(N, 1, 1)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)

        # 生成均值为0的正态分布样本
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        # 计算新的坐标点
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        return new_xyz

    def no_optimizer(self):
        self.reset_gradient_accum()
        self.optimizer = None
