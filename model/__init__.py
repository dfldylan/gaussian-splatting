import logging

import numpy as np
import torch
from torch import nn

from model.gaussfluids import GaussfluidsModel
from model.gaussians import Gaussians
from utils.general_utils import get_expon_lr_func
from utils.graphics_utils import BasicPointCloud
from utils.math import inverse_sigmoid, build_rotation, ActivationType
from utils.tools import generate_random_bool_tensor, classify_mask


class Gaussfluids(GaussfluidsModel):

    def __init__(self, sh_degree=3, channel=3, base_time=0, track_channel=64, hidden_sizes=[256, 256, 256, 256],
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP):
        super().__init__(
            xyz=torch.empty(0), scaling=torch.empty(0), rotation=torch.empty(0),
            opacity=torch.empty(0), features_dc=torch.empty(0), features_rest=torch.empty(0),
            feats=torch.empty(0),
            active_sh_degree=0, max_sh_degree=sh_degree, channel=channel,
            opacity_activation_type=opacity_activation_type,
            scaling_activation_type=scaling_activation_type,
            base_time=base_time, track_channel=track_channel, hidden_sizes=hidden_sizes, multires=4)
        # grad controller
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.T_sum = torch.empty(0)
        self.T_count = torch.empty(0)

        self.optimizer = None

        # hyperparameter
        self._percent_dense = 0
        self._xyz_scheduler_args = None

    def create_from_pcd(self, pcd: BasicPointCloud, init_color=None):
        super().create_from_pcd(pcd, init_color)
        self.reset_grad()

    def setup(self, training_args, spatial_lr_scale: float, position_lr_max_steps: int, opt_dict=None):
        self._percent_dense = training_args.percent_dense

        l = [
            {'params': [self.xyz], 'lr': training_args.position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [self.scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self.features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.feats], 'lr': training_args.track_feat_lr, "name": "track_feats"},
            {'params': list(self.mlp.parameters()), 'lr': training_args.track_mlp_lr, "name": "track_mlp"}
        ]

        self._xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * spatial_lr_scale,
                                                     lr_final=training_args.position_lr_final * spatial_lr_scale,
                                                     lr_delay_mult=training_args.position_lr_delay_mult,
                                                     max_steps=position_lr_max_steps)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

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

            self.feats,
            self.track_channel,
            self.base_time,
            self.multires,
            self.mlp.state_dict(),

            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.T_sum,
            self.T_count,

            self.optimizer.state_dict(),
        )

    def restore(self, model_args, strict=True):
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

            self.feats,
            track_channel,
            base_time,
            multires,
            mlp_dict,

            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.T_sum,
            self.T_count,

            opt_dict,
        ) = model_args
        assert channel == self.channel
        assert sh_degree == self.max_sh_degree
        assert opacity_activation_type == self.opacity_activation_type
        assert scaling_activation_type == self.scaling_activation_type
        assert track_channel == self.track_channel
        assert base_time == self.base_time
        assert multires == self.multires

        self.mlp.load_state_dict(mlp_dict, strict=strict)
        return opt_dict

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self._xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter, T_sum=0, T_count=0):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        self.T_sum += T_sum
        self.T_count += T_count

    def prune_optimizer(self, mask):
        prume_num = (~mask).sum()
        logging.info("Prune {} points, {} points left".format(prume_num, self.get_num - prume_num))

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group['params']) == 1:
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
            if len(group["params"]) == 1:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self.xyz = optimizable_tensors["xyz"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        self.opacity = optimizable_tensors["opacity"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.feats = optimizable_tensors["track_feats"]

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.T_sum = self.T_sum[valid_points_mask]
        self.T_count = self.T_count[valid_points_mask]

    def densification_postfix(self, new_xyz, new_scaling, new_rotation,
                              new_opacities, new_features_dc, new_features_rest, feats):
        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacities,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "track_feats": feats,
        }

        add_num = new_xyz.shape[0]
        logging.info("Add {} points, {} points left".format(add_num, self.get_num + add_num))

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.xyz = optimizable_tensors["xyz"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        self.opacity = optimizable_tensors["opacity"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.feats = optimizable_tensors["track_feats"]

        self.reset_grad()

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self._percent_dense * scene_extent)

        new_xyz = self.xyz[selected_pts_mask]
        new_scaling = self.scaling[selected_pts_mask]
        new_rotation = self.rotation[selected_pts_mask]
        new_opacities = self.opacity[selected_pts_mask]
        new_features_dc = self.features_dc[selected_pts_mask]
        new_features_rest = self.features_rest[selected_pts_mask]
        new_feats = self.feats[selected_pts_mask]

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacities, new_features_dc, new_features_rest, new_feats)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_num
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self._percent_dense * scene_extent)

        new_xyz = self.cal_split_xyz(selected_pts_mask, N)
        new_scaling = self._inverse_scaling_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_feats = self.feats[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacity, new_features_dc, new_features_rest, new_feats)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def reset_grad(self):
        self.max_radii2D = torch.zeros((self.get_num), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_num, 1), device="cuda")
        self.denom = torch.zeros((self.get_num, 1), device="cuda")
        self.T_sum = torch.zeros((self.get_num, 1), device="cuda")
        self.T_count = torch.zeros((self.get_num, 1), device="cuda")

    def prune_seg_bg(self):
        prune_mask = (self.xyz_gradient_accum == 0).squeeze()
        self.prune_points(prune_mask)
        self.reset_grad()

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
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # sh_mask = torch.all(torch.all(self.get_features < 1e-5, dim=-1), dim=-1)
        # prune_mask = torch.logical_or(prune_mask, sh_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

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
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def set_opacity(self, value):
        opacities_new = inverse_sigmoid(value)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def double_scaling(self, multiplier=2):
        new_scaling = self._inverse_scaling_activation(multiplier * self.get_scaling)
        optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
        self.scaling = optimizable_tensors["scaling"]

    def prune_points_random(self, num):
        mask = generate_random_bool_tensor(self.get_num, num)
        self.prune_points(mask)

    def split_ellipsoids(self, target_radius=None, max_num=200000, N=2):
        threshold = target_radius
        selected_pts_mask = torch.any(self.get_scaling > threshold, dim=1)
        if selected_pts_mask.sum() < 1:
            return
        if self.get_num + selected_pts_mask.sum() * (N - 1) > max_num:
            self.prune_points_random(self.get_num + selected_pts_mask.sum() * (N - 1) - max_num)
            selected_pts_mask = torch.any(self.get_scaling > threshold, dim=1)
        new_xyz = self.cal_split_xyz(selected_pts_mask, N)
        new_scaling = self._inverse_scaling_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N, 1)
        new_feats = self.feats[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacity, new_features_dc, new_features_rest, new_feats)

        # 删除原始需要分裂的椭球（示例）
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def move_0(self) -> Gaussians:
        return Gaussians(self.xyz, self.scaling, self.rotation,
                         self.opacity, self.features_dc, self.features_rest,
                         self.active_sh_degree, self.max_sh_degree, self.channel,
                         self.opacity_activation_type, self.scaling_activation_type)

    def move(self, dt_xyz, dt_scaling, dt_rotation) -> Gaussians:
        return Gaussians(self.xyz + dt_xyz, self.scaling, self.rotation,
                         self.opacity, self.features_dc, self.features_rest,
                         self.active_sh_degree, self.max_sh_degree, self.channel,
                         self.opacity_activation_type, self.scaling_activation_type)

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

    def split_ball(self, target_radius, max_num=200000):
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

        rots = torch.repeat_interleave(build_rotation(self.rotation[selected_pts_mask]), N, dim=0)
        stds = torch.repeat_interleave(self.get_scaling[selected_pts_mask], N, dim=0)

        # 生成均值为0的正态分布样本
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        # 计算新的坐标点
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  torch.repeat_interleave(self.get_xyz[selected_pts_mask], N, dim=0)

        new_scaling = self._inverse_scaling_activation(radius.unsqueeze(0).repeat(torch.sum(split_num), 3))
        new_rotation = torch.repeat_interleave(self.get_rotation[selected_pts_mask], split_num, dim=0)
        new_features_dc = torch.repeat_interleave(self.features_dc[selected_pts_mask], split_num, dim=0)
        new_features_rest = torch.repeat_interleave(self.features_rest[selected_pts_mask], split_num, dim=0)
        new_opacity = torch.repeat_interleave(self.opacity[selected_pts_mask], split_num, dim=0)
        new_feats = torch.repeat_interleave(self.feats[selected_pts_mask], split_num, dim=0)

        self.densification_postfix(new_xyz, new_scaling, new_rotation,
                                   new_opacity, new_features_dc, new_features_rest, new_feats)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(split_num.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
