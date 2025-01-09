import torch
from torch import nn

from model.gaussians import Gaussians
from model.mlp import MLP
from utils.graphics_utils import BasicPointCloud
from utils.math import ActivationType
from utils.position_encoding import Embedder


class Gaussfluids(Gaussians):
    feats: torch.Tensor
    track_channel: int
    base_time: float
    multires: int
    mlp: torch.nn.Module

    def __init__(self, sh_degree=3, channel=3, base_time=0, track_channel=64, hidden_sizes=[256, 256, 256, 256],
                 opacity_activation_type=ActivationType.SIGMOID, scaling_activation_type=ActivationType.EXP,
                 shared_feature=False, shared_opacity=False, ):
        super().__init__(xyz=torch.empty(0), scaling=torch.empty(0), rotation=torch.empty(0),
                         opacity=torch.empty(0), features_dc=torch.empty(0), features_rest=torch.empty(0),
                         active_sh_degree=0, max_sh_degree=sh_degree, channel=channel,
                         opacity_activation_type=opacity_activation_type,
                         scaling_activation_type=scaling_activation_type,
                         shared_feature=shared_feature, shared_opacity=shared_opacity)
        self.feats = torch.empty(0)
        self.track_channel = track_channel
        self.base_time = base_time
        self.multires = 4
        self._embedder = Embedder(multires=self.multires, input_dims=1)
        self.mlp = MLP(track_channel + self._embedder.out_dim, hidden_sizes, 3 + 3 + 4)

        self.optimizer = None

    def create_from_pcd(self, pcd: BasicPointCloud, init_color=None):
        super().create_from_pcd(pcd, init_color)
        self.feats = nn.Parameter(torch.zeros((self.get_num, self.track_channel), device='cuda'))

    def save(self):
        _super = super().save()
        return _super, (

            self.feats,
            self.track_channel,
            self.base_time,
            self.multires,
            self.mlp.state_dict(),
        )

    def restore(self, model_args, strict=True):
        _super, (

            self.feats,
            track_channel,
            base_time,
            multires,
            mlp_dict,
        ) = model_args
        super().restore(_super)
        assert track_channel == self.track_channel
        assert base_time == self.base_time
        assert multires == self.multires

        self.mlp.load_state_dict(mlp_dict, strict=strict)

    def build_optimizer_args(self, training_args, spatial_lr_scale: float, position_lr_max_steps: int):
        l = super().build_optimizer_args(training_args, spatial_lr_scale, position_lr_max_steps)
        l += [
            {'params': [self.feats], 'lr': training_args.track_feat_lr, "name": "track_feats"},
            {'params': list(self.mlp.parameters()), 'lr': training_args.track_mlp_lr, "name": "track_mlp"}
        ]
        return l

    def prune_points(self, mask):
        optimizable_tensors = super().prune_points(mask)
        self.feats = optimizable_tensors["track_feats"]
        return optimizable_tensors

    def densification_postfix(self, d):
        optimizable_tensors = super().densification_postfix()
        self.feats = optimizable_tensors["track_feats"]
        return optimizable_tensors

    def build_clone_data(self, grads, grad_threshold, scene_extent):
        selected_pts_mask, d = super().build_clone_data(grads, grad_threshold, scene_extent)
        new_feats = self.feats[selected_pts_mask]
        d["track_feats"] = new_feats
        return selected_pts_mask, d

    def get_static(self, time) -> Gaussians:
        dt_time = time - self.base_time
        output = self.mlp(
            torch.concat((self.feats, self._embedder(torch.full(self.feats[:, :1].size(), dt_time, device='cuda'))),
                         dim=-1))
        dt_xyz, dt_scaling, dt_rotation = torch.split(dt_time * output, [3, 3, 4], dim=-1)
        return Gaussians(self.xyz + dt_xyz, self.scaling, self.rotation,
                         self.opacity, self.features_dc, self.features_rest,
                         self.active_sh_degree, self.max_sh_degree, self.channel,
                         self.opacity_activation_type, self.scaling_activation_type,
                         self.is_shared_feature, self.is_shared_opacity)

    def move_0(self) -> Gaussians:
        return Gaussians(self.xyz, self.scaling, self.rotation,
                         self.opacity, self.features_dc, self.features_rest,
                         self.active_sh_degree, self.max_sh_degree, self.channel,
                         self.opacity_activation_type, self.scaling_activation_type,
                         self.is_shared_feature, self.is_shared_opacity)

    def move(self, dt_xyz, dt_scaling, dt_rotation) -> Gaussians:
        return Gaussians(self.xyz + dt_xyz, self.scaling, self.rotation,
                         self.opacity, self.features_dc, self.features_rest,
                         self.active_sh_degree, self.max_sh_degree, self.channel,
                         self.opacity_activation_type, self.scaling_activation_type,
                         self.is_shared_feature, self.is_shared_opacity)

    def build_split_mask_data(self, selected_pts_mask, N=2):
        d, add_num = super().build_split_mask_data(selected_pts_mask, N)
        new_feats = self.feats[selected_pts_mask].repeat(N, 1)
        d["track_feats"] = new_feats
        return d, add_num

    # def split_ball(self, target_radius, max_num=200000):
    #     """
    #     将椭球切割为多个正球
    #     """
    #     if self.get_num > max_num:
    #         return
    #     scaling = self.get_scaling
    #     radius = torch.tensor(target_radius).float().cuda()
    #     split_num = torch.floor(torch.prod(torch.clip(scaling / radius, min=1), dim=1)).int()
    #     ratio = max_num / torch.sum(split_num)
    #     if ratio < 1:  # exceed
    #         split_num = torch.clip(torch.round(split_num * ratio), min=1).int()
    #     selected_pts_mask = split_num > 1
    #     if not selected_pts_mask.any():
    #         return  # 如果没有任何点需要切割，则直接返回
    #     split_num = split_num[selected_pts_mask]
    #     N = split_num
    #
    #     rots = torch.repeat_interleave(build_rotation(self.rotation[selected_pts_mask]), N, dim=0)
    #     stds = torch.repeat_interleave(self.get_scaling[selected_pts_mask], N, dim=0)
    #
    #     # 生成均值为0的正态分布样本
    #     means = torch.zeros((stds.size(0), 3), device="cuda")
    #     samples = torch.normal(mean=means, std=stds)
    #
    #     # 计算新的坐标点
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
    #               torch.repeat_interleave(self.get_xyz[selected_pts_mask], N, dim=0)
    #
    #     new_scaling = self._inverse_scaling_activation(radius.unsqueeze(0).repeat(torch.sum(split_num), 3))
    #     new_rotation = torch.repeat_interleave(self.get_rotation[selected_pts_mask], split_num, dim=0)
    #     new_features_dc = torch.repeat_interleave(self.features_dc[selected_pts_mask], split_num, dim=0)
    #     new_features_rest = torch.repeat_interleave(self.features_rest[selected_pts_mask], split_num, dim=0)
    #     new_opacity = torch.repeat_interleave(self.opacity[selected_pts_mask], split_num, dim=0)
    #     new_feats = torch.repeat_interleave(self.feats[selected_pts_mask], split_num, dim=0)
    #
    #     d = {
    #         "xyz": new_xyz,
    #         "scaling": new_scaling,
    #         "rotation": new_rotation,
    #         "opacity": new_opacity,
    #         "f_dc": new_features_dc,
    #         "f_rest": new_features_rest,
    #         "track_feats": new_feats,
    #     }
    #     add_num = new_xyz.shape[0]
    #     logging.info("Add {} points, {} points left".format(add_num, self.get_num + add_num))
    #
    #     self.densification_postfix(d)
    #
    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(split_num.sum(), device="cuda", dtype=bool)))
    #     self.prune_points(prune_filter)
