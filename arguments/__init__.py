from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelParams():
    # Pipeline parameters
    data_device: str = "cuda"
    sh_degree: int = 1
    white_background: bool = False

    # Model parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    track_channel: int = 64


@dataclass
class PipelineParams():
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    # Pipeline
    dynamics_color: str = None
    random_background: bool = False
    time_scaling: float = 1.0  # for timestep stride


@dataclass
class OptimizationParams():
    # Learning rate parameters
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    feature_lr: float = 0.0025
    opacity_lr: float = 0.01
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    track_feat_lr: float = 0.01
    track_mlp_lr: float = 0.0001
    # Loss parameters
    lambda_dssim: float = 0.2
    lambda_dens: float = 0.1
    lambda_aniso: float = 0.1
    lambda_vol: float = 0.1
    lambda_opacity: float = 0.1
    lambda_feats: float = 0.1
    # Depth Setup
    depth_l1_weight_init = 1.0
    depth_l1_weight_final = 0.01
    # Frame parameters
    start_frame: int = 0
    end_frame: int = -1
    # Iteration parameters
    warm_iterations: int = 10_000
    dynamics_iterations: int = 150_000
    iterations: int = 200_000
    # Threshold
    densify_grad_threshold: float = 0.001
    max_num_points: int = 200_000
    min_opacity: float = 0.005
    max_screen_size: int = 20
    target_radius: float = 0.1

    # # todo move to specific config
    # static_start: int = 0
    # static_end: int = 0
    # eps: float = 1.0
    # bg_iterations: int = 10_000
