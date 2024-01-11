import torch
import open3d.ml.torch as o3dml
from open3d.ml.torch.classes.ragged_tensor import RaggedTensor


def poly6_kernel(r, h):
    """
    Poly6核函数，用于密度估计。
    :param r: 粒子间距离。
    :param h: 核函数的影响半径。
    :return: 核函数值。
    """
    result = torch.clip((315 / (64 * torch.pi * h ** 9)) * (h ** 2 - r ** 2) ** 3, 0, torch.inf)
    return result


def compute_density(positions, h):
    """
    使用Open3D的固定半径搜索优化的密度计算函数。
    :param positions: 粒子位置的张量，形状为[-1, 3]。
    :param h: 核函数的影响半径。
    :return: 粒子的密度估计，形状为[-1, 1]。
    """
    # 固定半径搜索
    fixed_radius_search = o3dml.layers.FixedRadiusSearch(index_dtype=torch.int64)
    neighbors_index, neighbors_row_splits, _ = fixed_radius_search(queries=positions, points=positions, radius=h)
    # dens = o3dml.ops.reduce_subarrays_sum(poly6_kernel(torch.sqrt(torch.clamp(_, min=0)), h), neighbors_row_splits)
    # 计算重复次数
    repeats = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
    # 逐元素复制
    positions_repeat = torch.repeat_interleave(positions, repeats,dim=0)
    # 计算密度
    dist = positions[neighbors_index] - positions_repeat
    dens = poly6_kernel(torch.norm(dist, dim=-1), h)  # [-1, None]
    dens = o3dml.ops.reduce_subarrays_sum(dens, neighbors_row_splits)
    return dens


if __name__ == '__main__':
    # 示例用法
    positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 示例粒子位置
    h = 1.5  # 核函数的影响半径
    densities = compute_density(positions, h)
    print(densities)
