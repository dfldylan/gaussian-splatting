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


def compute_density(positions, h=0.2, k=64):
    positions_cpu = positions.cpu()

    nsearch = o3dml.layers.KNNSearch(index_dtype=torch.int64)
    ans = nsearch(points=positions_cpu, queries=positions_cpu, k=k)

    neighbors_positions = positions[ans.neighbors_index].reshape(-1, k, 3)  # [N,k,3]
    expanded_positions = positions.unsqueeze(1)  # [N,1,3]
    dist = neighbors_positions - expanded_positions  # broadcast [N,k,3]
    dens = poly6_kernel(torch.norm(dist, dim=-1), h)  # [N,k]
    dens = torch.sum(dens, dim=1)  # [N]
    return dens


if __name__ == '__main__':
    import open3d.core as o3c
    import torch
    import numpy as np

    # 确保PyTorch Tensor在GPU上
    device = torch.device('cuda')

    # 创建随机的点云数据作为PyTorch Tensor
    points = torch.rand(100, 3, device=device, dtype=torch.float32)
    query_points = torch.rand(10, 3, device=device, dtype=torch.float32)

    # 使用Open3D进行k-NN搜索
    # 注意：Open3D的k-NN功能可能需要将数据从PyTorch Tensor转换为Open3D格式
    # 这里是直接操作Open3D Tensor的示范
    points_o3d = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(points))
    query_points_o3d = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query_points))

    k = 5
    knn_result = o3c.knn_search(points_o3d, query_points_o3d, k)

    # 注意：结果处理部分依据实际情况编写，这里没有展示结果转回PyTorch Tensor的步骤
    pass
