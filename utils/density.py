import torch
from pytorch3d.ops import knn_points


def poly6_kernel(r, h):
    """
    Poly6核函数，用于密度估计。
    :param r: 粒子间距离。
    :param h: 核函数的影响半径。
    :return: 核函数值。
    """
    result = torch.clip((315 / (64 * torch.pi * h ** 9)) * (h ** 2 - r ** 2) ** 3, 0)
    return result


def compute_density(positions, h=0.2, k=64):
    dists, _, _ = knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    dens = poly6_kernel(torch.sqrt(dists[0, :, 1:] + 1e-8), h)  # [N,k]
    dens = torch.sum(dens, dim=1)  # [N]
    dens = dens + poly6_kernel(torch.zeros_like(positions[0, 0]), h)
    return dens


if __name__ == '__main__':
    # 启用CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建两个随机点云，并设置requires_grad=True以跟踪梯度
    p1 = torch.rand(2, 5, 3, device=device, requires_grad=True)  # Shape: (N=2, P1=5, D=3)
    p2 = torch.rand(2, 5, 3, device=device, requires_grad=True)  # Shape: (N=2, P2=5, D=3)

    # K-最近邻搜索
    K = 3  # 查找每个点的3个最近邻
    dists, idx, _ = knn_points(p1, p2, K=K, return_nn=False)

    # 使用最近邻距离的平方和作为损失函数
    loss = dists.sum()

    # 反向传播
    loss.backward()

    # 检查梯度
    print("Gradient for p1:", p1.grad)
    print("Gradient for p2:", p2.grad)
