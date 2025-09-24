from enum import IntEnum

import torch


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_qvec_from_rotation(R):
    # R: [N,3,3] or [3,3]
    # 输出 q: [N,4] or [4] (w,x,y,z)
    
    # 处理单个矩阵输入
    is_single = R.dim() == 2
    if is_single:
        R = R.unsqueeze(0)

    N = R.shape[0]
    q = torch.empty((N, 4), dtype=R.dtype, device=R.device)

    # 提取对角线元素
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask_t_pos = t > 0
    if torch.any(mask_t_pos):
        R_sub = R[mask_t_pos]
        t_sub = t[mask_t_pos]
        
        s = 0.5 / torch.sqrt(t_sub + 1.0)
        w = 0.25 / s
        x = (R_sub[:, 2, 1] - R_sub[:, 1, 2]) * s
        y = (R_sub[:, 0, 2] - R_sub[:, 2, 0]) * s
        z = (R_sub[:, 1, 0] - R_sub[:, 0, 1]) * s
        q[mask_t_pos] = torch.stack([w, x, y, z], dim=1)

    # Case 2, 3, 4: trace <= 0
    mask_t_neg = ~mask_t_pos
    if torch.any(mask_t_neg):
        R_sub = R[mask_t_neg]
        
        # Case 2: R[0,0] is the largest diagonal component
        mask_c2 = (R_sub[:, 0, 0] > R_sub[:, 1, 1]) & (R_sub[:, 0, 0] > R_sub[:, 2, 2])
        if torch.any(mask_c2):
            R_c2 = R_sub[mask_c2]
            s = 2.0 * torch.sqrt(1.0 + R_c2[:, 0, 0] - R_c2[:, 1, 1] - R_c2[:, 2, 2])
            w = (R_c2[:, 2, 1] - R_c2[:, 1, 2]) / s
            x = 0.25 * s
            y = (R_c2[:, 0, 1] + R_c2[:, 1, 0]) / s
            z = (R_c2[:, 0, 2] + R_c2[:, 2, 0]) / s
            q[mask_t_neg][mask_c2] = torch.stack([w, x, y, z], dim=1)

        # Case 3: R[1,1] is the largest diagonal component
        mask_c3 = (R_sub[:, 1, 1] > R_sub[:, 2, 2]) & ~mask_c2
        if torch.any(mask_c3):
            R_c3 = R_sub[mask_c3]
            s = 2.0 * torch.sqrt(1.0 + R_c3[:, 1, 1] - R_c3[:, 0, 0] - R_c3[:, 2, 2])
            w = (R_c3[:, 0, 2] - R_c3[:, 2, 0]) / s
            x = (R_c3[:, 0, 1] + R_c3[:, 1, 0]) / s
            y = 0.25 * s
            z = (R_c3[:, 1, 2] + R_c3[:, 2, 1]) / s
            q[mask_t_neg][mask_c3] = torch.stack([w, x, y, z], dim=1)

        # Case 4: R[2,2] is the largest diagonal component
        mask_c4 = ~mask_c2 & ~mask_c3
        if torch.any(mask_c4):
            R_c4 = R_sub[mask_c4]
            s = 2.0 * torch.sqrt(1.0 + R_c4[:, 2, 2] - R_c4[:, 0, 0] - R_c4[:, 1, 1])
            w = (R_c4[:, 1, 0] - R_c4[:, 0, 1]) / s
            x = (R_c4[:, 0, 2] + R_c4[:, 2, 0]) / s
            y = (R_c4[:, 1, 2] + R_c4[:, 2, 1]) / s
            z = 0.25 * s
            q[mask_t_neg][mask_c4] = torch.stack([w, x, y, z], dim=1)

    # 确保 w >= 0
    q[q[:, 0] < 0] *= -1.0

    # 如果输入是单个矩阵，则返回单个向量
    if is_single:
        return q.squeeze(0)
    return q


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


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


class ActivationType(IntEnum):
    EXP = 0
    SIGMOID = 1
    MODIFIED_SIGMOID = 2

# 定义激活函数和对应逆函数
activation_functions = [
    (torch.exp, torch.log),
    (torch.sigmoid, inverse_sigmoid),
    (modified_sigmoid, modified_sigmoid_inverse)
]

import numpy as np


def max_distance(vectors):
    if len(vectors) < 2:
        return None, None, 0  # Not enough vectors to compare

    max_dist = 0
    vec1 = vec2 = None

    # Calculate the distance between each pair of vectors
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = np.linalg.norm(np.array(vectors[i]) - np.array(vectors[j]))
            if dist > max_dist:
                max_dist = dist
                vec1 = vectors[i]
                vec2 = vectors[j]

    return vec1, vec2, max_dist

