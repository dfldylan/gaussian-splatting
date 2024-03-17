import open3d as o3d
import numpy as np
import torch




def similarity_mask(vectors, target, threshold=0.9, ret_fixed=False):
    if not isinstance(target, torch.Tensor):
        # Convert the list to a PyTorch tensor
        target = torch.tensor(target, dtype=torch.float32).cuda()

    # Calculate the Euclidean distance between each vector and the target
    bias = (vectors - target)
    dist2 = torch.sum(bias ** 2, axis=1)

    # Create a mask where distances are less than or equal to the threshold
    mask = dist2 <= threshold ** 2

    if ret_fixed:
        normal = bias / torch.sqrt(dist2).unsqueeze(-1)
        fixed = threshold * normal + target
        return mask, fixed

    return mask, None
