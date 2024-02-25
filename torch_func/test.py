import torch
from torch_func import classify_ball_op  # 替换为你的模块名

def test_classify_ball_cuda():
    # 设置输入参数
    N = 10  # 假设有10个球
    dis_thr = 0.0
    color_thr = 0.5

    # 生成随机输入数据
    center = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    radius = torch.rand(N, device='cuda', dtype=torch.float32)
    color = torch.rand(N, 3, device='cuda', dtype=torch.float32)

    # 调用CUDA函数
    out_label = classify_ball_op(center, radius, color, dis_thr, color_thr)

    # 检查输出
    assert out_label is not None
    assert out_label.size(0) == N
    assert out_label.dtype == torch.int32
    print("Test passed.")

# 运行测试
if __name__ == "__main__":
    test_classify_ball_cuda()
