import torch
from typing import Any
import classify_ball

def classify_ball_op(center, radius, color, dis_thr, color_thr):
    return _ClassifyBall.apply(center, radius, color, dis_thr, color_thr)


class _ClassifyBall(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, center, radius, color, dis_thr, color_thr):
        args = (center, radius, color, dis_thr, color_thr)
        label = classify_ball.classify_ball(*args)
        return label

__all__ = ['classify_ball_op']