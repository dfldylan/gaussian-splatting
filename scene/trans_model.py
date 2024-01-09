import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.position_encoding import get_embedder
from arguments import ModelParams
from scene.dataset_readers import TimeSeriesInfo


class TransModel:
    def __init__(self, args: ModelParams, time_info: TimeSeriesInfo):
        if args.hidden_sizes is None:
            args.hidden_sizes = [64, 32, 16]
        self.base_time = time_info.get_time(args.base_frame)
        self.multires = args.multires
        self.time_encoding = args.time_encoding
        if self.time_encoding:
            self.embed, embedder_dim = get_embedder(self.multires, i=4)
            self.mlp = MLP(embedder_dim, args.hidden_sizes, 3 + 3 + 4)
        else:
            self.embed, embedder_dim = get_embedder(self.multires)
            self.mlp = MLP(embedder_dim + 1, args.hidden_sizes, 3 + 3 + 4)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.0000016, eps=1e-15)

    def __call__(self, time, xyz, *args, **kwargs):
        dt_time = time - self.base_time
        if self.time_encoding:
            embedder_xyz = self.embed(
                torch.concat((xyz, torch.full(xyz[:, :1].size(), dt_time, device='cuda')), dim=-1))
            output = self.mlp(embedder_xyz)
        else:
            embedder_xyz = self.embed(xyz)  # [-1, embedder_dim]
            output = self.mlp(
                torch.concat((embedder_xyz, torch.full(embedder_xyz[:, :1].size(), dt_time, device='cuda')), dim=-1))
        dt_xyz, dt_scaling, dt_rotation = torch.split(dt_time * output, [3, 3, 4], dim=-1)
        return dt_xyz, dt_scaling, dt_rotation

    def capture(self):
        return (
            self.base_time,
            self.multires,
            self.mlp,
            self.optimizer.state_dict(),
        )

    def restore(self, params):
        (self.base_time,
         self.multires,
         self.mlp,
         opt_dict) = params
        self.embed, _ = get_embedder(self.multires)
        self.optimizer.load_state_dict(opt_dict)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0], device='cuda'))
        # self.dropout = nn.Dropout(dropout_prob)

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], device='cuda'))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size, device='cuda')

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            # x = self.dropout(x)  # Apply dropout after the activation function
        x = self.output_layer(x)
        return x
