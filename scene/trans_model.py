import torch
import torch.nn as nn
from arguments import ModelParams
from utils.time_utils import TimeSeriesInfo


class TransModel:
    def __init__(self, args: ModelParams, time_info: TimeSeriesInfo, points_num:int):
        self.base_time = time_info.get_time(args.base_frame)
        self.feats= nn.Parameter(torch.zeros((points_num,args.track_channel),device='cuda'))
        self.mlp = MLP(args.track_channel + 1, args.hidden_sizes, 3 + 3 + 4)
        self.optimizer = torch.optim.Adam(list(self.mlp.parameters()) + [self.feats],lr=0.0001)

    def __call__(self, time, *args, **kwargs):
        dt_time = time - self.base_time
        output = self.mlp(torch.concat((self.feats, torch.full(self.feats[:, :1].size(), dt_time, device='cuda')), dim=-1))
        dt_xyz, dt_scaling, dt_rotation = torch.split(dt_time * output, [3, 3, 4], dim=-1)
        return dt_xyz, dt_scaling, dt_rotation

    def capture(self):
        return (
            self.base_time,
            self.feats,
            self.mlp.state_dict(),
            self.optimizer.state_dict(),
        )

    def restore(self, params):
        (self.base_time,
         self.feats,
         mlp_dict,
         opt_dict) = params
        self.mlp.load_state_dict(mlp_dict)
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
            x = torch.sigmoid(layer(x))
            # x = self.dropout(x)  # Apply dropout after the activation function
        x = self.output_layer(x)
        return x
