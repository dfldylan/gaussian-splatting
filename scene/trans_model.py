import torch
import torch.nn as nn
from arguments import ModelParams, OptimizationParams
from utils.time_utils import TimeSeriesInfo
from utils.position_encoding import get_embedder


class TransModel:
    def __init__(self, args: ModelParams, time_info: TimeSeriesInfo):
        self.base_time = time_info.get_time(args.end_frame)
        self.pos_encoder, dims = get_embedder(multires=4, i=1)
        self.mlp = MLP(args.track_channel + dims, args.hidden_sizes, 3 + 3 + 4)

    def set_optimizer(self, training_args):
        l = [
            {'params': self.feats, 'lr': training_args.track_feat_lr, "name": "track_feats"},
            {'params': list(self.mlp.parameters()), 'lr': training_args.track_mlp_lr, "name": "track_mlp"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def set_model(self, args, points_num: int, training_args: OptimizationParams):
        self.feats = nn.Parameter(torch.zeros((points_num, args.track_channel), device='cuda'))
        self.set_optimizer(training_args)

    def __call__(self, time, *args, **kwargs):
        dt_time = time - self.base_time
        output = self.mlp(torch.concat(
            (self.feats, self.pos_encoder(torch.full(self.feats[:, :1].size(), dt_time, device='cuda')))
            , dim=-1))
        dt_xyz, dt_scaling, dt_rotation = torch.split(dt_time * output, [3, 3, 4], dim=-1)
        return dt_xyz, dt_scaling, dt_rotation

    def capture(self):
        return (
            self.base_time,
            self.feats,
            self.mlp.state_dict(),
            self.optimizer.state_dict(),
        )

    def restore(self, params, training_args, strict=True, reset_time=True):
        if reset_time:
            (self.base_time,
             self.feats,
             mlp_dict,
             opt_dict) = params
        else:
            (_,
             self.feats,
             mlp_dict,
             opt_dict) = params
        self.mlp.load_state_dict(mlp_dict, strict=strict)
        self.set_optimizer(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def prune_points(self, valid_points_mask):
        print("remove {} points".format(torch.sum(~valid_points_mask).cpu().detach().numpy()))
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self.feats = optimizable_tensors["track_feats"]

    def densify(self, selected_pts_mask, N=1):
        print("add {} points".format(2 * torch.sum(selected_pts_mask).cpu().detach().numpy()))
        if isinstance(N, int):
            feats = self.feats[selected_pts_mask].repeat(N, 1)
        else:
            feats = torch.repeat_interleave(self.feats[selected_pts_mask], N, dim=0)
        self.feats = self.cat_tensors_to_optimizer({'track_feats': feats})['track_feats']

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == "track_feats":
                extension_tensor = tensors_dict["track_feats"]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors["track_feats"] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors["track_feats"] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == "track_feats":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors['track_feats'] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors['track_feats'] = group["params"][0]
        return optimizable_tensors


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0], device='cuda'))
        # self.dropout = nn.Dropout(dropout_prob)

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], device='cuda'))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size, device='cuda')
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        for layer in self.layers:
            x = self.leaky_relu(layer(x))
            # x = self.dropout(x)  # Apply dropout after the activation function
        x = self.output_layer(x)
        return x
