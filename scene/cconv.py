import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np


class MyParticleNetwork(torch.nn.Module):

    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
            gravity=(0, -9.81, 0),
            feats_channels=13,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep
        gravity = torch.FloatTensor(gravity)
        self.register_buffer('gravity', gravity)

        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr) ** 3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0", in_channels=feats_channels, filters=2 * self.layer_channels[0],
                                activation=None)
        self.dense0_fluid = torch.nn.Linear(in_features=feats_channels, out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
        torch.nn.init.zeros_(self.dense0_fluid.bias)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch *= 3
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense.weight)
            torch.nn.init.zeros_(dense.bias)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)

    def compute_correction(self,
                           pos,
                           feats):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = torch.tensor(self.filter_extent)

        feats = [torch.ones_like(pos[:, 0:1]), feats]
        feats = torch.cat(feats, axis=-1)

        self.ans_conv0_fluid = self.conv0_fluid(feats, pos, pos, filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(feats)

        feats = torch.cat([self.ans_conv0_fluid, self.ans_dense0_fluid], axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction

    def forward(self, inputs):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        xyz, feats = inputs

        pos_correction = self.compute_correction(xyz, feats)

        return pos_correction
