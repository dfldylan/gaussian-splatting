import torch

from scene.gaussian_model import GaussianFrame
from scene.cconv import MyParticleNetwork as Model
from scene.cconv_box import MyParticleNetwork as Model0
from utils.system_utils import searchForMaxIteration
from collections import namedtuple
import os


class TransModel:
    def __init__(self, time_step, model='cconv', load_iteration=None, model_path=None, box_info=None):
        self.model_path = model_path
        self.time_step = time_step
        self.gravity = torch.tensor([0, -9.81, 0], requires_grad=False, device='cuda')
        self.use_cov3D_feats = True
        self.use_colors_feats = False
        self.box_info = box_info
        if self.box_info:
            self.box, self.box_normals = (torch.tensor(box_info['box']).float().cuda(),
                                          torch.tensor(box_info['box_normals']).float().cuda())
        self.model: Model = self._create_model(model, feats_channels=13, box=self.box_info)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-6, eps=1e-15)

        if load_iteration:
            path = os.path.join(self.model_path, 'ckpt_trans')
            if load_iteration == -1:
                items = [item for item in os.listdir(path) if item.split('.')[-1] == 'pth']
                iters = [int(item.split('.')[0]) for item in items]
                load_iteration = max(iters)
                load_path = os.path.join(path, str(load_iteration) + '.pth')
            else:
                load_path = os.path.join(path, str(load_iteration) + '.pth')
                assert os.path.exists(load_path)
            self.model.load_state_dict(torch.load(load_path))
            print("Loading trained trans model at iteration {}".format(load_iteration))

    def _create_model(self, name, feats_channels, box=None, weights_path=None):
        assert name == 'cconv'
        if box:
            model = Model0(feats_channels=feats_channels)
        else:
            model = Model(feats_channels=feats_channels)
        model.to(torch.device('cuda'))
        if weights_path:
            weights = torch.load(weights_path)
            model.load_state_dict(weights)
        return model

    def trans_with_box(self, gaussians: GaussianFrame, *args, **kwargs) -> GaussianFrame:
        cfd = gaussians.get_cfd

        xyz = gaussians.get_xyz
        vel = gaussians.get_vel

        new_vel = vel + self.time_step * self.gravity
        delta_xyz1 = (self.time_step * (vel + new_vel) / 2)
        new_xyz = xyz + delta_xyz1

        features = torch.cat((gaussians._vel, gaussians._cfd, gaussians._opacity), dim=1)

        if self.use_cov3D_feats:
            features = torch.cat((features, gaussians._scaling, gaussians._rotation), dim=1)

        if self.use_colors_feats:
            features = torch.cat((features, gaussians.get_features.view(-1, 3 * (gaussians.max_sh_degree + 1) ** 2)),
                                 dim=1)

        inputs = new_xyz, features, self.box, self.box_normals

        delta_xyz2 = self.model(inputs)

        delta_xyz = (delta_xyz1 + delta_xyz2)

        new_xyz = xyz + delta_xyz
        new_vel = delta_xyz / self.time_step * 2 - vel

        new_gaussians = GaussianFrame(gaussians.active_sh_degree, gaussians.max_sh_degree, new_xyz, new_vel,
                                      gaussians._features_dc, gaussians._features_rest, gaussians._scaling,
                                      gaussians._rotation, gaussians._opacity, gaussians._cfd)

        return new_gaussians

    def trans(self, gaussians: GaussianFrame, *args, **kwargs) -> GaussianFrame:
        cfd = gaussians.get_cfd

        xyz = gaussians.get_xyz
        vel = gaussians.get_vel

        new_vel = vel + self.time_step * self.gravity
        delta_xyz1 = cfd * (self.time_step * (vel + new_vel) / 2)
        new_xyz = xyz + delta_xyz1

        features = torch.cat((cfd * gaussians._vel, gaussians._cfd, gaussians._opacity), dim=1)

        if self.use_cov3D_feats:
            features = torch.cat((features, gaussians._scaling, gaussians._rotation), dim=1)

        if self.use_colors_feats:
            features = torch.cat((features, gaussians.get_features.view(-1, 3 * (gaussians.max_sh_degree + 1) ** 2)),
                                 dim=1)

        inputs = new_xyz, features

        delta_xyz2 = self.model(inputs)

        delta_xyz = cfd * (delta_xyz1 + delta_xyz2)

        new_xyz = xyz + delta_xyz
        new_vel = delta_xyz / self.time_step * 2 - vel

        new_gaussians = GaussianFrame(gaussians.active_sh_degree, gaussians.max_sh_degree, new_xyz, new_vel,
                                      gaussians._features_dc, gaussians._features_rest, gaussians._scaling,
                                      gaussians._rotation, gaussians._opacity, gaussians._cfd)

        return new_gaussians

    def __call__(self, gaussians: GaussianFrame, *args, **kwargs) -> GaussianFrame:
        if self.box_info:
            return self.trans_with_box(gaussians)
        else:
            return self.trans(gaussians)
