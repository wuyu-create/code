# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn

from Models.smpl import VIBE_DATA_DIR
from Models.resnet50 import hmr
from Models.regressor import Regressor


class VIBE_Demo(nn.Module):
    def __init__(
            self,
            batch_size=64,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):
        super(VIBE_Demo, self).__init__()

        self.batch_size = batch_size

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)
        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size [n, w, h, channel]
        n, nc, h, w = input.shape

        feature = self.hmr(input) ##  batch_size * seqlen X 2048

        feature = feature.reshape(n, -1)  ## n X T X 2048
        feature = feature.reshape(-1, feature.size(-1))    ## n*T X 2048

        smpl_output = self.regressor(feature, J_regressor=J_regressor) # J_regressor=None

        # 修改Tensor形状
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(n, -1)   ## (n, 85)
            s['verts'] = s['verts'].reshape(n, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(n, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(n, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(n, -1, 3, 3)## (n, 24, 3, 3)

        return smpl_output
