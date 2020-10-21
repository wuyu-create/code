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

import torch.nn as nn

from lib.models.spin import Regressor

class VIBE_Demo(nn.Module):
    def __init__(
            self,
            batch_size=64,
    ):
        super(VIBE_Demo, self).__init__()
        self.batch_size = batch_size

        self.regressor = Regressor()

    def forward(self, feature, J_regressor=None):
        # input size [n, fr]
        n = feature.shape[0]

        smpl_output = self.regressor(feature, J_regressor=J_regressor) # J_regressor=None

        # 修改Tensor形状
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(n, -1)       ## (n, 85)
            s['verts'] = s['verts'].reshape(n, -1, 3)    ## (n, 6890, 3)
            s['kp_2d'] = s['kp_2d'].reshape(n, -1, 2)    ## (n, 49, 3)
            s['kp_3d'] = s['kp_3d'].reshape(n, -1, 3)    ## (n, 49, 3)
            s['rotmat'] = s['rotmat'].reshape(n, -1, 3, 3)## (n, 24, 3, 3) # 旋转矩阵

        return smpl_output
