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
import random
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset
from lib.core.config import VIBE_DB_DIR

logger = logging.getLogger(__name__)

class Dataset3D(Dataset):
    def __init__(self, set, seqlen, overlap=0., folder=None, dataset_name=None, debug=False):

        self.folder = folder
        self.set = set
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))
        self.debug = debug
        self.db = self.load_db()

    def __len__(self):
        return len(self.db['frame_id'])

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(VIBE_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')
        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_single_item(self, index):

        kp_3d = self.db['joints3D'][index]
        pose  = self.db['pose'][index]
        shape = self.db['shape'][index]
        input = torch.from_numpy(self.db['features'][index]).float()
        # theta shape (85,)
        theta = np.concatenate((np.array([1., 0., 0.]), pose, shape),axis=0)
        target = {
            'features': input,
            'theta': torch.from_numpy(theta).float(), # camera, pose and shape
            'kp_3d': torch.from_numpy(kp_3d).float(), # 3D keypoints
        }
        return target





