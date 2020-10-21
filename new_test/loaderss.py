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

from torch.utils.data import DataLoader
from lib.dataset import *


def get_data_loaders(cfg):

    # ===== 2D keypoint datasets =====
    # ===== Evaluation dataset =====

    train_db = eval(cfg.TRAIN.DATASET_EVAL)(set='train', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    valid_loader = DataLoader(
        dataset=train_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(set='val', seqlen=1, debug=cfg.DEBUG)
    test_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    return valid_loader, test_loader

