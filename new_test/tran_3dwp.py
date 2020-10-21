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
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.core.losser import VIBELoss
from lib.core.trainer_3dpw import Trainer
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir
from lib.models import VIBE_Demo
from lib.dataset.loaderss import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer



def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg)


    # ========= Compile Loss ========= #
    loss = VIBELoss(
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
    )
    # ========= Initialize networks, optimizers and lr_schedulers ========= #

    # temporal generator include the neural network ResNet50,temporal encoder and smpl regressor
    # VIBE
    generator = VIBE_Demo(
        batch_size=cfg.TRAIN.BATCH_SIZE,            # 小批量训练 64
    ).to(cfg.DEVICE)


    #  定义generator 模型优化算法  常见的优化算法有: sgd,adam
    gen_optimizer = get_optimizer(
        model=generator,                 # 模型
        optim_type=cfg.TRAIN.GEN_OPTIM,  # 使用优化算法的类型，有：sgd,adam
        lr=cfg.TRAIN.GEN_LR,             # 学习率
        weight_decay=cfg.TRAIN.GEN_WD,   #  regularization 超参数设置
        momentum=cfg.TRAIN.GEN_MOMENTUM, #  动量法 超参数
    )

    # generator 中学习率的调整
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    # ========= Start Training ========= #
    Trainer(
        data_loaders=data_loaders,
        generator=generator,
        criterion=loss,
        gen_optimizer=gen_optimizer,
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        logdir=cfg.LOGDIR,
        lr_scheduler=lr_scheduler,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
        debug_freq=cfg.DEBUG_FREQ,
    ).fit()
if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    #print(cfg)
    main(cfg)


