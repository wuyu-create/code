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

import time
import torch
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar


from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter

from lib.utils.eval_utils import (
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(
            self,
            data_loaders,
            generator,
            gen_optimizer,
            end_epoch,
            criterion,
            start_epoch=0,
            lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=1000,
            logdir='output',                    # m模型保存路径
            num_iters_per_epoch=1000,
    ):

        # Prepare dataloaders
        self.valid_loader, self.test_loader = data_loaders

        self.train_3dpw_iter = iter(self.valid_loader)

        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer



        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir



        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.num_iters_per_epoch = num_iters_per_epoch
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        # Single epoch training routine

        losses = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.generator.train()
        start = time.time()
        summary_string = ''
        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            # Dirty solution to reset an iterator
            target_3dpw =  None
            if self.train_3dpw_iter:
                try:
                    target_3dpw = next(self.train_3dpw_iter)
                except StopIteration:
                    self.train_3dpw_iter = iter(self.valid_loader)
                    target_3dpw = next(self.train_3dpw_iter)
            move_dict_to_device(target_3dpw, self.device)
            timer['data'] = time.time() - start
            start = time.time()
            inp = target_3dpw['features']

            preds = self.generator(inp)            #dict = {’pose‘:}

            timer['forward'] = time.time() - start
            start = time.time()

            gen_loss = self.criterion(
                generator_outputs=preds,
                real_pose = target_3dpw['pose'],
                real_shape=target_3dpw['shape'],
                real_joints3D = target_3dpw['joints3D']
            )
            # =======>

            timer['loss'] = time.time() - start
            start = time.time()

            # <======= Backprop generator and discriminator
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            # <======= Log training info
            total_loss = gen_loss

            losses.update(total_loss.item(), inp.size(0))

            timer['backward'] = time.time() - start

            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'



            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>
        bar.finish()
        logger.info(summary_string)

    def validate(self):
        self.generator.eval()
        start = time.time()
        summary_string = ''
        bar = Bar('Validation', fill='#', max=len(self.valid_loader))
        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.valid_loader):

            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                inp = target['features']

                preds = self.generator(inp, J_regressor=J_regressor)

                # convert to 14 keypoint format for evaluation
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].view(-1, 85).cpu().numpy()


                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)
            # =============>

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)

            # log the learning rate
            for param_group in self.gen_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(epoch)

            if performance > 80.0:
                exit(f'MPJPE error is {performance}, higher than 80.0. Exiting!...')

        self.writer.close()

    def save_model(self,epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
        }
        filename = osp.join(self.logdir, f'the_{epoch}_checkpoint.pth.tar')
        torch.save(save_dict, filename)

    def evaluate(self):
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float() #

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis
        # Absolute error (MPJPE)
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)                     #
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()  #
        m2mm = 1000
        pa_mpjpe = np.mean(errors_pa) * m2mm  #
        return pa_mpjpe
