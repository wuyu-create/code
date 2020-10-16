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

import sys
sys.path.append('.')

import os
import torch
import pickle as pkl
import os.path as osp
from tqdm import tqdm


from lib.data_utils.kp_utils import *
from lib.core.config import VIBE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR


NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def read_data(folder, set):
    dataset = {
        'shape': [],
        'pose': [],
    }
    # 取出 文件夹下所有pkl文件的名字。
    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles', set))]
    ### 对 set 下每个pkl文件进行操作  set = ['train','test', 'validation']
    for i, seq in tqdm(enumerate(sequences)):
        # 单独取出一个pkl文件
        data_file = osp.join(folder, 'sequenceFiles', set, seq + '.pkl')
        # 读出pkl文件
        data = pkl.load(open(data_file, 'rb'), encoding='latin1')
        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)

        for p_id in range(num_people):
            pose = torch.from_numpy(data['poses'][p_id]).float()
            shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
            dataset['shape'].append(shape.numpy())
            dataset['pose'].append(pose.numpy())
    for k in dataset.keys():
        ## 连接
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)
    return dataset
if __name__ == '__main__':
    dir = 'lib/data_utils/data/3dpw'
    dataset = read_data(dir, 'validation')



