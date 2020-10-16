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
import numpy as np

def read_data(folder, set):
    dataset = {
        'shape': [],
        'pose': [],
        'img_path':[],
    }
    # 取出 文件夹下所有pkl文件的名字。
    '''sequences代表该测试集下所有pkl文件的名称'''
    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles', set))]

    ### 对 set 下每个pkl文件进行操作  set = ['train','test', 'validation']
    for i, seq in tqdm(enumerate(sequences)):
        '''seq 单独一个pkl文件的名字'''
        # 单独取出一个pkl文件
        data_file = osp.join(folder, 'sequenceFiles', set, seq + '.pkl')
        # 读出pkl文件
        data = pkl.load(open(data_file, 'rb'), encoding='latin1')

        # 图片中人数的大小
        num_people = len(data['poses'])
        if num_people > 1:
            continue
        '''保存初始图片的文件夹名称与seq是一样的'''
        img_dir = osp.join(folder, 'imageFiles', seq)
        dataset['img_path'].append(img_dir)
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)
        for p_id in range(num_people):
            pose = torch.from_numpy(data['poses'][p_id]).float()
            shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
            dataset['shape'].append(shape.numpy())
            dataset['pose'].append(pose.numpy())
            '''
            img_paths = []
            for id in range(num_frames):
                path = osp.join(img_dir+'/image_{:05d}.jpg'.format(id))
                img_paths.append(path)
            img_path_array = np.array(img_paths)
            dataset['img_path'].append(img_path_array)
            '''

    for k in dataset.keys():
        ## 连接
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)
    return dataset

import pickle
def save_obj(dic, path, name):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
    with open(path + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
'''
dateset的数据格式：
{'pose':array([]),               shape:(n, 72)
 'shape':array([]),              shape:(n, 10)
 'img_path':array([]),           shape:(n, )
}
'''
if __name__ == '__main__':
    dir = 'data/3dpw'
    save = 'data/3dpw_user'
    ## 保存验证数据
    dataset = read_data(dir, 'validation')
    save_obj(dataset,save,'vali')
    ## 保存训练数据
    dataset = read_data(dir, 'train')
    save_obj(dataset, save, 'train')
    ## 保存测试数据
    dataset = read_data(dir, 'test')
    save_obj(dataset, save, 'test')