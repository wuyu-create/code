# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import os
import pickle as pkl
import os.path as osp
from tqdm import tqdm

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def read_data(folder, set):

    # 取出 文件夹下所有pkl文件的名字。
    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles', set))]

    for i, seq in tqdm(enumerate(sequences)):

        # 单独取出一个pkl文件
        data_file = osp.join(folder, 'sequenceFiles', set, seq + '.pkl')

        # 读出pkl文件
        data = pkl.load(open(data_file, 'rb'), encoding='latin1')
        print(type(data))
        for key in data.key():
            print(f'the key is{key},the shape is{data[key].shape}')

if __name__ == '__main__':
    dir = 'lib/data_utils/data/3dpw'
    read_data(dir, 'validation')

