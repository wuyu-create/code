from transfer_data.threedpw_utils import load_obj
from Initial_data_process.dataset import get_dataset
import numpy as np
THREE_DPW = '../data/3dpw_user'
DATA_NAME = 'train'
def test():
    datadir = load_obj(THREE_DPW, DATA_NAME)
    '''
    datadir的数据格式：
    {'pose':array([]),               shape:(n, 72)
     'shape':array([]),              shape:(n, 10)
     'img_path':array([]),           shape:(N, )
    }
    '''
    '''
    data_result是一个字典：
    {1:
        {
        'dataset':[array([img_1,...,img_n])]
        'bboxes':[array([x_1,y_1,w_1,h_1],...,[x_n,y_n,w_n,h_n])]
        'frames':[array([1,2,...,n])]
        },
     2:..
    }
    (h,w,c)
    '''
    imgs = np.empty([0, 3*224*224])
    pose = np.empty([0, 72])
    shape = np.empty([0, 10])
    i = 0
    len = 0
    for id in range(datadir['img_path'].shape[0]):
        img_path = '../'+ datadir['img_path'][id]
        data_result = get_dataset(img_path)
        i = i + len
        imgs = np.insert(imgs, i, data_result['dataset'], axis=0)
        pose = np.insert(pose, i, data_result['pose'], axis=0)
        shape = np.insert(shape, i, data_result['shape'], axis=0)
        len = data_result['dataset'].shape[0]
    return [imgs,pose,shape]











