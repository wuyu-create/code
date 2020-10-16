import torch
from tqdm import tqdm
from Initial_data_process.inference import inference
from multi_person_tracker import MPT

def data_processing(image_folder):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mot = MPT(
        device=device,
        batch_size=12,
        display=False,
        detector_type=False,
        output_format='dict',
        yolo_img_size=416,
    )
    ## 这是一个实验数据的路径，检测目标检测代码是否正常运行
    tracking_results = mot(image_folder)
    return tracking_results

###
'''
tracking_result的格式是：
{1:{
    'bbox':array([x_1,y_1,w_1,h_1],...,[x_n,y_n,w_n,h_n]),
    'frames':array([1,2,...,n])
    },
 2:{
    'bbox':array([x_1,y_1,w_1,h_1],...,[x_n,y_n,w_n,h_n]),
    'frames':array([1,2,...,n])
    },
 ...
}
其中字典key:1,2,...表示一张图片中出现人的数目
n:表示训练的图片有n
'''
def get_dataset(image_folder,bbox_scale = 1.1):
    tracking_results = data_processing(image_folder)
    data_result= dict()
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
        dataset = inference(
            image_folder=image_folder,
            bboxes=bboxes,
            joints2d=None,
            scale=bbox_scale,
        )
        data_result[person_id] = {
            'dataset': dataset,
            'bboxes': bboxes,
            'frames': frames
        }
        '''
        if person_id in data_result.keys():
            data_result[person_id]['dataset'].append(dataset)
            data_result[person_id]['bboxes'].append(bboxes)
            data_result[person_id]['frames'].append(frames)
        else:
            data_result[person_id] = {
                'dataset':[],
                'bboxes':[],
                'frames':[]
            }
            data_result[person_id]['dataset'].append(dataset)
            data_result[person_id]['bboxes'].append(bboxes)
            data_result[person_id]['frames'].append(frames)
        '''
    return data_result
