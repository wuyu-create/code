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

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
import os.path as osp
import subprocess

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

def generative(save_folder = '/data2/liuxunyu/tmp/test/'):
    image_folder = 'test.jpg'
    os.makedirs(save_folder, exist_ok=True)
    img = cv2.imread(image_folder)
    img_shape = img.shape
    for i in range(30):
        path = save_folder + f'{i:06d}.png'
        cv2.imwrite(path, img)
    num_img = len(os.listdir(save_folder))
    return save_folder,num_img,img_shape

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    video_file = 'test.mp4'
    output_folder = 'output/'
    output_path = os.path.join(output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)
    print(f'the output path is:{output_path}')

    image_folder, num_frames, img_shape=generative()

    print(f'image_folder is:{image_folder},num_frames is:{num_frames},imshape is:{img_shape}')

    orig_height, orig_width = img_shape[:2]
    total_time = time.time()
    # ========= Run tracking ========= #
    #  tracking_method = bbox
    bbox_scale = 1.1
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=12,   # 12
        display=False,                 # true
        detector_type='yolo',          # yolo
        output_format='dict',
        yolo_img_size=416,     #416 * 416
     )
    tracking_results = mot(image_folder)
    print(f'output the result of tracking->{tracking_results}')
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]
    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)
    # ========= Load pretrained weights ========= #
    # 加载预训练模型
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)

    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None
        # bboxes = n X 4
        bboxes = tracking_results[person_id]['bbox']
        ## frames n X 1
        frames = tracking_results[person_id]['frames']
        # 一个已经标准化了的图像 dataset  其总体大小为 T X 224 X 224 X 3
        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,             # none
            scale=bbox_scale,
        )
        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False
        dataloader = DataLoader(dataset, batch_size=64, num_workers=16)

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)            #在dim = 0 增加一维张量  变为： 1 X T x 244 x 244 x 3
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]  # 得到 seqlen是一段时间序列图像

                output = model(batch)[-1]
                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))  # 1*T X 3
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1)) # 1*T X 72
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))# 1*T X 10
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))    #

            pred_cam = torch.cat(pred_cam, dim=0)                  #(T, 3)
            pred_verts = torch.cat(pred_verts, dim=0)              #(T, ,3)
            pred_pose = torch.cat(pred_pose, dim=0)                #(T, 72)
            pred_betas = torch.cat(pred_betas, dim=0)              #(T, 10)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)        #(T, ,3)
            del batch
        # ========= Save results to a pickle file ========= #
        # 由tensor 转换成 numpy
        pred_cam = pred_cam.cpu().numpy()                 #(T, 3)
        pred_verts = pred_verts.cpu().numpy()             #(T, ,3)
        pred_pose = pred_pose.cpu().numpy()               #(T, 72)
        pred_betas = pred_betas.cpu().numpy()             #(T, 10)
        pred_joints3d = pred_joints3d.cpu().numpy()       #(T, ,3)
        # 结果为： T X 4
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,                    # 预测的 cam   T X 3
            bbox=bboxes,                     # 目标检测的结果 T X 4
            img_width=orig_width,            # 最初图像的宽
            img_height=orig_height           # 最初图像的高
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }
        vibe_results[person_id] = output_dict
    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)
    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')
    print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

    if not args.no_render:
        # ========= Render results as a single video ========= #
        # 图像高和宽 resolution
        ## 一个 工具 真确传参数即可使用
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            # 读取图片
            img = cv2.imread(img_fname)

            # true
            if args.sideview:
                side_img = np.zeros_like(img)
            #
            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                mc = mesh_color[person_id]

                # 3D模型
                mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                os.makedirs(mesh_folder, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                # 输出是一张图片
                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                side_img = renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0,1,0],
                )
            img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')
    args = parser.parse_args()
    main(args)
