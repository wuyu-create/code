# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
import torch.nn as nn
from utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from .smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS

class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        # pose 的大小为 24 * 6
        npose = 24 * 6
        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )
        '''
           'data/vibe_data/smpl_mean_params.npz'
           缺少文件smpl_mean_params.npz初始化
        '''
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)                    # 1 X 24*6
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)# 1 X 10
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)                         # 1 X 3

        # 向模块添加持久缓冲区。
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)



    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        # 这里的 batch_size =  n * T
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)      # batch_size X 24*6
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)    # batch_size X 10
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)        # batch_size X 3

        '''
          预测的pose,shape以及cam 初始化
        '''
        pred_pose = init_pose                                      # batch_size X 24*6
        pred_shape = init_shape                                    # batch_size X 10
        pred_cam = init_cam                                        # batch_size X 3
        ''' n_iter 迭代次数 '''
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose               # batch_size X 24*6
            pred_shape = self.decshape(xc) + pred_shape            # batch_size X 10
            pred_cam = self.deccam(xc) + pred_cam                  # batch_size X 3

        ''' 经过 rot6d_to_rotmat后 数据形状 为 batch_size * 24 X 3 X 3 '''
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3) # 改变形状后为 (batch_size, 24, 3, 3)

        ''' 通过smpl Regressor模型生成预测smpl模型参数'''
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],                                   # batch_size X 23 X 3 X 3
            global_orient=pred_rotmat[:, 0].unsqueeze(1),                   # batch_size X 1 X 3 X 3
            pose2rot=False                                                  #
        )

        pred_vertices = pred_output.vertices                        #预测定点
        pred_joints = pred_output.joints                            #关节

        ###J_regressor == None
        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)  # pose --> (batxh_size,72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),  # --->(batxh_size, 85)
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)  #
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]