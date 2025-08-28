import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from utils._projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self,
        path,                               # 数据集路径
        split='train',
        num_obs=1,                          # 观测数量
        num_action=20,                      # 动作序列长度
        voxel_size=0.005,                   # 点云体素大小
        cam_ids=['750612070851'],           # 相机 ID 列表，表示要使用哪几个相机采集的数据 - 使用不到

        aug=False,                          # 是否进行数据增强
        aug_trans_min=[-0.2, -0.2, -0.2],
        aug_trans_max=[0.2, 0.2, 0.2],
        aug_rot_min=[-30, -30, -30],
        aug_rot_max=[30, 30, 30],
        aug_jitter=False,                           # 是否启用随机噪声 jitter 增强
        aug_jitter_params=[0.4, 0.4, 0.2, 0.1],     # 颜色（RGB）扰动强度、深度扰动强度、点坐标扰动尺度、扰动幅度范围等
        aug_jitter_prob=0.2,                        # jitter
        with_cloud=False                            # 是否附带原始点云数据输出

    ):
        # 我暂时没有想好怎么定义 train val 我感觉对于机械臂来说应该就是diffusion一次输出后续路径的与真实路径的比较
        # 但是模型里面其实存在问题 就是我动作不是连续的，我可能会存在停顿，但是模型预测肯定是连续的
        # 其实我觉得训练diffusion应该使用20个点运行的路径的重叠程度或者两点位置这种来评估，因为对于diffusion是结果导向的，不过具体还是应该看代码

        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)      # path/train
        self.calib_path = os.path.join(path, "calib")   # path/calib - 摄像头npy
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size

        # 数据增强
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob

        self.with_cloud = with_cloud
        
        self.all_demos = sorted(os.listdir(self.data_path)) # 一级文件夹名
        self.num_demos = len(self.all_demos)    # 这直接是数据 - 我还要做taget比较 - 还有时间

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.projectors = {}
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            for cam_id in cam_ids:  # 不同摄像头拍摄的信息
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata - 感觉应该什么都有
                with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                    meta = json.load(f)
                # get frame ids - 过滤fin之后的帧
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                # get calib timestamps
                with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                    calib_timestamp = f.readline().rstrip()
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)  # 进行观测组合
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))  # action打包

                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)

                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]        # 防止前没帧
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after    # 防止后没帧

                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def load_point_cloud(self, colors, depths, cam_id):
        h, w = depths.shape
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]    # 这只支持一个摄像头 index是随机取样的
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')               # 获取PGB图像
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')                   # 末端中心点 ？ 为什么在color里面
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')   # 夹爪命令

        # load camera projector by calib timestamp
        timestamp_path = os.path.join(data_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, timestamp))
        projector = self.projectors[timestamp]  # 这个应该是cam坐标系转base用的 - 但是数据get的时候居然没有控制坐标 ？

        # create color jitter - 数据增强
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths - 存储RGB图像和depth图像
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds - 处理点云图像
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id)  # 获取点云图像
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)    # output - clouds

        # 只需要末端和夹爪
        action_tcps = []
        action_grippers = []

        for frame_id in action_frame_ids:
            # 加载工具中心点（TCP）位姿数据，从对应帧文件中取前七个值（通常是 x, y, z, qw, qx, qy, qz）
            tcp = np.load(os.path.join(tcp_dir, f"{frame_id}.npy"))[:7].astype(np.float32)  # 我是需要从两个文件中分别读取的

            # 使用投影模块将 TCP 从世界坐标系转换到相机坐标系
            # projector.project_tcp_to_camera_coord 应该返回在图像中的投影位置或在相机系的坐标向量
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)  # 这种坐标变化是需要重新写的

            # 加载夹具宽度数据（可能是 gripper 状态），提取第一个元素并解码为实际宽度
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, f"{frame_id}.npy"))[0])  # 我的夹爪是True和False，

            # 将转换后的 TCP 和夹具宽度加入列表
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)

        # 将列表转换为 NumPy 数组，便于后续 batch 处理和模型对接
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        # 对点云进行数据增强（仅在训练阶段且设置了 aug=True 时执行） 是否启用数据增强
        if self.split == 'train' and self.aug:
            clouds, action_tcps = self._augmentation(clouds, action_tcps)

        # 将 TCP 位姿从四元数表示转换为“6D”旋转表示（适用于 Graus et al. 提出的连续旋转表示）
        # 这种表示通常由 rotation matrix 的前两列组成，形成一个连续、无奇异的旋转编码方式 :contentReference[oaicite:1]{index=1}
        action_tcps = xyz_rot_transform(
            action_tcps,
            from_rep="quaternion",
            to_rep="rotation_6d"
        )

        # 将转换后的 TCP （转换后为 [x, y, z, rot6d...]）和 gripper 宽度拼接起来构成整个 action 向量
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis=-1)

        # 对 actions 做归一化处理（通常会标准化位移/各向速度/旋转等，提升训练收敛性）
        actions_normalized = self._normalize_tcp(actions.copy())

        # 体素处理
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # 归一化操作
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,     # 点的空间坐标
            'input_feats_list': input_feats_list,       # 点的特征值
            'action': actions,                          # 动作向量
            'action_normalized': actions_normalized     # 动作归一化向量
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095
