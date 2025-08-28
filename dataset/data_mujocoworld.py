import os
import json
import torch
import glob
import random
import h5py
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs
from torch.utils.data import Dataset
from utils.constants import *
from utils._projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
from dataset.data_BatchSampler import PerTableBatchSampler

TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized', "clouds_list"]
class MujocoWorldDataset(Dataset):
    def __init__(
        self,
        path,
        path_time,
        split='train',
        target_obj="banana",
        train_rate=0.8,
        train_seed=2025,
        num_obs=1,
        num_action=20,
        voxel_size=0.005,
        cam_ids='frontview',
        aug=False,
        aug_trans_min=[-0.2, -0.2, -0.2],
        aug_trans_max=[0.2, 0.2, 0.2],
        aug_rot_min=[-30, -30, -30],
        aug_rot_max=[30, 30, 30],
        aug_jitter=False,
        aug_jitter_params=[0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob=0.2,
        with_cloud=False,
        depth_scale = 1.0
    ):
        # split #
        self.split = split
        assert self.split in ['train', 'val']

        # data #
        self.path = path
        self.target_obj = target_obj
        self.data_path = os.path.join(self.path, f"data/{path_time}/{target_obj}")
        file_list = glob.glob(os.path.join(self.data_path, "*.hdf5"))     # hdf5 list

        random.seed(train_seed)                         # seed
        random.shuffle(file_list)
        train_size = int(len(file_list) * train_rate)
        self.data_file = {}
        self.data_file["train"] = file_list[:train_size]
        self.data_file["val"] = file_list[train_size:]

        # calib #
        self.cam_ids = cam_ids
        self.calib_path = os.path.join(self.path, "calib")

        cam_dir = os.path.join(self.calib_path, self.cam_ids)
        self.K = np.load(os.path.join(cam_dir, "Intrinsics.npy"))    # Intrinsics
        self.T = np.load(os.path.join(cam_dir, "Extrinsics.npy"))    # Extrinsics
        self.projector = Projector(cam_dir)

        # other #
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.with_cloud = with_cloud
        self.depth_scale = depth_scale

        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob

        self.data_paths = []
        self.cam_ids = cam_ids
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []

        self.entries = {'train': [], 'val': []}

        for _split in ['train', 'val']:
            for hdf5_path in self.data_file[_split]:
                with h5py.File(hdf5_path, "r") as f:
                    data = f["data"]
                    file_num = data.attrs["total"]      # step
                    env_args = json.loads(data.attrs["env_args"])
                    target_obj = env_args.get("env_kwargs", {}).get("target_obj", None)

                    if target_obj != self.target_obj:
                        raise ValueError(f"<MujocoWorldDataset> target_obj Error!")

                    frame_ids = list(range(file_num))
                    obs_ids_list, act_ids_list = [], []
                    for cur_idx in range(file_num - 1):
                        obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                        act_pad_after = max(0, self.num_action - (file_num - 1 - cur_idx))
                        fb = max(0, cur_idx - self.num_obs + 1)
                        fe = min(file_num, cur_idx + self.num_action + 1)
                        obs_ids = frame_ids[:1] * obs_pad_before + frame_ids[fb:cur_idx + 1]
                        act_ids = frame_ids[cur_idx + 1:fe] + frame_ids[-1:] * act_pad_after
                        obs_ids_list.append(obs_ids)
                        act_ids_list.append(act_ids)

                    self.entries[_split].append({
                        'hdf5_path': hdf5_path,
                        'file_num': file_num,
                        'obs_ids_list': obs_ids_list,  # List[List[int]]
                        'act_ids_list': act_ids_list,  # List[List[int]]
                    })

        self.limit = [len(entry['obs_ids_list']) for entry in self.entries[self.split]]

    def __len__(self):
        total_len = sum(x for x in self.limit)
        return total_len

    # action_aug
    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps

    # action_nor
    def _normalize_tcp(self, tcp_list):
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def _h5_take_allow_repeated(self, dset, ids):
        idxs = np.asarray(ids, dtype=np.int64)
        if idxs.size == 0:
            return np.empty((0,) + dset.shape[1:], dtype=dset.dtype)

        strictly_increasing = np.all(idxs[1:] > idxs[:-1])
        no_dup = (np.unique(idxs).size == idxs.size)

        if strictly_increasing and no_dup:
            return dset[idxs]
        else:
            return np.stack([dset[int(i)] for i in idxs], axis=0)

    def __getitem__(self, index_table):
        index, table_idx = index_table
        index = int((index / len(self)) * self.limit[table_idx])
        entry = self.entries[self.split][table_idx]
        obs_ids = entry['obs_ids_list'][index]
        act_ids = entry['act_ids_list'][index]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        with h5py.File(entry['hdf5_path'], 'r') as f:
            data = f['data']['demo']['obs']
            clouds = []
            for oid in obs_ids:
                image = np.flipud(data[f'{self.cam_ids}_image'][oid])
                depth = np.flipud(data[f'{self.cam_ids}_depth'][oid])

                if self.split == 'train' and self.aug_jitter:
                    image = jitter(image)

                image_o3d = o3d.geometry.Image(image.astype(np.uint8))
                depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    image_o3d, depth_o3d,
                    depth_scale=self.depth_scale,
                    convert_rgb_to_intensity=False
                )

                h, w = image.shape[:2]
                fx, fy = self.K[0, 0], self.K[1, 1]
                cx, cy = self.K[0, 2], self.K[1, 2]
                intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                cloud.transform([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

                cloud = cloud.voxel_down_sample(self.voxel_size)
                points = np.array(cloud.points)
                colors = np.array(cloud.colors)

                # x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
                # y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
                # z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
                # mask = (x_mask & y_mask & z_mask)
                # points = points[mask]
                # colors = colors[mask]
                # colors = (colors - IMG_MEAN) / IMG_STD

                cloud_np = np.concatenate([points, colors], axis=-1)    # cloud -> np
                clouds.append(cloud_np)

            sample_action = {
                'eef_pos': self._h5_take_allow_repeated(data['eef_pos'], act_ids),
                'eef_quat': self._h5_take_allow_repeated(data['eef_quat'], act_ids),
                'gripper_qpos': self._h5_take_allow_repeated(data['gripper_qpos'], act_ids),
                'joint_pos': self._h5_take_allow_repeated(data['joint_pos'], act_ids),
            }

            tcp_P_base = np.hstack([
                sample_action['eef_pos'],
                sample_action['eef_quat']
            ])
            tcp_P_cam_quat = self.projector.to_cam(tcp_P_base)

            # gripper_state
            gripper_state = np.asarray(sample_action['gripper_qpos'], dtype=np.int8).reshape(-1, 1)   # gripper on-1 / gripper off-0

            # action aug
            if self.split == 'train' and self.aug:
                clouds, tcp_P_cam_quat = self._augmentation(clouds, tcp_P_cam_quat)

            # tcp_P_base - 6D / gripper_state
            tcp_P_cam_6D = xyz_rot_transform(
                tcp_P_cam_quat,
                from_rep="quaternion",
                to_rep="rotation_6d"
            )

            actions = np.concatenate([tcp_P_cam_6D, gripper_state], axis=1)
            actions_normalized = self._normalize_tcp(actions.copy())
            actions = torch.from_numpy(actions).float()
            actions_normalized = torch.from_numpy(actions_normalized).float()

            # cloud_voxel
            input_coords_list = []
            input_feats_list = []
            for cloud in clouds:
                # Upd Note. Make coords contiguous.
                coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype=np.int32)
                # Upd Note. API change.
                input_coords_list.append(coords)
                input_feats_list.append(cloud.astype(np.float32))

            # if self.with_cloud:
            #     return input_coords_list, input_feats_list, actions, actions_normalized, clouds
            # else:
            #     return input_coords_list, input_feats_list, actions, actions_normalized

            ret_dict = {
                'input_coords_list': input_coords_list,
                'input_feats_list': input_feats_list,
                'action': actions,
                'action_normalized': actions_normalized
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
