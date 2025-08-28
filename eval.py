import faulthandler
faulthandler.enable()

import os, sys
import argparse
import time
import torch
import open3d as o3d
import MinkowskiEngine as ME
from policy.policy import RISE
from utils._projector import Projector
from utils.transformation import xyz_rot_transform
from utils.RealLift import RealLift
from utils.constants import *
from utils.utils import _quat_mul, normalize_quat_xyzw
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper
from robosuite import load_composite_controller_config
from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.controllers.composite.composite_controller import WholeBody

def get_clond(args, rgb, depth, K):
    rgb = np.ascontiguousarray(np.flipud(rgb), dtype=np.uint8)
    depth = np.ascontiguousarray(np.flipud(depth), dtype=np.float32)

    h, w = rgb.shape[:2]
    image_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)  # HxW float32
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    cloud.transform([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    cloud = cloud.voxel_down_sample(args.voxel_size)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    coord_xyz = np.ascontiguousarray(points / args.voxel_size, dtype=np.int32)
    batch_col = np.zeros((coord_xyz.shape[0], 1), dtype=np.int32)
    coords = np.concatenate([batch_col, coord_xyz], axis=1)
    feats = np.concatenate([points, colors], axis=1).astype(np.float32)

    coords_t = torch.from_numpy(coords).int().to(device)
    feats_t = torch.from_numpy(feats).float().to(device)
    cloud_data = ME.SparseTensor(features=feats_t, coordinates=coords_t, device=device)

    return cloud_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    base_path = os.getcwd()
    parser.add_argument('--data_path', type=str, default=f"{base_path}")
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints/2025-08-28_12-51-18")
    parser.add_argument("--environment", type=str, default="RealLift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--controller", type=str, default="BASIC")
    parser.add_argument("--target_obj", type=str, default="banana")
    parser.add_argument("--sys_max_fr", default=100, type=int)
    parser.add_argument('--cam_names', type=str, default='agentview')
    parser.add_argument("--cam_height", default=480, type=int)
    parser.add_argument("--cam_width", default=640, type=int)
    parser.add_argument('--num_obs', type=int, default=1)
    parser.add_argument('--num_action', type=int, default=20)
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--obs_feature_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt = f"{args.data_path}/{args.ckpt_dir}/{args.target_obj}/best_val_seed_2025.ckpt"
    calib_path = f"{args.data_path}/calib/{args.cam_names}"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    K = np.load(os.path.join(calib_path, "Intrinsics.npy"))
    T = np.load(os.path.join(calib_path, "Extrinsics.npy"))
    projector = Projector(calib_path)

    model = RISE(
        num_action=args.num_action,
        input_dim=6,
        obs_feature_dim=args.obs_feature_dim,
        action_dim=10,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    this_dir = os.path.dirname(os.path.abspath(__file__))
    obj_list = [
        os.path.join(this_dir, "model/Assets/banana/banana.xml"),
    ]

    env = suite.make(
        env_name=args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=True,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_names=args.cam_names,
        camera_heights=args.cam_height,
        camera_widths=args.cam_width,
        camera_depths=True,
        reward_shaping=True,
        control_freq=100,
        hard_reset=False,
        table_full_size=(0.8, 0.8, 0.05),
        table_offset=(0.0, 0.0, 0.8),
        grid_edge_length=0.01,  # (m)
        external_objects=obj_list,
        remove_default_object=True,
        target_object_name=args.target_obj
    )

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    while True:
        obs = env.reset()
        rgb = obs[f"{args.cam_names}_image"]
        depth = obs[f"{args.cam_names}_depth"]
        depth = get_real_depth_map(env.sim, depth)
        cloud_data = get_clond(args, rgb, depth, K)
        env.render()

        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([-1], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]
        step_idx = 0
        t0 = time.time()
        active_robot = env.robots[0]
        robot_index = env.robots.index(active_robot)

        while True:
            with torch.no_grad():
                pred_action = model(cloud_data, actions=None, batch_size=1)\
                    .squeeze(0).detach().cpu().numpy()
            T = pred_action.shape[0]

            for i in range(T):
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
                eef_quat = normalize_quat_xyzw(eef_quat)
                for gripper_ac in all_prev_gripper_actions[robot_index]:
                    eef_grip = all_prev_gripper_actions[robot_index][gripper_ac]
                pos_cam = pred_action[i, 0:3]
                rot6d = pred_action[i, 3:9]
                grip = pred_action[i, 9:10]

                tcp_P_cam_6D = np.concatenate([pos_cam, rot6d], axis=-1)
                tcp_P_cam_quat = xyz_rot_transform(tcp_P_cam_6D,
                                                   from_rep="rotation_6d", to_rep="quaternion")
                tcp_P_base_quat = projector.to_base(tcp_P_cam_quat)

                taget_pos = tcp_P_base_quat[0:3]
                taget_rot = tcp_P_base_quat[3:]
                grip_cmd = 1 if grip > 0 else -1
                dpos = (np.asarray(taget_pos) - np.asarray(eef_pos)).astype(np.float32)
                qc = np.asarray(eef_quat, dtype=np.float64)
                qt = np.asarray(taget_rot, dtype=np.float64)
                qe = _quat_mul(qt, np.array([-qc[0], -qc[1], -qc[2], qc[3]], dtype=np.float64))
                qe = qe / (np.linalg.norm(qe) + 1e-12)
                angle = 2.0 * np.arccos(np.clip(qe[3], -1.0, 1.0))
                s = np.sqrt(max(1.0 - qe[3] * qe[3], 0.0))
                axis = np.zeros(3, dtype=np.float32) if s < 1e-8 else (qe[:3] / s).astype(np.float32)
                drot = (axis * angle).astype(np.float32)

                action_dict = {}
                for arm in active_robot.arms:
                    action_dict[arm] = np.concatenate([dpos, drot], axis=0)

                for gripper_ac in all_prev_gripper_actions[robot_index]:
                    all_prev_gripper_actions[robot_index][gripper_ac] = grip_cmd

                env_action = active_robot.create_action_vector(action_dict)

                obs, reward, done, info = env.step(env_action)
                rgb = obs[f"{args.cam_names}_image"]
                depth = obs[f"{args.cam_names}_depth"]
                depth = get_real_depth_map(env.sim, depth)
                cloud_data = get_clond(args, rgb, depth, K)

                env.render()

                if args.sys_max_fr is not None:
                    dt = time.time() - t0
                    rest = 1.0 / args.sys_max_fr - dt
                    if rest > 0:
                        time.sleep(rest)

                step_idx += 1


