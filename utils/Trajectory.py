import os
import json
import h5py
import numpy as np
import xml.etree.ElementTree as ET
from robosuite.utils.camera_utils import get_real_depth_map

class TrajectoryDataCollector:

    def __init__(self, env, output_dir="data", target_obj=None):

        self.env = env
        self.target_obj = target_obj
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.control_freq = getattr(env, "control_freq", None)
        if self.control_freq is None:
            base_env = getattr(env, "env", env)
            self.control_freq = getattr(base_env, "control_freq", 100)

        self._reset_buffers()

    def _reset_buffers(self):
        self.step_count = 0

        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

        self.current_target = None

    def _read_model_name(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return root.attrib.get("model", None)

    def start_demonstration(self, initial_obs, env=None):

        self._reset_buffers()
        camera_names = self.env.camera_names

        if self.target_obj is None:
            raise ValueError("<TrajectoryDataCollector> target_object Error!")
        else:
            if self.target_obj not in [self._read_model_name(xml_file) for xml_file in self.env.external_objects]:
                raise ValueError(f"<TrajectoryDataCollector> target_object -2 Error!")

            self.current_target = self.target_obj

        if camera_names is None or (isinstance(camera_names, (list, str)) and len(camera_names)==0):
            raise ValueError("<TrajectoryDataCollector> No camera Error!")
        camera_depths = self.env.camera_depths

        self.current_images = {}
        for i, (cam_name, cam_depth_flag) in enumerate(zip(camera_names, camera_depths)):
            rgb = initial_obs.get(f"{cam_name}_image")
            if rgb is None:
                raise KeyError(f"[{cam_name}] obs['{cam_name}_image'] = None")
            if cam_depth_flag == True:
                depth_norm = initial_obs.get(f"{cam_name}_depth")
                if depth_norm is None:
                    raise KeyError(f"[{cam_name}] obs['{cam_name}_depth'] = None")

                if not np.isfinite(depth_norm).all():
                    n_nan = np.isnan(depth_norm).sum()
                    n_inf = np.isinf(depth_norm).sum()
                    raise ValueError(f"[{cam_name}] depth_norm 存在非有限值：NaN={int(n_nan)}, Inf={int(n_inf)}")

                eps = 1e-7
                dmin = float(np.nanmin(depth_norm))
                dmax = float(np.nanmax(depth_norm))
                if dmin < -eps or dmax > 1.0 + eps:
                    raise ValueError(
                        f"[{cam_name}] depth_norm 超出[0,1]：min={dmin:.6f}, max={dmax:.6f}, dtype={depth_norm.dtype}"
                    )

                if env is not None:
                    depth = get_real_depth_map(env.sim, depth_norm)
            else:
                depth = None

        self.obs_buffer.append(initial_obs)
        self.step_count = 0

    def record_step(self, obs, action, reward, done, env=None):

        self.step_count += 1
        camera_names = self.env.camera_names
        camera_depths = self.env.camera_depths

        for i, (cam_name, cam_depth_flag) in enumerate(zip(camera_names, camera_depths)):
            rgb = obs.get(f"{cam_name}_image")
            if rgb is None:
                raise KeyError(f"[{cam_name}] obs['{cam_name}_image'] = None")
            if cam_depth_flag == True:
                depth_norm = obs.get(f"{cam_name}_depth")
                if depth_norm is None:
                    raise KeyError(f"[{cam_name}] obs['{cam_name}_depth'] = None")

                if not np.isfinite(depth_norm).all():
                    n_nan = np.isnan(depth_norm).sum()
                    n_inf = np.isinf(depth_norm).sum()
                    raise ValueError(f"[{cam_name}] depth_norm 存在非有限值：NaN={int(n_nan)}, Inf={int(n_inf)}")

                eps = 1e-7
                dmin = float(np.nanmin(depth_norm))
                dmax = float(np.nanmax(depth_norm))
                if dmin < -eps or dmax > 1.0 + eps:
                    raise ValueError(
                        f"[{cam_name}] depth_norm 超出[0,1]：min={dmin:.6f}, max={dmax:.6f}, dtype={depth_norm.dtype}"
                    )

                if env is not None:
                    depth = get_real_depth_map(env.sim, depth_norm)
            else:
                depth = None

        self.action_buffer.append(np.array(action, dtype=np.float32))       # action
        self.reward_buffer.append(float(reward))                            # reward
        self.done_buffer.append(done)                                       # done
        self.obs_buffer.append(obs)                                         # obs

    def end_demonstration(self, num_count, env=None):

        if len(self.obs_buffer) == 0:
            raise ValueError(f"<TrajectoryDataCollector> end_demonstration Collector Error!")

        self.done_buffer[-1] = True
        env_kwargs = {}

        file_name = os.path.join(self.output_dir, f"collection_{num_count}.hdf5")
        total_step = len(self.action_buffer)
        env_name = getattr(self.env, "env_name", None)
        if env_name is None:
            env_name = self.env.__class__.__name__
        env_type = "robosuite"

        try:
            env_kwargs["target_obj"] = self.target_obj or None
            env_kwargs["robots"] = [robot.robot_model.name for robot in self.env.robots] \
                if hasattr(self.env, "robots") else None
        except Exception:
            pass


        with h5py.File(file_name, "w") as hf:
            data_grp = hf.create_group("data")
            data_grp.attrs["total"] = total_step
            env_args_attr = {
                "env_name": env_name,
                "env_type": env_type,
                "env_kwargs": env_kwargs
            }
            data_grp.attrs["env_args"] = json.dumps(env_args_attr)

            demo_grp = data_grp.create_group("demo")
            demo_grp.attrs["num_samples"] = total_step

            actions_arr = np.asarray(self.action_buffer, dtype=np.float32)
            rewards_arr = np.asarray(self.reward_buffer, dtype=np.float32)
            dones_arr = np.asarray(self.done_buffer, dtype=np.bool_)

            obs_grp = demo_grp.create_group("obs")

            camera_names = self.env.camera_names
            camera_depths = self.env.camera_depths
            if isinstance(camera_depths, bool):
                camera_depths = [camera_depths] * len(camera_names)

            depth_flags = dict(zip(camera_names, camera_depths))

            rgb_buffers = {cam: [] for cam in camera_names}
            depth_buffers = {cam: [] for cam in camera_names if depth_flags[cam]}

            joint_pos_list = []
            eef_pos_list = []
            eef_quat_list = []
            gripper_qpos_list = []

            for t in range(total_step):
                obs_dict = self.obs_buffer[t]
                action_dick = self.action_buffer[t]

                for cam in camera_names:
                    rgb = obs_dict.get(f"{cam}_image")
                    rgb_buffers[cam].append(rgb)
                    if depth_flags[cam]:
                        depth = obs_dict.get(f"{cam}_depth")
                        if env is not None:
                            depth = get_real_depth_map(env.sim, depth)
                        depth_buffers[cam].append(depth)

                if hasattr(self.env, "robots"):
                    robot = self.env.robots[0]

                    if "robot0_joint_pos" in obs_dict:
                        _joint_pos = obs_dict["robot0_joint_pos"]
                        joint_pos_list.append(np.array(_joint_pos, dtype=np.float32))
                    else:
                        raise ValueError(f"<TrajectoryDataCollector> end_demonstration joint get Error!")

                    # end point
                    if ("robot0_eef_pos" in obs_dict) and ("robot0_eef_quat" in obs_dict):
                        _eef_pos = obs_dict["robot0_eef_pos"]
                        _eef_quat = obs_dict["robot0_eef_quat"]
                        eef_pos_list.append(np.array(_eef_pos, dtype=np.float32))
                        eef_quat_list.append(np.array(_eef_quat, dtype=np.float32))
                    else:
                        raise ValueError(f"<TrajectoryDataCollector> end_demonstration end_point get Error!")

                    # gripper -1 open / 1 close
                    if action_dick[-1] == -1 or 1:
                        discrete_gripper = action_dick[-1]
                        gripper_qpos_list.append(discrete_gripper)
                    else:
                        raise ValueError(f"<TrajectoryDataCollector> end_demonstration gripper get Error!")

            for cam in camera_names:
                rgb_list = rgb_buffers.get(cam, [])
                if rgb_list:
                    frames = np.stack(rgb_list, axis=0)
                    obs_grp.create_dataset(f"{cam}_image", data=frames, compression="gzip")

                if depth_flags.get(cam, False):
                    depth_list = depth_buffers.get(cam, [])
                    if depth_list:
                        dframes = np.stack(depth_list, axis=0)
                        obs_grp.create_dataset(f"{cam}_depth", data=dframes, compression="gzip")

            obs_grp.create_dataset("joint_pos", data=np.stack(joint_pos_list, axis=0))
            obs_grp.create_dataset("eef_pos", data=np.stack(eef_pos_list, axis=0))
            obs_grp.create_dataset("eef_quat", data=np.stack(eef_quat_list, axis=0))
            obs_grp.create_dataset("gripper_qpos", data=gripper_qpos_list)

