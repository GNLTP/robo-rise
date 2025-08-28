import faulthandler
faulthandler.enable()

import os
import argparse
import time
from copy import deepcopy
import numpy as np
from datetime import datetime

from utils.RealLift import RealLift
from utils._projector import Projector
from utils.Trajectory import TrajectoryDataCollector
import robosuite as suite
from robosuite.devices import Keyboard
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", type=str, default="RealLift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--target_obj", type=str, default="banana")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument('--cam_id', type=str, default='agentview')
    parser.add_argument("--cam_names", nargs="*", type=str, default=["agentview"])
    parser.add_argument("--cam_height", default=480, type=int)
    parser.add_argument("--cam_width", default=640, type=int)
    parser.add_argument("--image_freq_delta", default=1, type=int)      # compare robot freq
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)   # pos freq
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)   # rot freq
    parser.add_argument("--sys_max_fr", default=100, type=int)          # system max freq
    parser.add_argument("--get_max_fr", default=1, type=int)            # get max freq

    args = parser.parse_args()

    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0])

    detla_fr = args.sys_max_fr / args.get_max_fr

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

    calib_dir = os.path.join(this_dir, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    for i, camera_name in enumerate(list(args.cam_names)):
        h, w = env.camera_heights[i], env.camera_widths[i]
        K = get_camera_intrinsic_matrix(env.sim, camera_name, h, w)
        T = get_camera_extrinsic_matrix(env.sim, camera_name)
        cam_dir = os.path.join(calib_dir, camera_name)
        os.makedirs(cam_dir, exist_ok=True)
        intrinsics_path = os.path.join(cam_dir, "Intrinsics.npy")
        extrinsics_path = os.path.join(cam_dir, "Extrinsics.npy")
        np.save(intrinsics_path, K)
        np.save(extrinsics_path, T)

    projector = Projector(f"{calib_dir}/{args.cam_id}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    collector = TrajectoryDataCollector(env,
                                        output_dir=f"data/{timestamp}/{args.target_obj}",
                                        target_obj=args.target_obj)

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    if args.device == "keyboard":
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    else:
        raise Exception("Error device!")

    num_count = 0

    while True:
        obs = env.reset()
        env.render()

        collector.start_demonstration(obs, env=env)
        device.start_control()

        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([-1], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        start = time.time()
        epoch_num = 0
        active_robot = env.robots[device.active_robot]

        while True:
            input_ac_dict = device.input2action()

            if input_ac_dict is None:   # if None break
                break

            action_dict = deepcopy(input_ac_dict)

            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                else:
                    raise ValueError("input action Error! ")

            env_action = []
            for i, robot in enumerate(env.robots):
                _action = robot.create_action_vector(all_prev_gripper_actions[i])
                env_action.append(_action)
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)

            # update gripper status
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            obs2, reward, done, info = env.step(env_action)

            env.render()    # render

            if epoch_num % detla_fr == 0:
                collector.record_step(obs2, env_action, reward, done, env=env)

            # limit freq
            if args.sys_max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.sys_max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)

            epoch_num += 1

        collector.end_demonstration(num_count=num_count, env=env)
        num_count += 1
        print(f"target data collection_{num_count} success")
