import math
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from robosuite.environments.base import register_env
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

@register_env
class RealLift(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        table_offset=(0, 0, 0.8),
        grid_edge_length = None,
        external_objects = None,
        remove_default_object = False,
        target_object_name=None
    ):
        if external_objects is None and remove_default_object is True:
            raise ValueError("Reward Error!")

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

        self.grid_edge_length = grid_edge_length
        self.external_objects = list(external_objects) if external_objects else []
        self.remove_default_object = remove_default_object
        self.target_object_name = target_object_name

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        Lx, Ly, _ = table_full_size

        if grid_edge_length is not None and grid_edge_length > 0:
            ncx = max(1, math.floor(Lx / grid_edge_length))
            ncy = max(1, math.floor(Ly / grid_edge_length))
            cell = min(Lx / ncx, Ly / ncy)
            rows = max(1, math.floor(Ly / cell))
            cols = max(1, math.floor(Lx / cell))
        else:
            rows = 1
            cols = 1
            cell = min(Lx / cols, Ly / rows)

        self._grid_cell = cell
        self._grid_rows = rows
        self._grid_cols = cols

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _inject_checker_edge_texture(self, mujoco_arena, rows: int, cols: int):

        tex_name = "grid_checker_edge_tex"
        mat_name = "grid_checker_edge_mat"

        texture = Element("texture", {
            "name": tex_name,
            "type": "2d",
            "builtin": "checker",
            "rgb1": "1 1 1",
            "rgb2": "1 1 1",
            "mark": "edge",
            "markrgb": "0 0 0",
            "width": "512",
            "height": "512"
        })
        mat = Element("material", {
            "name": mat_name,
            "texture": tex_name,
            "texuniform": "true",
            "texrepeat": f"{cols} {rows}"
        })
        mujoco_arena.asset.append(texture)
        mujoco_arena.asset.append(mat)

        table_geom = mujoco_arena.worldbody.find(".//geom[@name='table_visual']")
        if table_geom is not None:
            table_geom.set("material", mat_name)

    def _read_model_name(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return root.attrib.get("model", None)

    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # inject texture
        self._inject_checker_edge_texture(
            mujoco_arena=mujoco_arena,
            rows=self._grid_rows,
            cols=self._grid_cols,
        )

        self.objects = []

        if self.remove_default_object is not True:
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "1 1",
                "specular": "0.4",
                "shininess": "0.1",
            }
            redwood = CustomMaterial(
                texture="WoodRed",
                tex_name="redwood",
                mat_name="redwood_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.cube = BoxObject(
                name="cube",
                size_min=[0.020, 0.020, 0.020],
                size_max=[0.022, 0.022, 0.022],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
            self.objects.append(self.cube)

        target_object_flag = False
        if self.external_objects is not None:
            for idx, xml_path in enumerate(self.external_objects):
                xml_file = xml_path_completion(xml_path)
                model_name = self._read_model_name(xml_file)

                obj = MujocoXMLObject(
                    fname=xml_file,
                    name=model_name,
                    # joints=[dict(type="free", damping="0.0005")],
                    joints="default",
                    obj_type="all",                     # collision / visual / all
                    duplicate_collision_geoms=True
                )
                self.objects.append(obj)

                if model_name == self.target_object_name:
                    self.target_object_name = obj       # target_object_name 只能使用 xml
                    target_object_flag = True

        if target_object_flag == False:
            self.target_object_name = self.objects[0]

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objects,
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=np.pi / 4,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.objects
        )

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 2.25

        elif self.reward_shaping:
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.target_object_name.root_body,
                target_type="body",
                return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.target_object_name):
                reward += 0.25

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _setup_references(self):
        super()._setup_references()

        # ** get self.cube_body_id(self.target_object_id)
        target_name = self.target_object_name.root_body
        target_body_id = self.sim.model.body_name2id(target_name)
        self.cube_body_id = target_body_id

    def _setup_observables(self):
        observables = super()._setup_observables()

        target_name = self.target_object_name.root_body
        target_body_id = self.sim.model.body_name2id(target_name)

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                taget_pos = np.array(self.sim.data.body_xpos[target_body_id])
                return taget_pos

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                taget_rot = convert_quat(np.array(self.sim.data.body_xquat[target_body_id]), to="xyzw")
                return taget_rot

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.target_object_name)

    def _check_success(self):
        target_name = self.target_object_name.root_body
        target_body_id = self.sim.model.body_name2id(target_name)
        target_height = self.sim.data.body_xpos[target_body_id][2]

        table_height = self.model.mujoco_arena.table_offset[2]

        return target_height > table_height + 0.04

