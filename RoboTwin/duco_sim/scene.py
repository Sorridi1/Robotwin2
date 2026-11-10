"""Single-arm SAPIEN task runtime for RoboTwin and the real-Duco checkpoint."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import sapien

from .common import add_geometry_path, add_policy_path
from .robot import DucoKinematics, corrected_urdf, estimate_mount, mount_matrix


def quat(rotation):
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def optical_to_native_camera():
    """SAPIEN camera +X forward/+Y left/+Z up in OpenCV optical axes."""
    result = np.eye(4)
    result[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    return result


class DucoScene:
    def setup_demo(self, is_test=True, **kwargs):
        self.config = copy.deepcopy(kwargs)
        add_geometry_path(kwargs['geometry_root'])
        add_policy_path()
        from duco_wipe_data import load_recording, read_deploy_calibration
        from duco_wipe_scene import estimate_scene, star_local_geometry
        from duco_wipe_layout import assert_layout_valid
        from maniflow.common.rise_duco_data import RGBDPointCloudPreprocessor
        self.recording = load_recording(kwargs['dataset_root'], kwargs.get('demo', 50))
        self.calibration = read_deploy_calibration(kwargs['deploy_root'])
        if kwargs.get('scene_profile'):
            self.profile = json.loads(Path(kwargs['scene_profile']).read_text())
            if self.profile['source_demo'] != str(self.recording.path):
                raise ValueError('Scene profile belongs to a different source demo')
        else:
            self.profile = estimate_scene(self.recording, self.calibration, kwargs['object_config'])
        assert_layout_valid(self.profile)
        self.fk = DucoKinematics(kwargs['urdf_path'])
        if kwargs.get('tool_calibration'):
            self.tool_calibration = json.loads(Path(kwargs['tool_calibration']).read_text())
        elif kwargs.get('tool_mount', 'estimated') == 'estimated':
            reference = load_recording(kwargs['dataset_root'], 50)
            reference_profile = estimate_scene(reference, self.calibration, kwargs['object_config'])
            self.tool_calibration = estimate_mount(self.fk, reference, reference_profile)
        else:
            self.tool_calibration = dict(xyz_m=[0., 0., 0.], rpy_deg=[0., 0., 0.], verified=False, source='original URDF assembly')
        self.tool_transform = mount_matrix(self.tool_calibration)
        self.base_height = float(self.profile['world_base_height_m'])
        self.world_offset = np.array([0., 0., self.base_height])
        self.control_freq = int(kwargs.get('control_freq', 10))
        self.sim_freq = int(kwargs.get('sim_freq', 240))
        if self.sim_freq % self.control_freq:
            raise ValueError('sim_freq must be divisible by control_freq')
        self.substeps = self.sim_freq // self.control_freq
        self.step_lim = int(kwargs.get('max_steps', 300))
        self.take_action_cnt = 0
        self.instruction = self.recording.instruction
        self.viewer = None
        self._tmp = tempfile.TemporaryDirectory(prefix='duco_robotwin_')
        self.generated_urdf = corrected_urdf(self.fk, self.tool_calibration, self._tmp.name)
        self.preprocessor = RGBDPointCloudPreprocessor(device=kwargs.get('preprocess_device', 'auto'))
        # Avoid exhausting the beta renderer's small default material table.
        sapien.render.set_global_config(max_num_materials=2048, max_num_textures=1024)
        self._materials = {}
        sapien.render.set_camera_shader_dir(kwargs.get('rendering', {}).get('shader', 'default'))
        # The legacy Engine wrapper ignores renderer/device selection in b1.
        # Use explicit systems so GUI and sensors share the selected CUDA GPU.
        self.render_device = sapien.Device(kwargs.get('render_device', 'cuda:0'))
        self.renderer = sapien.render.SapienRenderer(self.render_device)
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(self.render_device)])
        self.scene.set_timestep(1 / self.sim_freq)
        self.scene.set_ambient_light([.65, .65, .65])
        self.scene.add_directional_light([.3, .4, -1], [.8, .8, .8], shadow=True)
        self.physical_material = sapien.physx.PhysxMaterial(1.2, 1.0, 0.)
        self.load_actors()
        self.load_robot()
        self.load_camera()
        self.ink_pieces, self.ink_samples = star_local_geometry(self.profile)
        self.ink_entities = []
        for index, piece in enumerate(self.ink_pieces):
            local = sapien.Pose(piece['position'], quat(Rotation.from_euler('z', piece['yaw_rad']).as_matrix()))
            self.ink_entities.append(self._box(f'duco_ink_{index}', piece['half_size'], self.board.pose * local,
                                              [.10, .08, .16, 1], collision=False))
        self.erased = np.zeros(self.ink_samples.shape[:2], dtype=bool)
        self.max_lift = 0.
        self.max_board_force = 0.
        self.max_tracking_error = np.zeros(6)
        self.bilateral_contact_steps = 0
        self.grasp_run_substeps = 0
        self.max_grasp_run_substeps = 0
        self.current_finger_forces = np.zeros(2)
        self.action_clip_steps = 0
        self.last_action = np.r_[self.recording.joints[0], self.recording.gripper[0] / 1000.]
        self.gripper_command = float(self.last_action[-1])
        self.robot.set_qpos(self._ordered(self.fk.qpos(self.last_action)))
        self.robot.set_qvel(np.zeros(8))
        self._drive(self.fk.qpos(self.last_action))
        # Settle dynamic props using real physics, while holding the initial arm.
        for _ in range(self.substeps):
            self._physics_step()
        self.initial_eraser_pose = self.eraser.pose
        self.eval_success = False
        self.info = dict(info={'task': 'duco_wipe_whiteboard', 'arm': 'duco'}, scene=self.profile,
                         tool_calibration=self.tool_calibration)

    def _material(self, color):
        key = tuple(color)
        if key in self._materials:
            return self._materials[key]
        material = sapien.render.RenderMaterial()
        material.base_color = color
        material.roughness = .8
        self._materials[key] = material
        return material

    def _box(self, name, half, pose, color=(.6, .6, .6, 1), collision=True, dynamic=False, density=1000):
        builder = self.scene.create_actor_builder()
        if collision:
            builder.add_box_collision(half_size=half, material=self.physical_material, density=density)
        builder.add_box_visual(half_size=half, material=self._material(color))
        builder.initial_pose = pose
        return builder.build(name) if dynamic else builder.build_kinematic(name)

    def load_actors(self):
        from duco_wipe_layout import surface_frame, tray_geometry
        from duco_wipe_scene import board_surface_frame, board_surface_half_size, plane_z
        table = self.profile['table']
        low, high = np.asarray(table['xy_low']), np.asarray(table['xy_high'])
        center, rotation = surface_frame(table['plane'], (low + high) / 2)
        thickness = table['thickness_m']
        center += self.world_offset - rotation[:, 2] * thickness / 2
        self.table = self._box('duco_table', [*((high - low) / 2), thickness / 2], sapien.Pose(center, quat(rotation)))
        self._box('duco_floor', [3, 3, .02], sapien.Pose([0, 0, -.025]), [.25, .25, .25, 1])
        for i, corner in enumerate([low, [low[0], high[1]], [high[0], low[1]], high]):
            xy = np.asarray(corner) + np.sign((low + high) / 2 - corner) * .04
            height = self.base_height + plane_z(table['plane'], *xy) - thickness
            self._box(f'duco_table_leg_{i}', [.025, .025, height / 2], sapien.Pose([*xy, height / 2]))
        pad = max(.002, -plane_z(table['plane'], 0, 0))
        self._box('duco_base_mount_pad', [.11, .11, pad / 2], sapien.Pose([0, 0, self.base_height - pad / 2]))
        board = self.profile['board']
        origin, board_rotation = board_surface_frame(board)
        builder = self.scene.create_actor_builder()
        outer, white = np.asarray(board['size_xy_m']) / 2, board_surface_half_size(board)
        thickness = board['thickness_m']
        builder.add_box_collision(half_size=[*outer, thickness / 2], pose=sapien.Pose([0, 0, -thickness / 2]), material=self.physical_material)
        builder.add_box_visual(half_size=[*outer, thickness / 2], pose=sapien.Pose([0, 0, -thickness / 2]), material=self._material([.08, .08, .08, 1]))
        builder.add_box_visual(half_size=[*white, .0002], pose=sapien.Pose([0, 0, -.0002]), material=self._material([.94, .94, .92, 1]))
        builder.initial_pose = sapien.Pose(origin + self.world_offset, quat(board_rotation))
        self.board = builder.build_kinematic('duco_whiteboard')
        tray = self.profile['tray']
        origin, rotation = surface_frame(table['plane'], tray['xy_center'])
        builder = self.scene.create_actor_builder()
        floor = tray['floor_thickness_m']
        pose = sapien.Pose([0, 0, floor / 2], [2**-.5, 0, 2**-.5, 0])
        builder.add_cylinder_collision(radius=tray['radius_m'], half_length=floor / 2, pose=pose)
        builder.add_cylinder_visual(radius=tray['radius_m'], half_length=floor / 2, pose=pose, material=self._material([.82, .25, .3, 1]))
        for piece in tray_geometry(tray):
            pose = sapien.Pose(piece['position'], quat(Rotation.from_euler('z', piece['yaw_rad']).as_matrix()))
            builder.add_box_collision(half_size=piece['half_size'], pose=pose)
            builder.add_box_visual(half_size=piece['half_size'], pose=pose, material=self._material([.82, .25, .3, 1]))
        builder.initial_pose = sapien.Pose(origin + self.world_offset, quat(rotation))
        self.tray = builder.build_kinematic('duco_tray')
        eraser = self.profile['eraser']
        origin, rotation = surface_frame(table['plane'], eraser['xy_center'])
        rotation = rotation @ Rotation.from_euler('z', eraser['yaw_rad']).as_matrix()
        size = np.asarray(eraser['size_xyz_m'])
        center = origin + self.world_offset + rotation[:, 2] * (size[2] / 2 + .0005)
        self.eraser = self._box('duco_eraser', size / 2, sapien.Pose(center, quat(rotation)), [.09, .10, .12, 1],
                                dynamic=True, density=eraser['mass_kg'] / np.prod(size))
        self.eraser_body = self.eraser.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

    def load_robot(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        # These assets are STL, not multi-convex OBJ collision files.
        loader.load_multiple_collisions_from_file = False
        loader.set_material(1.2, 1.0, 0.)
        self.robot = loader.load(str(self.generated_urdf))
        if self.robot is None:
            raise RuntimeError('SAPIEN could not load Duco URDF')
        self.robot.set_root_pose(sapien.Pose(self.world_offset, [2**-.5, 0, 0, 2**-.5]))
        self.joints = self.robot.get_active_joints()
        self.active_names = [j.name for j in self.joints]
        if set(self.active_names) != set(self.fk.names):
            raise ValueError(f'URDF joint mapping mismatch: {self.active_names}')
        self.links = {link.name: link for link in self.robot.get_links()}
        for name, link in self.links.items():
            link.entity.name = 'duco_' + name
            # Suppress intra-robot overlaps in the imported CAD assembly only.
            # Robot/table/object collision masks remain fully enabled.
            for shape in link.collision_shapes:
                groups = list(shape.collision_groups)
                groups[2] |= 1
                shape.set_collision_groups(groups)
        settings = self.config['robot']
        for joint in self.joints:
            group = 'arm' if joint.name in self.fk.names[:6] else 'gripper'
            joint.set_drive_property(settings[f'{group}_stiffness'], settings[f'{group}_damping'], settings[f'{group}_force_limit'])

    def load_camera(self):
        options = self.config['camera']
        self.camera = self.scene.add_camera('duco_head_camera', options['width'], options['height'], 1., options['near'], options['far'])
        k = self.calibration['intrinsics']
        if [options['width'], options['height']] != [1280, 720]:
            raise ValueError('Training intrinsics require 1280x720; do not silently resize')
        self.camera.set_perspective_parameters(options['near'], options['far'], k[0, 0], k[1, 1], k[0, 2], k[1, 2], 0.)
        optical_to_sapien = optical_to_native_camera()
        world_camera = self.calibration['camera_to_base'].copy()
        world_camera[:3, 3] += self.world_offset
        self.camera.entity.pose = sapien.Pose(world_camera @ optical_to_sapien)

    def _ordered(self, source_qpos):
        values = dict(zip(self.fk.names, source_qpos))
        return np.array([values[n] for n in self.active_names], dtype=np.float32)

    def _drive(self, qpos):
        for name, target in zip(self.fk.names, qpos):
            self.joints[self.active_names.index(name)].set_drive_target(float(target))

    def _physics_step(self):
        self.robot.set_qf(self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        self.scene.step()

    def take_action(self, action, action_type='qpos'):
        if action_type != 'qpos':
            raise ValueError('This checkpoint outputs absolute joint positions, not TCP/delta actions')
        raw = np.asarray(action, dtype=float)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError('Non-finite or non-7D model action')
        bounded = np.r_[np.clip(raw[:6], self.fk.lower[:6], self.fk.upper[:6]), np.clip(raw[6], 0, 1)]
        max_delta = float(self.config['robot']['max_joint_step_rad'])
        bounded[:6] = self.last_action[:6] + np.clip(bounded[:6] - self.last_action[:6], -max_delta, max_delta)
        self.action_clip_steps += int(not np.allclose(bounded, raw))
        self.last_action = bounded.copy()
        self.gripper_command = float(bounded[-1])
        self._drive(self.fk.qpos(bounded))
        self.current_finger_forces[:] = 0
        for _ in range(self.substeps):
            self._physics_step()
            self._update_contacts()
        self.take_action_cnt += 1
        self.max_tracking_error = np.maximum(self.max_tracking_error, np.abs(self.get_state()[:6] - bounded[:6]))
        if not np.isfinite(self.robot.get_qpos()).all() or not np.isfinite(self.eraser.pose.p).all():
            raise RuntimeError('Non-finite simulation state')
        self.eval_success = self.check_success()
        return bounded.copy()

    def get_state(self):
        values = dict(zip(self.active_names, self.robot.get_qpos()))
        return np.array([values[n] for n in self.fk.names[:6]] + [self.gripper_command], dtype=np.float32)

    def _update_contacts(self):
        board_force, fingers = 0., np.zeros(2)
        for contact in self.scene.get_contacts():
            bodies = contact.bodies
            if self.eraser_body not in bodies:
                continue
            force = np.linalg.norm(sum((np.asarray(point.impulse) for point in contact.points), np.zeros(3))) * self.sim_freq
            other = bodies[1] if bodies[0] == self.eraser_body else bodies[0]
            if other.entity == self.board:
                board_force += force
            for index, name in enumerate(['left_finger', 'right_finger']):
                if other == self.links[name]:
                    fingers[index] += force
        self.current_finger_forces = fingers
        self.max_board_force = max(self.max_board_force, board_force)
        self.max_lift = max(self.max_lift, float(self.eraser.pose.p[2] - self.initial_eraser_pose.p[2]))
        if np.all(fingers > .1):
            self.bilateral_contact_steps += 1
        lift = float(self.eraser.pose.p[2] - self.initial_eraser_pose.p[2])
        if np.all(fingers > .1) and lift >= .03 and np.linalg.norm(self.eraser_body.linear_velocity) < .5:
            self.grasp_run_substeps += 1
            self.max_grasp_run_substeps = max(self.max_grasp_run_substeps, self.grasp_run_substeps)
        else:
            self.grasp_run_substeps = 0
        if board_force < .1:
            return
        matrix = self.board.pose.to_transformation_matrix()
        samples = self.ink_samples @ matrix[:3, :3].T + matrix[:3, 3]
        eraser = self.eraser.pose.to_transformation_matrix()
        local = (samples - eraser[:3, 3]) @ eraser[:3, :3]
        half = np.asarray(self.profile['eraser']['size_xyz_m']) / 2
        inside = (np.abs(local[..., :2]) <= half[:2] + .0008).all(axis=-1)
        inside &= np.abs(local[..., 2] + half[2]) <= .0015
        before = self.erased.all(axis=1)
        self.erased |= inside
        for i in np.flatnonzero(self.erased.all(axis=1) & ~before):
            component = self.ink_entities[i].find_component_by_type(sapien.render.RenderBodyComponent)
            component.visibility = 0.

    def task_metrics(self):
        from duco_wipe_layout import surface_frame
        tray = self.profile['tray']
        origin, rotation = surface_frame(self.profile['table']['plane'], tray['xy_center'])
        half = np.asarray(self.profile['eraser']['size_xyz_m']) / 2
        corners = np.array([[x, y, z] for x in [-half[0], half[0]] for y in [-half[1], half[1]] for z in [-half[2], half[2]]])
        matrix = self.eraser.pose.to_transformation_matrix()
        world = corners @ matrix[:3, :3].T + matrix[:3, 3]
        local = (world - self.world_offset - origin) @ rotation
        contained = bool(np.all(np.linalg.norm(local[:, :2], axis=1) <= tray['radius_m'] - tray['rim_width_m']))
        resting = abs(float(local[:, 2].min()) - tray['floor_thickness_m']) < .006
        released = bool(self.gripper_command > .8 and np.all(self.current_finger_forces < .2))
        # Contact at one time and an unrelated later fling are NOT a grasp.
        grasped = bool(self.max_grasp_run_substeps >= int(.2 * self.sim_freq))
        coverage = float(self.erased.mean())
        return dict(task_success=bool(grasped and coverage >= .95 and contained and resting and released),
                    grasp_verified=grasped, wipe_contact_coverage=coverage, max_eraser_lift_m=self.max_lift,
                    max_board_contact_force_N=self.max_board_force, bilateral_contact_substeps=self.bilateral_contact_steps,
                    max_verified_grasp_run_substeps=self.max_grasp_run_substeps,
                    eraser_in_tray=bool(contained and resting), released_from_fingers=released,
                    action_clip_steps=self.action_clip_steps, max_joint_tracking_error_rad=self.max_tracking_error.tolist())

    def check_success(self):
        return self.task_metrics()['task_success']

    def get_obs(self):
        self.scene.update_render()
        self.camera.take_picture()
        rgb = np.rint(np.clip(self.camera.get_picture('Color')[..., :3], 0, 1) * 255).astype(np.uint8)
        # SAPIEN Position texture is OpenGL camera coordinates: depth = -Z.
        depth = np.maximum(-self.camera.get_picture('Position')[..., 2], 0) * 1000.
        depth[~np.isfinite(depth)] = 0
        cloud = self.preprocessor(rgb, depth.astype(np.float32))
        state = self.get_state()
        return dict(observation={'head_camera': dict(rgb=rgb, depth=depth)},
                    pointcloud=cloud, joint_action={'vector': state}, agent_pos=state)

    def render_human(self):
        if self.viewer is None:
            import os
            if not os.environ.get('DISPLAY'):
                raise RuntimeError('Native Viewer requires a desktop DISPLAY; use --headless for SSH inference')
            print(f'[duco GUI] DISPLAY={os.environ["DISPLAY"]}; device={self.render_device.name}; can_present={self.render_device.can_present()}', flush=True)
            if not self.render_device.can_present():
                raise RuntimeError('Selected GPU cannot present on this DISPLAY. Use the desktop GPU (tested GPU 2 on :2); no image fallback is used.')
            from sapien.utils.viewer import Viewer
            self.viewer = Viewer(self.renderer, resolutions=(1280, 900))
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(.8, 1.15, self.base_height + .7)
            self.viewer.set_camera_rpy(0, -.55, -2.15)
        self.scene.update_render()
        self.viewer.render()
        return self.viewer

    def close(self):
        if getattr(self, 'viewer', None) is not None:
            self.viewer.close()
            self.viewer = None
        if hasattr(self, 'scene'):
            self.scene.clear()
        if hasattr(self, '_tmp'):
            self._tmp.cleanup()

    def close_env(self, clear_cache=False):
        self.close()

    def get_scene_metadata(self):
        return copy.deepcopy(self.info)

    def set_instruction(self, instruction=None):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction
