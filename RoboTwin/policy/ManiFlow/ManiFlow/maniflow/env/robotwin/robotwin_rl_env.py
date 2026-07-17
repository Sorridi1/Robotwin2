from collections import deque
from pathlib import Path
from typing import Dict, Optional
import importlib
import os
import sys

import numpy as np
import yaml


ROBOTWIN_ROOT = Path(__file__).resolve().parents[6]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.append(str(ROBOTWIN_ROOT))
if str(ROBOTWIN_ROOT / "description" / "utils") not in sys.path:
    sys.path.append(str(ROBOTWIN_ROOT / "description" / "utils"))

_IMPORT_CWD = os.getcwd()
os.chdir(ROBOTWIN_ROOT)
try:
    from envs import CONFIGS_PATH  # noqa: E402
    from envs.utils.create_actor import UnStableError  # noqa: E402
finally:
    os.chdir(_IMPORT_CWD)

try:
    from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
except Exception:
    generate_episode_descriptions = None


def _read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _task_env(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise RuntimeError(f"No RoboTwin task class named {task_name}") from exc
    return env_class()


class RoboTwinRLEnv:
    """RoboTwin task wrapper with ManiFlow-compatible stacked observations."""

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean_ur5",
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 0,
        start_seed: Optional[int] = None,
        reward_mode: str = "sparse_success",
        reward_agg_method: str = "sum",
        expert_check: bool = True,
        instruction_type: str = "seen",
        ckpt_setting: str = "rl",
        render_freq: int = 0,
        clear_cache_freq: int = 5,
        max_seed_tries: int = 100,
        action_type: str = "qpos",
        reward_progress_scale: float = 1.0,
        reward_success_bonus: float = 1.0,
        reward_healthy: float = 0.0,
        reward_ctrl_cost_weight: float = 1.0e-3,
        reward_ctrl_cost_mode: str = "delta",
        reward_ctrl_cost_normalize: bool = True,
        reward_contact_cost_weight: float = 0.0,
        reward_time_cost: float = 0.0,
        reward_use_task_progress_fallback: bool = True,
        **kwargs,
    ):
        if reward_mode not in ("sparse_success", "stage_delta", "ant_style"):
            raise ValueError(f"Unsupported reward_mode={reward_mode}")
        if reward_agg_method != "sum":
            raise ValueError("RoboTwinRLEnv currently supports reward_agg_method='sum' only")
        if reward_ctrl_cost_mode not in ("delta", "absolute"):
            raise ValueError("reward_ctrl_cost_mode must be 'delta' or 'absolute'")

        self.task_name = task_name
        self.task_config = task_config
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.seed = seed
        self.next_seed = int(start_seed if start_seed is not None else 100000 * (1 + seed))
        self.reward_mode = reward_mode
        self.reward_agg_method = reward_agg_method
        self.expert_check = expert_check
        self.instruction_type = instruction_type
        self.ckpt_setting = ckpt_setting
        self.render_freq = render_freq
        self.clear_cache_freq = clear_cache_freq
        self.max_seed_tries = max_seed_tries
        self.action_type = action_type
        self.reward_progress_scale = float(reward_progress_scale)
        self.reward_success_bonus = float(reward_success_bonus)
        self.reward_healthy = float(reward_healthy)
        self.reward_ctrl_cost_weight = float(reward_ctrl_cost_weight)
        self.reward_ctrl_cost_mode = reward_ctrl_cost_mode
        self.reward_ctrl_cost_normalize = bool(reward_ctrl_cost_normalize)
        self.reward_contact_cost_weight = float(reward_contact_cost_weight)
        self.reward_time_cost = float(reward_time_cost)
        self.reward_use_task_progress_fallback = bool(reward_use_task_progress_fallback)

        self._prev_cwd = os.getcwd()
        os.chdir(ROBOTWIN_ROOT)
        self.task_env = _task_env(task_name)
        self.obs = deque(maxlen=n_obs_steps + 1)
        self.episode_id = 0
        self.test_num = 0
        self._success_seen = False
        self._last_stage_reward = 0.0
        self._last_progress_value = 0.0
        self._initial_place_empty_cup_xy_dist = None
        self._initial_place_empty_cup_z = None
        self._last_observation = None
        self._has_active_env = False
        self.base_args = self._build_task_args()

    def _build_task_args(self) -> Dict:
        task_cfg_path = ROBOTWIN_ROOT / "task_config" / f"{self.task_config}.yml"
        args = _read_yaml(task_cfg_path)
        args["task_name"] = self.task_name
        args["task_config"] = self.task_config
        args["ckpt_setting"] = self.ckpt_setting
        args["policy_name"] = "ManiFlow"
        args["eval_mode"] = True
        args["render_freq"] = self.render_freq
        args["eval_video_log"] = False
        args["eval_video_save_dir"] = None

        embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
        embodiment_types = _read_yaml(embodiment_config_path)

        def get_embodiment_file(embodiment_type):
            robot_file = embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise RuntimeError(f"No embodiment file configured for {embodiment_type}")
            return robot_file

        def get_embodiment_config(robot_file):
            return _read_yaml(Path(robot_file) / "config.yml")

        camera_config = _read_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")
        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        embodiment_type = args.get("embodiment")
        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise RuntimeError("embodiment items should be 1 or 3")
        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
        return args

    def _encode_obs(self, observation: Dict) -> Dict[str, np.ndarray]:
        obs = {
            "agent_pos": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "point_cloud": np.asarray(observation["pointcloud"], dtype=np.float32),
        }
        head = observation.get("observation", {}).get("head_camera", {})
        if "rgb" in head:
            obs["head_cam"] = (np.moveaxis(head["rgb"], -1, 0) / 255.0).astype(np.float32)
        return obs

    def _stack_last_n_obs(self):
        assert len(self.obs) > 0
        result = {}
        for key in self.obs[0].keys():
            items = [o[key] for o in self.obs]
            out = np.zeros((self.n_obs_steps,) + items[-1].shape, dtype=items[-1].dtype)
            start_idx = -min(self.n_obs_steps, len(items))
            out[start_idx:] = np.asarray(items[start_idx:])
            if self.n_obs_steps > len(items):
                out[:start_idx] = out[start_idx]
            result[key] = out
        return result

    def _sample_instruction(self, episode_info):
        if generate_episode_descriptions is None:
            return None
        try:
            episode_info_list = [episode_info["info"]]
            descriptions = generate_episode_descriptions(self.task_name, episode_info_list, 1)
            candidates = descriptions[0][self.instruction_type]
            return np.random.choice(candidates)
        except Exception:
            return None

    def _find_valid_seed(self, start_seed: int):
        seed = start_seed
        last_episode_info = {"info": {}}
        for _ in range(self.max_seed_tries):
            if not self.expert_check:
                return seed, last_episode_info
            args = dict(self.base_args)
            try:
                self.task_env.setup_demo(
                    now_ep_num=self.episode_id,
                    seed=seed,
                    is_test=True,
                    **args,
                )
                episode_info = self.task_env.play_once()
                valid = bool(self.task_env.plan_success and self.task_env.check_success())
                self.task_env.close_env(clear_cache=True)
                if valid:
                    return seed, episode_info
            except UnStableError:
                self.task_env.close_env(clear_cache=True)
            except Exception:
                self.task_env.close_env(clear_cache=True)
            seed += 1
        raise RuntimeError(
            f"Failed to find a valid RoboTwin seed for {self.task_name} after {self.max_seed_tries} tries"
        )

    def _current_stage_reward(self) -> float:
        fn = getattr(self.task_env, "stage_reward", None)
        if fn is None or not callable(fn):
            return 0.0
        try:
            return float(fn())
        except Exception:
            return 0.0

    def _current_progress_value(self) -> float:
        fn = getattr(self.task_env, "stage_reward", None)
        if fn is not None and callable(fn):
            try:
                return float(fn())
            except Exception:
                pass
        if not self.reward_use_task_progress_fallback:
            return 0.0
        if self.task_name == "place_empty_cup":
            return self._place_empty_cup_progress_value()
        return 0.0

    def _place_empty_cup_progress_value(self) -> float:
        try:
            cup_pose = np.asarray(self.task_env.cup.get_functional_point(0, "pose").p, dtype=np.float32)
            coaster_pose = np.asarray(
                self.task_env.coaster.get_functional_point(0, "pose").p,
                dtype=np.float32,
            )
        except Exception:
            return 0.0

        xy_dist = float(np.linalg.norm(cup_pose[:2] - coaster_pose[:2], ord=2))
        z_dist = float(abs(cup_pose[2] - coaster_pose[2]))
        if self._initial_place_empty_cup_xy_dist is None:
            self._initial_place_empty_cup_xy_dist = max(xy_dist, 0.05)
        if self._initial_place_empty_cup_z is None:
            self._initial_place_empty_cup_z = float(cup_pose[2])

        xy_score = 1.0 - min(xy_dist / self._initial_place_empty_cup_xy_dist, 2.0)
        z_score = 1.0 - min(z_dist / 0.08, 2.0)
        lift_score = np.clip((float(cup_pose[2]) - self._initial_place_empty_cup_z) / 0.08, 0.0, 1.0)
        return float(0.7 * xy_score + 0.2 * z_score + 0.1 * lift_score)

    def _control_cost(self, action: np.ndarray, previous_observation: Optional[Dict]) -> float:
        if self.reward_ctrl_cost_weight <= 0.0:
            return 0.0
        action = np.asarray(action, dtype=np.float32)
        if self.reward_ctrl_cost_mode == "delta" and previous_observation is not None:
            try:
                reference = np.asarray(previous_observation["joint_action"]["vector"], dtype=np.float32)
                action_term = action - reference
            except Exception:
                action_term = action
        else:
            action_term = action

        cost = float(np.sum(np.square(action_term)))
        if self.reward_ctrl_cost_normalize and action_term.size > 0:
            cost /= float(action_term.size)
        return self.reward_ctrl_cost_weight * cost

    def _contact_cost(self) -> float:
        if self.reward_contact_cost_weight <= 0.0:
            return 0.0
        get_contacts = getattr(getattr(self.task_env, "scene", None), "get_contacts", None)
        if not callable(get_contacts):
            return 0.0
        try:
            return self.reward_contact_cost_weight * float(len(get_contacts()))
        except Exception:
            return 0.0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.next_seed = int(seed)
        if self._has_active_env:
            self.task_env.close_env(clear_cache=True)
            self._has_active_env = False

        valid_seed, episode_info = self._find_valid_seed(self.next_seed)
        self.next_seed = valid_seed + 1

        args = dict(self.base_args)
        self.task_env.setup_demo(
            now_ep_num=self.episode_id,
            seed=valid_seed,
            is_test=True,
            **args,
        )
        self._has_active_env = True
        instruction = self._sample_instruction(episode_info)
        if instruction is not None:
            self.task_env.set_instruction(instruction=instruction)

        self.obs.clear()
        self._success_seen = False
        self._initial_place_empty_cup_xy_dist = None
        self._initial_place_empty_cup_z = None
        self._last_stage_reward = self._current_stage_reward()
        self._last_progress_value = self._current_progress_value()
        observation = self.task_env.get_obs()
        self._last_observation = observation
        self.obs.append(self._encode_obs(observation))
        self.episode_id += 1
        self.test_num += 1
        return self._stack_last_n_obs()

    def _raw_reward_after_action(
        self,
        action: np.ndarray,
        was_success: bool,
        previous_observation: Optional[Dict],
    ):
        now_success = bool(getattr(self.task_env, "eval_success", False))
        if self.reward_mode == "sparse_success":
            reward = 1.0 if (not self._success_seen and not was_success and now_success) else 0.0
            if now_success:
                self._success_seen = True
            return reward, {
                "forward": 0.0,
                "success": reward,
                "healthy": 0.0,
                "ctrl": 0.0,
                "contact": 0.0,
                "time": 0.0,
            }

        if self.reward_mode == "stage_delta":
            current_stage = self._current_stage_reward()
            reward = current_stage - self._last_stage_reward
            self._last_stage_reward = current_stage
            return float(reward), {
                "forward": float(reward),
                "success": 0.0,
                "healthy": 0.0,
                "ctrl": 0.0,
                "contact": 0.0,
                "time": 0.0,
            }

        current_progress = self._current_progress_value()
        progress_reward = self.reward_progress_scale * (current_progress - self._last_progress_value)
        self._last_progress_value = current_progress

        success_reward = (
            self.reward_success_bonus if (not self._success_seen and not was_success and now_success) else 0.0
        )
        healthy_reward = self.reward_healthy
        ctrl_cost = self._control_cost(action, previous_observation)
        contact_cost = self._contact_cost()
        time_cost = self.reward_time_cost

        reward = progress_reward + success_reward + healthy_reward - ctrl_cost - contact_cost - time_cost
        if now_success:
            self._success_seen = True
        return float(reward), {
            "forward": float(progress_reward),
            "success": float(success_reward),
            "healthy": float(healthy_reward),
            "ctrl": float(-ctrl_cost),
            "contact": float(-contact_cost),
            "time": float(-time_cost),
        }

    def step(self, action_chunk):
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None]

        raw_rewards = []
        reward_terms_list = []
        substep_success = []
        executed = 0
        for action in action_chunk[: self.n_action_steps]:
            if bool(getattr(self.task_env, "eval_success", False)):
                break
            if self.task_env.step_lim is not None and self.task_env.take_action_cnt >= self.task_env.step_lim:
                break

            was_success = bool(getattr(self.task_env, "eval_success", False))
            previous_observation = self._last_observation
            self.task_env.take_action(action, action_type=self.action_type)
            raw_reward, reward_terms = self._raw_reward_after_action(action, was_success, previous_observation)
            raw_rewards.append(raw_reward)
            substep_success.append(bool(getattr(self.task_env, "eval_success", False)))
            reward_terms_list.append(reward_terms)
            observation = self.task_env.get_obs()
            self._last_observation = observation
            self.obs.append(self._encode_obs(observation))
            executed += 1

            if bool(getattr(self.task_env, "eval_success", False)):
                break
            if self.task_env.step_lim is not None and self.task_env.take_action_cnt >= self.task_env.step_lim:
                break

        terminated = bool(getattr(self.task_env, "eval_success", False))
        truncated = bool(
            self.task_env.step_lim is not None and self.task_env.take_action_cnt >= self.task_env.step_lim
        )
        reward = float(np.sum(raw_rewards)) if len(raw_rewards) > 0 else 0.0
        reward_terms_sum = {
            key: float(np.sum([terms.get(key, 0.0) for terms in reward_terms_list]))
            for key in ("forward", "success", "healthy", "ctrl", "contact", "time")
        }
        info = {
            "raw_rewards": raw_rewards,
            "reward_terms": reward_terms_list,
            "reward_forward": reward_terms_sum["forward"],
            "reward_success": reward_terms_sum["success"],
            "reward_healthy": reward_terms_sum["healthy"],
            "reward_ctrl": reward_terms_sum["ctrl"],
            "reward_contact": reward_terms_sum["contact"],
            "reward_time": reward_terms_sum["time"],
            "substep_success": substep_success,
            "success": terminated or self._success_seen,
            "executed_action_steps": executed,
            "take_action_cnt": int(getattr(self.task_env, "take_action_cnt", 0)),
            "step_lim": int(self.task_env.step_lim) if self.task_env.step_lim is not None else None,
        }
        return self._stack_last_n_obs(), reward, terminated, truncated, info

    def close(self, clear_cache: bool = False):
        if self.task_env is not None and self._has_active_env:
            self.task_env.close_env(clear_cache=clear_cache)
            self._has_active_env = False
