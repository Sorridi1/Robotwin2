import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from maniflow.common.robotwin_env_pool import SerialEnvPool, SubprocessEnvPool
from maniflow.common.robotwin_seed_cache import ExpertSeedCache, seed_cache_fingerprint
from maniflow.env.robotwin.robotwin_rl_env import RoboTwinRLEnv, ROBOTWIN_ROOT


class FakeEnv:
    """Importable spawn fixture: exercises real processes without SAPIEN/CUDA."""
    def __init__(self, start_seed=10, barrier=None, fail_init=False, seed=0, sleep_seconds=0):
        if fail_init:
            raise ValueError("intentional initialization failure")
        self.next_seed = start_seed
        self.barrier = barrier
        self.last_reset_info = {"perf/expert_cache_hits": 1}
        self.t = 0
        self.sleep_seconds = sleep_seconds

    def _obs(self):
        return {
            "point_cloud": np.full((2, 3, 6), self.episode_seed + self.t, dtype=np.float32),
            "agent_pos": np.full((2, 2), self.t, dtype=np.float32),
        }

    def reset(self):
        self.episode_seed = self.next_seed
        self.next_seed += 1
        self.t = 0
        return self._obs()

    def step(self, action):
        value = float(np.asarray(action).sum())
        if value < -900:
            raise ValueError("intentional worker step failure")
        if self.barrier is not None:
            self.barrier.wait(timeout=15)
        time.sleep(self.sleep_seconds)
        self.t += 1
        return self._obs(), value, self.t == 2, False, {
            "success": self.t == 2, "worker_pid": os.getpid(), "executed_action_steps": 1,
        }

    def close(self, clear_cache=True):
        pass


def _configs(**kwargs):
    return [{"_target_": f"{__name__}.FakeEnv", "start_seed": 10 + i, **kwargs} for i in range(2)]


@pytest.mark.parametrize("use_shared_memory", [False, True])
def test_parallel_pool_preserves_transitions_autoreset_and_observation_ownership(use_shared_memory):
    serial = SerialEnvPool(_configs())
    shapes = {"point_cloud": (2, 3, 6), "agent_pos": (2, 2)} if use_shared_memory else None
    pool = SubprocessEnvPool(_configs(), observation_shapes=shapes, timeout_sec=30)
    names = [memory.name for memory in pool.memory]
    try:
        expected, actual = serial.reset(), pool.reset()
        previous = copy.deepcopy(actual)
        for step in range(3):
            actions = np.array([[step + 1], [step + 2]], dtype=np.float32)
            a, b = serial.step(actions), pool.step(actions)
            for left, right in zip(a, b):
                for key in left[0]:
                    np.testing.assert_array_equal(left[0][key], right[0][key])
                assert left[1:4] == right[1:4]
                assert left[4]["success"] == right[4]["success"]
                assert right[4]["worker_pid"] != os.getpid()
            assert serial.get_next_seeds() == pool.get_next_seeds()
        for before, after in zip(previous, actual):
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
        assert len({value[4]["worker_pid"] for value in b}) == 2
    finally:
        serial.close()
        pool.close()
    assert all(not process.is_alive() for process in pool.processes)
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_workers_execute_concurrently_and_propagate_failure():
    barrier = mp.get_context("spawn").Barrier(2)
    pool = SubprocessEnvPool(_configs(barrier=barrier), timeout_sec=30)
    try:
        pool.reset()
        # A sequential dispatcher would block forever on the first worker's barrier.
        pool.step(np.ones((2, 1)))
        with pytest.raises(RuntimeError, match="intentional worker step failure"):
            pool.step(np.full((2, 1), -1000))
        assert pool.closed
        assert all(not process.is_alive() for process in pool.processes)
    finally:
        pool.close()


def test_worker_initialization_failure_cleans_up_children():
    before = {process.pid for process in mp.active_children()}
    with pytest.raises(RuntimeError, match="intentional initialization failure"):
        SubprocessEnvPool(_configs(fail_init=True), timeout_sec=30)
    assert {process.pid for process in mp.active_children()} == before


def test_worker_timeout_cleans_up_shared_memory():
    pool = SubprocessEnvPool(_configs(sleep_seconds=1),
                             observation_shapes={"agent_pos": (2, 2)}, timeout_sec=30)
    pool.reset()
    names = [memory.name for memory in pool.memory]
    pool.timeout_sec = 0.05
    with pytest.raises(TimeoutError, match="timed out"):
        pool.step(np.ones((2, 1)))
    assert all(not process.is_alive() for process in pool.processes)
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_workspace_collects_identical_ppo_buffer_with_both_backends(capsys):
    import torch
    from omegaconf import OmegaConf
    from maniflow.common.ppo_buffer_pointcloud import PPOBufferPointcloud
    from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace

    class Actor(torch.nn.Module):
        last_action_noise_stats = {"sigma_mean": torch.tensor(0.02)}

        def get_actions(self, obs, **kwargs):
            batch = obs["agent_pos"].shape[0]
            return torch.ones(batch, 2, 2), torch.zeros(batch, 2, 2, 2), torch.zeros(batch)

    class Critic(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(obs["agent_pos"].shape[0])

    snapshots = []
    for backend in ("serial", "subprocess"):
        workspace = TrainReinFlowRLRoboTwinWorkspace.__new__(TrainReinFlowRLRoboTwinWorkspace)
        workspace.cfg = OmegaConf.create({
            "training": {"seed": 0, "tqdm_interval_sec": 1000},
            "n_obs_steps": 2, "n_action_steps": 2,
            "shape_meta": {"obs": {"point_cloud": {"shape": [3, 6]}, "agent_pos": {"shape": [2]}}},
            "rl": {"env": _configs()[0], "rollout": {
                "n_envs": 2, "backend": backend, "env_seed_stride": 1000,
                "reset_each_iteration": True, "worker_timeout_sec": 30,
            }},
        })
        workspace.itr = 0
        workspace.global_step = 0
        workspace.current_obs = None
        workspace.device = torch.device("cpu")
        workspace.ppo_actor = Actor()
        workspace.critic = Critic()
        buffer = PPOBufferPointcloud(3, 2, 2, (3, 6), (2,), 1, 2, 2, device="cpu")
        pool = workspace._make_envs()
        try:
            log = workspace.collect_rollout(pool, buffer)
            assert log["rollout_chunks"] == 6
            assert workspace.global_step == 6
            terminal = capsys.readouterr().out
            assert "itr=0 reset" in terminal
            assert "succ=2/2 step=6(+2)" in terminal
            snapshots.append(copy.deepcopy({key: getattr(buffer, key) for key in (
                "obs_trajs", "reward_trajs", "terminated_trajs", "truncated_trajs", "firsts_trajs",
            )}))
            next_seeds = workspace._close_envs_for_eval(pool)
            assert next_seeds == [100002, 101002]
            assert workspace.current_obs is None
        finally:
            pool.close()
    for key in snapshots[0]:
        if key == "obs_trajs":
            for field in snapshots[0][key]:
                np.testing.assert_array_equal(snapshots[0][key][field], snapshots[1][key][field])
        else:
            np.testing.assert_array_equal(snapshots[0][key], snapshots[1][key])


def test_seed_cache_atomic_concurrent_reuse_readonly_and_transient_failure(tmp_path):
    cache = ExpertSeedCache(tmp_path, "fingerprint")
    calls = []

    def validate():
        calls.append(1)
        time.sleep(0.05)
        return True, {"info": {"cup": "021_cup/base0"}}

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _: cache.get_or_validate(10, validate), range(4)))
    assert len(calls) == 1
    assert sum(not value[2] for value in values) == 1
    readonly = ExpertSeedCache(tmp_path, "fingerprint", "readonly")
    assert readonly.get_or_validate(10, lambda: pytest.fail("cache hit ran expert"))[2]
    with pytest.raises(FileNotFoundError, match="not prevalidated"):
        readonly.get_or_validate(11, validate)
    with pytest.raises(RuntimeError, match="resource failure"):
        cache.get_or_validate(12, lambda: (_ for _ in ()).throw(RuntimeError("resource failure")))
    assert not (cache.directory / "12.json").exists()
    assert cache.get_or_validate(13, lambda: (False, {"info": {}})) == (False, {"info": {}}, False)
    assert cache.get_or_validate(13, validate)[0] is False
    assert not list(cache.directory.glob(".*"))


def test_seed_fingerprint_tracks_scene_config_and_source_not_experiment_label(tmp_path):
    source = tmp_path / "task.py"
    source.write_text("version one")
    args = {"domain_randomization": {"cluttered_table": True}, "ckpt_setting": "0923"}
    first = seed_cache_fingerprint(args, [source])
    args["ckpt_setting"] = "1016"
    assert seed_cache_fingerprint(args, [source]) == first
    args["domain_randomization"]["cluttered_table"] = False
    assert seed_cache_fingerprint(args, [source]) != first
    args["domain_randomization"]["cluttered_table"] = True
    source.write_text("version two")
    assert seed_cache_fingerprint(args, [source]) != first


def test_rl_head_only_configuration_keeps_colored_cloud_and_task_yaml(monkeypatch):
    import maniflow.env.robotwin.robotwin_rl_env as module
    task_yaml = ROBOTWIN_ROOT / "task_config/demo_messy_ur5.yml"
    original = task_yaml.read_bytes()
    monkeypatch.setattr(module, "_task_env", lambda name: FakeEnv())
    env = RoboTwinRLEnv("place_empty_cup", "demo_messy_ur5", head_pointcloud_only=True)
    assert env.base_args["camera"]["collect_head_camera"]
    assert not env.base_args["camera"]["collect_wrist_camera"]
    assert env.base_args["data_type"]["pointcloud"]
    assert not env.base_args["data_type"]["rgb"]
    encoded = env._encode_obs({
        "pointcloud": np.ones((3, 6)), "joint_action": {"vector": np.zeros(14)},
        "observation": {"head_camera": {"rgb": np.ones((240, 320, 3))}},
    })
    assert set(encoded) == {"point_cloud", "agent_pos"}
    assert encoded["point_cloud"].shape[-1] == 6
    assert task_yaml.read_bytes() == original


def _base_task_method(name):
    # Execute the real method without importing the simulator's GPU/planner dependencies.
    path = ROBOTWIN_ROOT / "envs/_base_task.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Base_Task")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np, "deepcopy": copy.deepcopy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("guard", [None, "render_freq", "save_data", "eval_video_path", "crazy_random_light"])
def test_render_deferral_always_refreshes_observations_and_preserves_other_paths(guard):
    calls = []
    scene = SimpleNamespace(
        update_render=lambda: calls.append("sync"), ambient_light=np.ones(3),
        set_ambient_light=lambda value: None,
    )
    task = SimpleNamespace(
        render_on_observation_only=True, render_freq=0, save_data=False,
        eval_video_path=None, crazy_random_light=False, scene=scene,
        point_light_lst=[], direction_light_lst=[],
        robot=SimpleNamespace(left_camera=SimpleNamespace(get_pose=lambda: None),
                              right_camera=SimpleNamespace(get_pose=lambda: None)),
        cameras=SimpleNamespace(update_wrist_camera=lambda *args: None,
                                update_picture=lambda: calls.append("picture"), get_config=lambda: {}),
        data_type={},
    )
    if guard:
        setattr(task, guard, True)
    task._update_render = MethodType(_base_task_method("_update_render"), task)
    task.get_obs = MethodType(_base_task_method("get_obs"), task)
    task._update_render()
    task._update_render()
    assert calls.count("sync") == (2 if guard else 0)
    task.get_obs()
    assert calls[-2:] == ["sync", "picture"]


def test_chunk_still_observes_every_action_and_retains_last_two_frames():
    task = SimpleNamespace(eval_success=False, step_lim=100, take_action_cnt=0)
    observed = []

    def take_action(action, action_type):
        task.take_action_cnt += 1

    def get_obs():
        observed.append(task.take_action_cnt)
        return {"pointcloud": np.full((3, 6), task.take_action_cnt),
                "joint_action": {"vector": np.full(2, task.take_action_cnt)}}

    task.take_action = take_action
    task.get_obs = get_obs
    env = RoboTwinRLEnv.__new__(RoboTwinRLEnv)
    env.task_env = task
    env.n_action_steps = 16
    env.n_obs_steps = 2
    env.action_type = "qpos"
    env._success_seen = False
    env.obs = deque(maxlen=3)
    env._last_observation = get_obs()
    env.obs.append(env._encode_obs(env._last_observation))
    rewards_seen = []

    def reward(action, was_success, previous):
        rewards_seen.append(previous["joint_action"]["vector"][0])
        return float(task.take_action_cnt), {"forward": 1.0}

    env._raw_reward_after_action = reward
    obs, reward_sum, terminated, truncated, info = env.step(np.zeros((15, 2)))
    assert observed == list(range(16))
    assert rewards_seen == list(range(15))
    np.testing.assert_array_equal(obs["agent_pos"][:, 0], [14, 15])
    assert reward_sum == sum(range(1, 16))
    assert info["executed_action_steps"] == 15
    assert not terminated and not truncated
