import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maniflow.env.robotwin import robotwin_rl_env as module
from maniflow.env.robotwin.robotwin_rl_env import RoboTwinRLEnv, ROBOTWIN_ROOT
from maniflow.env.robotwin.scene_validator import CheapSceneValidator, SceneValidationResult


def pose(position=(0, 0, 0.8)):
    return SimpleNamespace(p=np.asarray(position, dtype=float), q=np.array([1., 0., 0., 0.]))


class Articulation:
    def __init__(self):
        self.qpos = np.zeros(3)
        self.qvel = np.zeros(3)
        self.root_pose = pose()
        self.limits = np.array([[-1., 1.], [-np.inf, np.inf], [-np.inf, 2.]])

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel

    def get_root_pose(self):
        return self.root_pose

    def get_qlimits(self):
        return self.limits


class Actor:
    def __init__(self, name, position=(0, 0, 0.8)):
        self.name = name
        self.pose = pose(position)

    def get_name(self):
        return self.name

    def get_pose(self):
        return self.pose

    def get_functional_point(self, *args):
        return self.pose


def task():
    robot = SimpleNamespace(left_entity=Articulation(), right_entity=Articulation())
    actors = [Actor("cup"), Actor("coaster", (.2, 0, .75))]
    scene = SimpleNamespace(get_all_actors=lambda: actors, get_all_articulations=lambda: [],
                            get_contacts=lambda: [])
    return SimpleNamespace(robot=robot, scene=scene, check_success=lambda: False)


def test_valid_scene_checks_are_read_only_and_unbounded_joints_are_legal():
    env = task()
    env.robot.left_entity.qpos[1] = 100.0
    before = env.robot.left_entity.qpos.copy()
    result = CheapSceneValidator().validate(env)
    assert result.valid
    assert result.metrics["actor_poses_checked"] == 2
    np.testing.assert_array_equal(before, env.robot.left_entity.qpos)


@pytest.mark.parametrize("field", ["qpos", "qvel", "root_position", "root_quaternion"])
def test_nonfinite_robot_state_is_rejected(field):
    env = task()
    entity = env.robot.left_entity
    if field in ("qpos", "qvel"):
        getattr(entity, field)[0] = np.nan
    else:
        getattr(entity.root_pose, "p" if field == "root_position" else "q")[0] = np.inf
    assert CheapSceneValidator().validate(env).reason == "nonfinite_robot_state"


def test_joint_limit_tolerance_and_nonfinite_actor_pose():
    env = task()
    env.robot.left_entity.qpos[0] = 1.02
    assert CheapSceneValidator({"joint_limit_tolerance": .05}).validate(env).valid
    assert CheapSceneValidator({"joint_limit_tolerance": .01}).validate(env).reason == "joint_limit_violation"
    env.robot.left_entity.qpos[0] = 0
    env.scene.get_all_actors()[0].pose.q[0] = np.nan
    assert CheapSceneValidator().validate(env).reason == "nonfinite_actor_pose"


def test_contacts_allow_supporting_contact_and_reject_only_deep_overlap():
    env = task()
    point = SimpleNamespace(separation=-.001)
    bodies = [SimpleNamespace(entity=actor) for actor in env.scene.get_all_actors()]
    env.scene.get_contacts = lambda: [SimpleNamespace(points=[point], bodies=bodies)]
    assert CheapSceneValidator().validate(env).valid
    point.separation = -.04
    result = CheapSceneValidator().validate(env)
    assert result.reason == "severe_penetration"
    assert result.metrics["max_penetration_m"] == .04
    assert CheapSceneValidator({"check_initial_penetration": False}).validate(env).valid
    point = SimpleNamespace()
    result = CheapSceneValidator().validate(env)
    assert result.valid and not result.metrics["penetration_check_supported"]


def test_initial_success_and_optional_hook():
    env = task()
    env.check_success = lambda: True
    assert CheapSceneValidator().validate(env).reason == "initial_success"
    assert CheapSceneValidator({"reject_initial_success": False}).validate(env).valid
    env.check_success = lambda: False
    env.cheap_validate_scene = lambda config: (False, "task_specific_invalid", {"actor": "cup"})
    assert CheapSceneValidator().validate(env).reason == "task_specific_invalid"
    assert CheapSceneValidator({"enable_task_specific_check": False}).validate(env).valid


def test_place_empty_cup_hook_uses_broad_table_bounds_not_expert_feasibility():
    source = ROBOTWIN_ROOT / "envs/place_empty_cup.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "cheap_validate_scene")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    env = SimpleNamespace(table=Actor("table", (0, 0, .74)),
                          cup=Actor("cup", (.5, .2, .8)), coaster=Actor("coaster", (-.4, -.2, .75)),
                          SCENE_VALIDATION_BOUNDS={"workspace_min": [-.65, -.4, -.1],
                                                   "workspace_max": [.65, .4, .75]})
    validate = namespace["cheap_validate_scene"]
    assert validate(env, {})[0]  # Far apart and outside original spawn strips is still legal.
    env.cup.pose.p[0] = 2
    assert validate(env, {})[1] == "task_specific_invalid"
    env.cup.pose.p = np.array([.2, 0, .1])
    assert validate(env, {})[1] == "task_specific_invalid"


class FakeUnstableError(Exception):
    pass


class ResetTask:
    def __init__(self, allow_expert=False, unstable_seed=None, fail_setup=False, expert_failure_seed=None):
        self.allow_expert = allow_expert
        self.unstable_seed = unstable_seed
        self.fail_setup = fail_setup
        self.expert_failure_seed = expert_failure_seed
        self.setup_seeds = []
        self.closed_seeds = []
        self.play_count = 0
        self.obs_count = 0

    def setup_demo(self, seed, **kwargs):
        self.seed = seed
        self.setup_seeds.append(seed)
        if seed == self.unstable_seed:
            raise FakeUnstableError("unstable candidate")
        if self.fail_setup:
            raise RuntimeError("missing robot asset")
        state = task()
        self.robot, self.scene = state.robot, state.scene
        self.cup, self.coaster = self.scene.get_all_actors()
        self.info = {"info": {}}
        self.expert_executed = False
        self.plan_success = False  # Expert feasibility is deliberately irrelevant to cheap mode.
        self.eval_success = False

    def get_scene_metadata(self):
        return copy.deepcopy(self.info)

    def play_once(self):
        if not self.allow_expert:
            raise AssertionError("cheap/none reset must never call play_once")
        self.play_count += 1
        self.expert_executed = True
        self.plan_success = self.seed != self.expert_failure_seed
        return self.info

    def check_success(self):
        return self.expert_executed

    def get_obs(self):
        self.obs_count += 1
        return {"pointcloud": np.zeros((1024, 6)), "joint_action": {"vector": np.zeros(14)}}

    def close_env(self, clear_cache=False):
        self.closed_seeds.append(self.seed)


def make_env(monkeypatch, fake, mode="cheap", tmp_path=None, **kwargs):
    monkeypatch.chdir(Path.cwd())
    monkeypatch.setattr(module, "_task_env", lambda name: fake)
    monkeypatch.setattr(module, "_is_unstable_setup", lambda exc: isinstance(exc, FakeUnstableError))
    options = {} if mode is None else {"scene_validation": {"mode": mode, "max_seed_tries": 3}}
    options.update(kwargs)
    env = RoboTwinRLEnv(
        "place_empty_cup", "demo_messy_ur5", start_seed=100000,
        expert_seed_cache_dir=str(tmp_path) if tmp_path else None,
        expert_seed_cache_mode="readwrite" if tmp_path else "off", **options,
    )
    env._clear_backend_cache = lambda: None
    return env


def test_cheap_reset_never_runs_expert_and_keeps_single_initialized_scene(monkeypatch):
    fake = ResetTask()
    env = make_env(monkeypatch, fake, expert_check=True)
    obs = env.reset()
    assert obs["point_cloud"].shape == (2, 1024, 6)
    assert fake.setup_seeds == [100000]
    assert fake.closed_seeds == []
    assert fake.play_count == 0 and not fake.plan_success
    assert env._has_active_env
    assert env.last_validation_info["accepted_seed"] == 100000
    assert env.last_validation_info["setup_demo_count"] == 1
    assert env.last_validation_info["validation_mode"] == "cheap"


def test_invalid_candidate_is_closed_next_seed_built_and_accepted_scene_kept(monkeypatch):
    fake = ResetTask()
    env = make_env(monkeypatch, fake)
    results = iter([SceneValidationResult(False, "severe_penetration"), SceneValidationResult(True)])
    env.scene_validator.validate = lambda scene: next(results)
    env.reset()
    assert fake.setup_seeds == [100000, 100001]
    assert fake.closed_seeds == [100000]
    assert env.next_seed == 100002
    assert env.last_validation_info["rejection_reasons"] == {"severe_penetration": 1}


@pytest.mark.parametrize("mode", ["cheap", "none", "expert"])
def test_unstable_setup_is_rejected_but_unknown_setup_errors_propagate(monkeypatch, mode):
    fake = ResetTask(allow_expert=True, unstable_seed=100000)
    env = make_env(monkeypatch, fake, mode)
    env.reset()
    assert env.last_validation_info["accepted_seed"] == 100001
    assert env.last_validation_info["rejection_reasons"] == {"unstable_setup": 1}
    assert fake.closed_seeds[0] == 100000
    fake.fail_setup = True
    with pytest.raises(RuntimeError, match="missing robot asset"):
        env.reset()
    assert env.last_validation_info["setup_exception_types"] == {"RuntimeError": 1}
    assert env.last_reset_info["validation"]["accepted_seed"] is None


def test_legacy_expert_configuration_and_failure_filter_are_preserved(monkeypatch):
    fake = ResetTask(allow_expert=True, expert_failure_seed=100000)
    env = make_env(monkeypatch, fake, mode=None, expert_check=True)
    env.reset()
    assert env.validation_mode == "expert"
    assert fake.play_count == 2
    assert fake.setup_seeds == [100000, 100001, 100001]
    assert fake.closed_seeds == [100000, 100001]
    assert env.last_validation_info["rejection_reasons"] == {"expert_failure": 1}


@pytest.mark.parametrize("legacy", [False, True])
def test_none_mode_skips_both_validators(monkeypatch, legacy):
    fake = ResetTask()
    env = make_env(monkeypatch, fake, mode=None if legacy else "none", expert_check=False)
    env.scene_validator.validate = lambda scene: pytest.fail("none mode ran cheap validation")
    env.reset()
    assert env.validation_mode == "none"
    assert fake.setup_seeds == [100000]
    assert fake.play_count == 0


def test_cache_modes_and_thresholds_are_isolated_and_positive_hits_are_rechecked(monkeypatch, tmp_path):
    expert = make_env(monkeypatch, ResetTask(allow_expert=True, expert_failure_seed=100000), "expert", tmp_path)
    assert not expert.validate_seed(100000)[0]
    cheap = make_env(monkeypatch, ResetTask(), "cheap", tmp_path)
    assert expert.expert_seed_cache.directory != cheap.expert_seed_cache.directory
    cheap.reset(seed=100000)
    assert cheap.last_validation_info["accepted_seed"] == 100000
    assert cheap.last_validation_info["cache_misses"] == 1
    results = iter([SceneValidationResult(False, "joint_limit_violation"), SceneValidationResult(True)])
    cheap.scene_validator.validate = lambda scene: next(results)
    cheap.reset(seed=100000)
    assert cheap.last_validation_info["cache_hits"] == 1
    assert cheap.last_validation_info["accepted_seed"] == 100001
    fresh = make_env(monkeypatch, ResetTask(), "cheap", tmp_path)
    fresh.reset(seed=100000)
    assert fresh.task_env.setup_seeds == [100001]  # Cached cheap rejection skips only this mode's seed.
    altered = make_env(monkeypatch, ResetTask(), "cheap", tmp_path,
                       scene_validation={"mode": "cheap", "joint_limit_tolerance": .1})
    assert fresh.expert_seed_cache.directory != altered.expert_seed_cache.directory


def test_rejection_exhaustion_reports_reasons_and_does_not_keep_invalid_scene(monkeypatch):
    env = make_env(monkeypatch, ResetTask())
    env.scene_validator.validate = lambda scene: SceneValidationResult(False, "initial_success")
    with pytest.raises(RuntimeError, match="initial_success"):
        env.reset()
    assert not env._has_active_env
    assert env.last_validation_info["rejected_count"] == 3
