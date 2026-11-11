"""Read-only checks for invalid initial scenes, not proofs of task solvability."""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


DEFAULT_SCENE_VALIDATION = {
    "check_finite_state": True,
    "check_joint_limits": True,
    "joint_limit_tolerance": 0.05,
    "reject_initial_success": True,
    "check_initial_penetration": True,
    "max_initial_penetration": 0.03,
    "enable_task_specific_check": True,
    "task_specific": {},
}


@dataclass(frozen=True)
class SceneValidationResult:
    valid: bool
    reason: str = "accepted"
    metrics: Dict[str, Any] = field(default_factory=dict)


class CheapSceneValidator:
    def __init__(self, config=None):
        self.config = {**DEFAULT_SCENE_VALIDATION, **dict(config or {})}
        for key in ("joint_limit_tolerance", "max_initial_penetration"):
            value = float(self.config[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"scene_validation.{key} must be finite and nonnegative")

    @staticmethod
    def _finite_pose(pose):
        return bool(np.isfinite(pose.p).all() and np.isfinite(pose.q).all())

    def validate(self, task_env):
        metrics = {}
        robot = task_env.robot
        articulations = []
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity", None)
            if entity is not None and not any(entity is item for item in articulations):
                articulations.append(entity)
        metrics["robot_articulations_checked"] = len(articulations)
        if not articulations:
            raise RuntimeError("Cheap scene validation cannot access RoboTwin robot articulations")

        for index, entity in enumerate(articulations):
            qpos = np.asarray(entity.get_qpos()).reshape(-1)
            if self.config["check_finite_state"]:
                qvel = np.asarray(entity.get_qvel()).reshape(-1)
                if (not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
                        or not self._finite_pose(entity.get_root_pose())):
                    return SceneValidationResult(False, "nonfinite_robot_state", {**metrics, "articulation": index})
            if self.config["check_joint_limits"]:
                get_limits = getattr(entity, "get_qlimits", None)
                if not callable(get_limits):
                    metrics["joint_limits_unavailable"] = True
                    continue
                limits = np.asarray(get_limits())
                if limits.shape != (qpos.size, 2) or np.isnan(limits).any():
                    raise ValueError(f"Invalid articulation joint limit metadata: {limits.shape}")
                lower, upper = limits[:, 0], limits[:, 1]
                if np.any(lower > upper):
                    raise ValueError("Articulation lower joint limits exceed upper limits")
                tolerance = float(self.config["joint_limit_tolerance"])
                below = np.isfinite(lower) & (qpos < lower - tolerance)
                above = np.isfinite(upper) & (qpos > upper + tolerance)
                if np.any(below | above):
                    return SceneValidationResult(False, "joint_limit_violation", {
                        **metrics, "articulation": index,
                        "joint_indices": np.flatnonzero(below | above).tolist(),
                    })

        if self.config["check_finite_state"]:
            actors = list(task_env.scene.get_all_actors())
            get_articulations = getattr(task_env.scene, "get_all_articulations", None)
            if callable(get_articulations):
                actors.extend(get_articulations())
            metrics["actor_poses_checked"] = len(actors)
            for actor in actors:
                if not self._finite_pose(actor.get_pose()):
                    return SceneValidationResult(False, "nonfinite_actor_pose", {
                        **metrics, "actor": actor.get_name(),
                    })

        if self.config["check_initial_penetration"]:
            get_contacts = getattr(task_env.scene, "get_contacts", None)
            metrics["penetration_check_supported"] = callable(get_contacts)
            max_depth = 0.0
            deepest_pair = []
            if callable(get_contacts):
                for contact in get_contacts():
                    for point in contact.points:
                        if not hasattr(point, "separation"):
                            metrics["penetration_check_supported"] = False
                            continue
                        separation = float(point.separation)
                        if not np.isfinite(separation):
                            return SceneValidationResult(False, "nonfinite_contact_state", metrics)
                        # SAPIEN exposes PhysX contact separation: negative means overlap.
                        depth = max(0.0, -separation)
                        if depth > max_depth:
                            max_depth = depth
                            deepest_pair = [body.entity.get_name() for body in contact.bodies]
            metrics["max_penetration_m"] = max_depth
            metrics["deepest_contact_pair"] = deepest_pair
            if max_depth > float(self.config["max_initial_penetration"]):
                return SceneValidationResult(False, "severe_penetration", metrics)

        if self.config["reject_initial_success"] and bool(task_env.check_success()):
            return SceneValidationResult(False, "initial_success", metrics)

        hook = getattr(task_env, "cheap_validate_scene", None)
        if self.config["enable_task_specific_check"] and callable(hook):
            # Task hook contract: (valid: bool, reason: str, metrics: dict).
            valid, reason, task_metrics = hook(dict(self.config["task_specific"]))
            metrics["task_specific"] = task_metrics
            if not valid:
                return SceneValidationResult(False, str(reason), metrics)
        return SceneValidationResult(True, metrics=metrics)
