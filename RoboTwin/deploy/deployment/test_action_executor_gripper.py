"""Tests for deployment-time binary gripper decoding."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.action_executor import ActionExecutor, decode_binary_gripper_widths
from deploy_bsxyz_duco import _plan_debounced_gripper_commands


def test_binary_gripper_decoding():
    signal = np.array([-1.2, -1.02, -0.2, -0.199, 0.7273, 1.2])
    widths, closed = decode_binary_gripper_widths(signal, threshold_norm=-0.2)

    np.testing.assert_array_equal(
        closed,
        np.array([True, True, False, False, False, False]),
    )
    np.testing.assert_allclose(
        widths,
        np.array([0.0, 0.0, 0.095, 0.095, 0.095, 0.095]),
    )


def test_executor_accepts_small_continuous_overshoot():
    action = np.zeros((20, 10), dtype=np.float32)
    action[:, 2] = 0.0
    action[:, 3] = 1.0
    action[:, 7] = 1.0
    action[:, 9] = np.linspace(-1.002, -1.023, 20)

    plan = ActionExecutor().prepare_commands(action, clip_workspace=True)

    assert plan["checks"]["gripper"]["safe"]
    assert plan["all_safe"]
    assert np.all(plan["gripper_widths_continuous"] < 0.0)
    np.testing.assert_array_equal(plan["gripper_widths"], np.zeros(20))


def test_chunk_start_anchor_preserves_terminal_waypoint():
    action = np.zeros((20, 10), dtype=np.float32)
    action[:, 0] = np.linspace(-0.2, 0.2, 20)
    action[:, 3] = 1.0
    action[:, 7] = 1.0
    action[:, 9] = -1.0

    executor = ActionExecutor()
    unanchored = executor.prepare_commands(action)
    current = unanchored["tcp_base_rot6d"][0].copy()
    current[:3] += np.array([0.02, -0.01, 0.005])
    anchored = executor.prepare_commands(
        action,
        current_tcp_base=current,
        current_tcp_rotation_rep="rotation_6d",
        anchor_first_position=True,
    )

    np.testing.assert_allclose(anchored["tcp_base_rot6d"][0, :3], current[:3])
    np.testing.assert_allclose(
        anchored["tcp_base_rot6d"][-1, :3],
        unanchored["tcp_base_rot6d"][-1, :3],
    )


def test_generic_gripper_debounce_rejects_only_short_runs():
    def schedule(closed, current_width):
        closed = np.asarray(closed, dtype=bool)
        return _plan_debounced_gripper_commands(
            {
                "gripper_closed": closed,
                "gripper_widths": np.where(closed, 0.0, 0.095),
            },
            current_width=current_width,
            closed_width_threshold=0.03,
            min_run_steps=3,
        )

    flicker = schedule([True] * 16 + [False, True, True, False], 0.0)
    assert flicker["commands"] == {}

    confirmed_open = schedule([True] * 15 + [False] * 5, 0.0)
    assert confirmed_open["commands"] == {15: 0.095}

    mixed_while_open = schedule([True] * 14 + [False] * 6, 0.095)
    assert mixed_while_open["commands"] == {0: 0.0, 14: 0.095}

    confirmed_close = schedule([True] * 20, 0.095)
    assert confirmed_close["commands"] == {0: 0.0}


if __name__ == "__main__":
    test_binary_gripper_decoding()
    test_executor_accepts_small_continuous_overshoot()
    test_chunk_start_anchor_preserves_terminal_waypoint()
    test_generic_gripper_debounce_rejects_only_short_runs()
    print("binary gripper decoding: ok")
