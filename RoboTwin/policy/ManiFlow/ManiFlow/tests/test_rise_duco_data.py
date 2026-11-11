from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from maniflow.common.rise_duco_data import (
    DatasetContractError,
    build_agent_state,
    discover_episodes,
    scan_episodes,
)


def _write_episode(root: Path, missing_depth_timestamp=None):
    episode = root / "demonstration_1"
    camera = episode / "cam_243222076209"
    for modality in ("color", "depth", "joint", "gripper_command"):
        (camera / modality).mkdir(parents=True, exist_ok=True)
    for index, timestamp in enumerate((1000, 1100, 1200)):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(
            camera / "color" / f"{timestamp}.png"
        )
        if timestamp != missing_depth_timestamp:
            Image.fromarray(np.full((4, 5), 500, dtype=np.uint16)).save(
                camera / "depth" / f"{timestamp}.png"
            )
        np.save(camera / "joint" / f"{timestamp}.npy", np.arange(6, dtype=np.float32) + index)
        np.save(
            camera / "gripper_command" / f"{timestamp}.npy",
            np.asarray([1000.0 if index == 2 else 0.0]),
        )
    (episode / "metadata.json").write_text('{"finish_time": 1200}', encoding="utf-8")
    (episode / "timestamp.txt").write_text("1200\n", encoding="utf-8")


def test_discover_and_scan_strictly_synchronized_episode(tmp_path):
    _write_episode(tmp_path)
    episodes = discover_episodes(tmp_path, strict_metadata=True)
    assert len(episodes) == 1
    assert episodes[0].frame_ids == (1000, 1100, 1200)

    report = scan_episodes(episodes, validate_arrays=True)
    assert report["episodes"] == 1
    assert report["frames_total"] == 3
    assert report["frame_delta_ms"]["median"] == 100.0
    assert report["gripper_raw_values"] == [0.0, 1000.0]


def test_missing_modality_timestamp_is_rejected(tmp_path):
    _write_episode(tmp_path, missing_depth_timestamp=1100)
    with pytest.raises(DatasetContractError, match="timestamp mismatch"):
        discover_episodes(tmp_path)


def test_agent_state_layout_and_gripper_range():
    state = build_agent_state(np.arange(6, dtype=np.float32), 1000.0)
    np.testing.assert_array_equal(
        state,
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(DatasetContractError, match="outside"):
        build_agent_state(np.zeros(6, dtype=np.float32), 1001.0)
