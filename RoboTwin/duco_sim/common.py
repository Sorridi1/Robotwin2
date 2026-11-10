"""Read-only reuse of existing RGB-D geometry tools, without ManiSkill imports."""
from pathlib import Path
import sys

ROOT = Path('/home/ljj/code/Robotwin2/RoboTwin')
GEOMETRY_ROOT = Path('/home/ljj/code/maniskill')
MANIFLOW_ROOT = ROOT / 'policy/ManiFlow/ManiFlow'
DEFAULT_RUN = Path('/media/Elements1/ljj/ManiFlow/outputs/2026.08.27/20.07.11_train_maniflow_pointcloud_policy_wipe_whiteboard')


def add_geometry_path(path=GEOMETRY_ROOT):
    path = Path(path).resolve()
    if not (path / 'duco_wipe_scene.py').is_file():
        raise FileNotFoundError(f'Missing existing pure RGB-D geometry modules: {path}')
    if str(path) not in sys.path:
        sys.path.append(str(path))


def add_policy_path():
    if str(MANIFLOW_ROOT) not in sys.path:
        sys.path.insert(0, str(MANIFLOW_ROOT))


def write_new_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
