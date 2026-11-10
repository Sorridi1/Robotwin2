"""Resolve this checkout without changing the installed Conda environment."""

import importlib
import os
from pathlib import Path
import sys

POLICY_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = POLICY_ROOT.parents[1]
MANIFLOW_SOURCE = ROBOTWIN_ROOT / "policy/ManiFlow/ManiFlow"


def configure_paths():
    paths = [str(POLICY_ROOT), str(MANIFLOW_SOURCE), str(ROBOTWIN_ROOT)]
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    package = importlib.import_module("maniflow")
    if not all(Path(path).resolve().is_relative_to(MANIFLOW_SOURCE) for path in package.__path__):
        raise ImportError(f"maniflow was imported from another checkout: {list(package.__path__)}")
    # Evaluation subprocesses need the same source; this changes this process only.
    previous = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + [p for p in previous if p]))
    os.environ.setdefault("VK_ICD_FILENAMES", str(ROBOTWIN_ROOT / "policy/ManiFlow/scripts/nvidia_icd.json"))
    os.environ.setdefault("CUROBO_TORCH_COMPILE", "0")
    os.environ.setdefault("CUROBO_TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
