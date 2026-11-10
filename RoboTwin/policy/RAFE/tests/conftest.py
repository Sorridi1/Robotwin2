"""Test the copied RAFE package without an editable install or simulator."""
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "ManiFlow"
sys.path.insert(0, str(PACKAGE_ROOT))
