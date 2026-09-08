"""Puts `python/` on sys.path so tests import `rl` and `communication` the same
way train.py does, without needing an installed package."""

import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
