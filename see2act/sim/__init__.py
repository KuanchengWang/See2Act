"""PyBullet simulation used by See2Act: a trimmed fork of Ravens (Zeng et al., 2020) with the
four occlusion tasks of the paper (place-red-in-green, bin-picking, put-within-shelf, bin-search)."""
import os

ASSETS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets") + os.sep

from see2act.sim.environment import Environment  # noqa: E402
from see2act.sim import tasks  # noqa: E402

__all__ = ["ASSETS_ROOT", "Environment", "tasks"]
