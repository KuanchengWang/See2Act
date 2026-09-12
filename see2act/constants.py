"""Fixed quantities shared by training and inference."""
import os

import numpy as np

KEYFRAMES = ("pick", "place")

# Canonical global views C_T: name -> (position, euler xyz), in the world frame of the tabletop.
INIT_VIEWS = {
    "top": ([0.5, 0.0, 1.5], [0.0, np.pi, np.pi]),
    "front": ([1.85, 0.0, 1.25], [np.pi / 3.5, np.pi, -np.pi / 2]),
    "back": ([-0.65, 0.0, 1.1], [-np.pi / 4, np.pi, -np.pi / 2]),
    "left": ([0.5, 1.0, 1.1], [-np.pi / 4.5, -np.pi, np.pi]),
    "right": ([0.5, -1.0, 1.1], [np.pi / 4.5, -np.pi, -np.pi]),
}

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_text_embeddings():
    """Fixed 512-d text embeddings of the keyframe names, used to condition the image encoder (FiLM)."""
    d = np.load(os.path.join(ASSETS_DIR, "keyframe_text_embeddings.npz"))
    return {k: d[k].astype(np.float32) for k in KEYFRAMES}
