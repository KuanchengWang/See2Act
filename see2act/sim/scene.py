"""Scene description: the list of task objects (name, position, quaternion) that defines an episode.

The policy never sees a scene directly; the renderer rebuilds it in PyBullet to synthesize the views
the policy asks for (the "digital twin"). For storage and inter-process transport a scene is encoded
into a flat float array with 8 numbers per object: [code, x, y, z, qx, qy, qz, qw].
"""
from collections import namedtuple

import numpy as np

SceneObject = namedtuple("SceneObject", ["name", "position", "quaternion"])

COLOR_CODES = {"blue": 1, "red": 2, "green": 3, "orange": 4, "yellow": 5, "purple": 6, "pink": 7, "cyan": 8,
               "brown": 9, "gray": 10, "white": 11}
TYPE_CODES = {"block": 1, "bowl": 2, "box": 3}
_CODE_TO_COLOR = {v: k for k, v in COLOR_CODES.items()}
_CODE_TO_TYPE = {v: k for k, v in TYPE_CODES.items()}


def split_name(name):
    """'red_block' -> ('red', 'block')."""
    color, otype = name.split("_")
    return color, otype


def encode_scene(scene):
    """List of SceneObject -> flat float64 array of shape (8 * n_objects,)."""
    out = []
    for obj in scene:
        color, otype = split_name(obj.name)
        code = COLOR_CODES[color] * 10 + TYPE_CODES[otype]
        out += [code, *[float(v) for v in obj.position], *[float(v) for v in obj.quaternion]]
    return np.asarray(out, dtype=np.float64)


def decode_scene(encoded):
    """Inverse of encode_scene."""
    encoded = np.asarray(encoded, dtype=np.float64).reshape(-1)
    assert encoded.shape[0] % 8 == 0, encoded.shape
    scene = []
    for j in range(0, encoded.shape[0], 8):
        code = int(round(encoded[j]))
        name = f"{_CODE_TO_COLOR[code // 10]}_{_CODE_TO_TYPE[code % 10]}"
        scene.append(SceneObject(name, tuple(encoded[j + 1:j + 4]), tuple(encoded[j + 4:j + 8])))
    return scene


def find_object(scene, name):
    for obj in scene:
        if obj.name == name:
            return obj
    return None
