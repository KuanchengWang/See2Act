"""Camera configurations."""
import numpy as np
import pybullet as p

IMAGE_SIZE = (320, 320)
INTRINSICS = (450.0, 0, 160.0, 0, 450.0, 160.0, 0, 0, 1)
ZRANGE = (0.01, 10.0)


def camera_config(position, rotation_xyzw, image_size=IMAGE_SIZE, intrinsics=INTRINSICS, zrange=ZRANGE):
    """Render config for Environment.render_camera. The camera looks along its local +z axis with local -y up."""
    return {
        "image_size": tuple(image_size),
        "intrinsics": tuple(intrinsics),
        "position": tuple(float(v) for v in position),
        "rotation": tuple(float(v) for v in rotation_xyzw),
        "zrange": tuple(zrange),
        "noise": False,
    }


class AgentCameras:
    """Fixed cameras recorded with every demonstration (top-down and front); not used by the policy,
    which renders its own actively chosen views."""

    top_down_position = (0.5, 0, 1.5)
    top_down_rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))
    front_position = (1.65, 0, 1.1)
    front_rotation = p.getQuaternionFromEuler((np.pi / 4, np.pi, -np.pi / 2))
    NAMES = ("top", "front")
    CONFIG = [camera_config(top_down_position, top_down_rotation), camera_config(front_position, front_rotation)]


class Oracle:
    """Near-orthographic top-down camera used only by the scripted demonstrator."""

    image_size = (480, 640)
    intrinsics = (63e4, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)
    position = (0.5, 0, 1000.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))
    CONFIG = [camera_config(position, rotation, image_size, intrinsics, (999.7, 1001.0))]
