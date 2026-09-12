"""Rigid-body geometry: action orientation parameterization and world <-> camera frame transforms."""
import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation as R
from transforms3d import euler as t3d_euler


# ----------------------------------------------------------------------------- action orientation
# A keyframe action is a 6-vector [x, y, z, rx, ry, rz]. The three angles are the (x, y, z) angles of an
# intrinsic z-x-y Euler decomposition of the keyframe quaternion; elsewhere (camera-frame transforms and the
# camera schedule) they are interpreted as extrinsic x-y-z angles. The two conventions coincide for the yaw-only
# orientations of the block-and-bowl tasks; the pair below is exactly what the policy was trained with.
def action_quat_to_euler(quat_xyzw):
    q = quat_xyzw
    zxy = t3d_euler.quat2euler(np.array([q[3], q[0], q[1], q[2]]), axes="rzxy")
    return np.array([zxy[1], zxy[2], zxy[0]], dtype=np.float32)


def action_euler_to_quat(euler):
    q = t3d_euler.euler2quat(euler[2], euler[0], euler[1], axes="rzxy")  # wxyz
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def quat_to_euler(quat_xyzw):
    """Extrinsic x-y-z angles (PyBullet convention)."""
    return np.array(p.getEulerFromQuaternion(list(quat_xyzw)), dtype=np.float64)


def euler_to_quat(euler_xyz):
    return np.array(p.getQuaternionFromEuler(list(euler_xyz)), dtype=np.float64)


# ----------------------------------------------------------------------------- frame transforms
def world_to_camera(actions, cam_pos, cam_euler):
    """Express world-frame actions (N, 6) in the frames of cameras at cam_pos (N, 3), cam_euler (N, 3)."""
    actions = np.asarray(actions, dtype=np.float32)
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_euler = np.asarray(cam_euler, dtype=np.float32)
    n = actions.shape[0]
    rot_act = R.from_euler("xyz", actions[:, 3:])
    rot_cam = R.from_euler("xyz", cam_euler)
    pos = np.array([rot_cam[i].inv().apply((actions[:, :3] - cam_pos)[i]) for i in range(n)], dtype=np.float32)
    ori = (rot_cam.inv() * rot_act).as_euler("xyz", degrees=False).astype(np.float32)
    return np.hstack([pos, ori])


def camera_to_world(actions, cam_pos, cam_euler):
    """Inverse of world_to_camera."""
    actions = np.asarray(actions, dtype=np.float32)
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_euler = np.asarray(cam_euler, dtype=np.float32)
    n = actions.shape[0]
    rot_cam = R.from_euler("xyz", cam_euler)
    rot_act = R.from_euler("xyz", actions[:, 3:])
    ori = (rot_cam * rot_act).as_euler("xyz", degrees=False).astype(np.float32)
    pos = np.array([rot_cam[i].apply(actions[i, :3]) for i in range(n)], dtype=np.float32) + cam_pos
    return np.hstack([pos, ori])


# ----------------------------------------------------------------------------- normalization
def normalize(actions, bounds):
    """Map each dimension from [lo, hi] to [-0.5, 0.5] (torch)."""
    bounds = torch.as_tensor(bounds, dtype=actions.dtype, device=actions.device)
    return -0.5 + (actions - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def unnormalize(actions, bounds):
    bounds = torch.as_tensor(bounds, dtype=actions.dtype, device=actions.device)
    return (actions + 0.5) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


# ----------------------------------------------------------------------------- quaternion helpers
def slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    omega = np.arccos(dot)
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) / so) * q0 + (np.sin(t * omega) / so) * q1


def rotation_angle(q0, q1):
    """Angle (rad) of the relative rotation between two unit quaternions."""
    dot = abs(float(np.clip(np.dot(np.asarray(q0), np.asarray(q1)), -1.0, 1.0)))
    return 2.0 * np.arccos(dot)


def view_direction(quat_xyzw):
    """Unit vector along the local +z axis (the viewing / approach direction)."""
    v = np.array(p.rotateVector(list(quat_xyzw), [0, 0, 1]), dtype=np.float64)
    return v / np.linalg.norm(v)
