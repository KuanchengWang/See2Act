"""Camera schedule: where the camera is for a given action estimate and diffusion time.

Following the paper (Sec. 3.2), the terminal view C_0 sits at distance d from the target along the
approach axis of the (estimated) keyframe and looks back at it; the view at diffusion time t is the
linear (position) / slerp (orientation) interpolation between C_0 and the global initial view C_T.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from see2act import geometry
from see2act.constants import INIT_VIEWS


@dataclass
class CameraSchedule:
    init_view: str = "front"
    offset: float = 0.6
    jitter: bool = True
    jitter_position: float = 0.05
    jitter_rotation_deg: float = 5.0

    @classmethod
    def from_config(cls, camera_cfg):
        return cls(**{k: camera_cfg[k] for k in cls.__dataclass_fields__ if k in camera_cfg})

    def to_dict(self):
        return asdict(self)


def init_view_pose(name):
    """(position, quaternion xyzw) of a canonical global view."""
    pos, eul = INIT_VIEWS[name]
    return np.array(pos, dtype=np.float64), np.array(p.getQuaternionFromEuler(eul), dtype=np.float64)


def terminal_view(action_pos, action_euler, offset):
    """C_0 for a keyframe action: `offset` metres away along the approach axis, looking back at the target.

    The orientation is the action orientation rotated by 180 degrees about its x axis (the camera looks
    along its +z axis, so this points it at the keyframe).
    """
    q = np.array(p.getQuaternionFromEuler(list(action_euler)), dtype=np.float64)
    pos = np.asarray(action_pos, dtype=np.float64) + offset * geometry.view_direction(q)
    quat = np.array([-q[3], -q[2], q[1], q[0]])  # q * (1, 0, 0, 0)_xyzw up to sign
    return pos, quat


def jitter_pose(pos, quat, t, max_translation, max_rotation_rad, rng=np.random):
    """Random pose perturbation whose magnitude scales with t (no perturbation at the terminal view)."""
    pos = np.asarray(pos, dtype=np.float64)
    d_pos = (rng.rand(3) - 0.5) * 2 * max_translation * t
    eul = np.array(p.getEulerFromQuaternion(list(quat)))
    d_rot = (rng.rand(3) - 0.5) * 2 * max_rotation_rad * t
    rot = R.from_euler("xyz", eul) * R.from_euler("xyz", d_rot)
    return pos + d_pos, np.array(p.getQuaternionFromEuler(rot.as_euler("xyz")), dtype=np.float64)


def view_at(schedule, action_pos, action_euler, t, init_pose, train=False, rng=np.random):
    """Camera pose C_t = Interp(C_T, C_0; t) for t in [0, 1] (t = 1 is the initial view)."""
    t = float(t)
    p1, q1 = init_pose
    p1 = np.asarray(p1, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    p0, q0 = terminal_view(action_pos, action_euler, schedule.offset)
    pos = (1 - t) * p0 + t * p1
    quat = geometry.slerp(q0, q1, t)
    if train and schedule.jitter:
        pos, quat = jitter_pose(pos, quat, t, schedule.jitter_position, np.radians(schedule.jitter_rotation_deg), rng)
    return pos, quat


def pose7(pos, quat):
    return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])


def camera_delta_variance(cam_traj, last=None):
    """Variance of the camera-pose changes along a refinement run (the paper's rollout-selection score).

    cam_traj: (T, 7) poses [x, y, z, qx, qy, qz, qw]. The pose change between consecutive views is the
    6-vector [translation, rotation vector]; the score is the sum of the per-dimension variances. With
    `last=k` only the final k views are considered (stability of the end of the trajectory).
    """
    traj = np.asarray(cam_traj, dtype=np.float64)
    if last is not None:
        traj = traj[-int(last):]
    if traj.shape[0] < 3:
        return 0.0
    d_pos = np.diff(traj[:, :3], axis=0)
    rots = R.from_quat(traj[:, 3:7])
    d_rot = (rots[1:] * rots[:-1].inv()).as_rotvec()
    deltas = np.concatenate([d_pos, d_rot], axis=1)
    return float(np.sum(np.var(deltas, axis=0, ddof=1)))


def action_history_score(history, method, last=None):
    """Uncertainty of a run from its per-step world-frame action estimates (T, 6); lower is more certain."""
    actions = np.asarray(history, dtype=np.float64)
    if last is not None:
        actions = actions[-int(last):]
    if not np.all(np.isfinite(actions)):
        return np.inf
    if actions.shape[0] < 2:
        return 0.0
    var = np.var(actions, axis=0, ddof=1)
    if method == "action_variance":
        return float(np.mean(var))
    if method == "action_entropy":
        return float(np.sum(0.5 * (np.log(2 * np.pi * np.e) + np.log(np.maximum(var, 1e-12)))))
    raise ValueError(method)


def run_score(method, history, cam_traj, final_window=5):
    """Score of one refinement run under a selection criterion (lower = keep)."""
    if method == "none":
        return 0.0
    if method == "camera_variance":
        return camera_delta_variance(cam_traj)
    if method == "final_camera_variance":
        return camera_delta_variance(cam_traj, last=final_window + 1)
    if method == "final_action_variance":
        return action_history_score(history, "action_variance", last=final_window)
    if method in ("action_variance", "action_entropy"):
        return action_history_score(history, method)
    raise ValueError(method)
