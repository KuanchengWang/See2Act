"""Multi-process PyBullet renderer: synthesizes the view the policy asks for, from the scene description.

Each worker keeps one PyBullet client with a pool of pre-loaded bodies (ScenePool). Rendering a view means
re-positioning those bodies to the requested scene, letting the rigid ones settle, and taking one image.
This avoids reloading URDFs for every image and the GPU-memory leak of PyBullet's EGL renderer on
resetSimulation.
"""
import multiprocessing as mp
import os
import signal
import time

import numpy as np
import pybullet as p

from see2act.sim import ASSETS_ROOT, cameras, tasks, utils
from see2act.sim.environment import Environment, PLANE_URDF_PATH, UR5_WORKSPACE_URDF_PATH
from see2act.sim.pybullet_utils import load_urdf
from see2act.sim.scene import decode_scene, split_name

PARK_POSE = ((0.0, 0.0, -5.0), (0.0, 0.0, 0.0, 1.0))
_env = None
_pool = None


class ScenePool:
    """Pre-loaded bodies that can be arranged into any scene of the task by re-positioning."""

    def __init__(self, env, task, n_blocks=16, n_bowls=16, n_box_slots=3):
        self.env = env
        self.task = task
        self.shelf = isinstance(task, tasks.PutWithinShelf)
        self.box_size = task.box_size
        env.obj_ids = {"fixed": [], "rigid": [], "deformable": []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)  # reproducible contact resolution
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        load_urdf(p, os.path.join(env.assets_root, PLANE_URDF_PATH), [0, 0, -0.001])
        load_urdf(p, os.path.join(env.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])
        self.blocks = [env.add_object("stacking/block.urdf", PARK_POSE) for _ in range(n_blocks)]
        self.bowls = [env.add_object("bowl/bowl.urdf", PARK_POSE) for _ in range(n_bowls)]
        self.green_bowl = env.add_object("bowl/bowl.urdf", PARK_POSE, "fixed")
        self.box_slots = []
        if self.box_size is not None:
            for _ in range(n_box_slots):
                urdf = task._box_urdf()
                ids = [env.add_object(urdf, PARK_POSE, "fixed")]
                if self.shelf:
                    # The upper shelf box carries a texture on its link 4. The EGL renderer ignores
                    # changeVisualShape(texture) after loading, so the texture is baked into the URDF material.
                    tex_path = os.path.join(env.assets_root, "container", "legooo.png")
                    txt = open(urdf).read()
                    parts = txt.split('<material name="brown">')
                    assert len(parts) >= 6, "unexpected container template"
                    parts[5] = parts[5].split("</material>", 1)[1]
                    txt = ('<material name="brown">'.join(parts[:5])
                           + f'<material name="tex"><texture filename="{tex_path}"/></material>' + parts[5])
                    urdf_tex = urdf + ".tex.urdf"
                    open(urdf_tex, "w").write(txt)
                    ids.append(env.add_object(urdf_tex, PARK_POSE, "fixed"))
                    os.remove(urdf_tex)
                os.remove(urdf)
                self.box_slots.append(ids)
        self.mass = {oid: p.getDynamicsInfo(oid, -1)[0] for oid in self.blocks + self.bowls}
        # resetBasePositionAndOrientation addresses the inertial (COM) frame while loadURDF places the link
        # frame (the bowl's COM is offset); convert link-frame poses to COM poses.
        all_ids = self.blocks + self.bowls + [self.green_bowl] + [oid for slot in self.box_slots for oid in slot]
        self.inertial = {oid: tuple(p.getDynamicsInfo(oid, -1)[3:5]) for oid in all_ids}
        for oid in self.blocks + self.bowls:
            p.changeDynamics(oid, -1, mass=0.0)  # parked = static
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.active_rigid = []

    def _set_link_pose(self, oid, pos, quat):
        ipos, iorn = self.inertial[oid]
        com_pos, com_orn = p.multiplyTransforms(list(pos), list(quat), list(ipos), list(iorn))
        p.resetBasePositionAndOrientation(oid, com_pos, com_orn)

    def _put(self, oid, pos, quat, color=None, rigid=True):
        self._set_link_pose(oid, pos, quat)
        if color is not None:
            p.changeVisualShape(oid, -1, rgbaColor=list(utils.COLORS[color]) + [1])
        if rigid:
            p.changeDynamics(oid, -1, mass=self.mass[oid])
            p.resetBaseVelocity(oid, [0, 0, 0], [0, 0, 0])
            self.active_rigid.append(oid)

    def _park(self, oid, rigid=True):
        p.resetBasePositionAndOrientation(oid, PARK_POSE[0], PARK_POSE[1])
        if rigid:
            p.changeDynamics(oid, -1, mass=0.0)
            p.resetBaseVelocity(oid, [0, 0, 0], [0, 0, 0])

    def place(self, scene, settle=True, max_steps=480):
        """Arrange the pool into `scene` (list of SceneObject); rigid objects settle under gravity."""
        self.active_rigid = []
        bi = wi = si = 0
        green_used = False
        for obj in scene:
            color, otype = split_name(obj.name)
            pos, quat = obj.position, obj.quaternion
            if otype == "block":
                self._put(self.blocks[bi], pos, quat, color)
                bi += 1
            elif otype == "bowl":
                if color == "green" and not green_used:
                    self._put(self.green_bowl, pos, quat, color, rigid=False)
                    green_used = True
                else:
                    self._put(self.bowls[wi], pos, quat, color)
                    wi += 1
            elif otype == "box":
                slot = self.box_slots[si]
                si += 1
                if self.shelf:
                    q = self.task._shelf_quat(quat)
                    self._set_link_pose(slot[0], pos, q)
                    self._set_link_pose(slot[1], (pos[0], pos[1], pos[2] + self.box_size[0]), q)
                else:
                    self._set_link_pose(slot[0], pos, quat)
            else:
                raise ValueError(f"unknown object type {otype}")
        for oid in self.blocks[bi:] + self.bowls[wi:]:
            self._park(oid)
        if not green_used:
            self._park(self.green_bowl, rigid=False)
        for slot in self.box_slots[si:]:
            for oid in slot:
                self._park(oid, rigid=False)
        if settle:
            for _ in range(max_steps):
                p.stepSimulation()
                if all(np.linalg.norm(p.getBaseVelocity(oid)[0]) < 5e-3 for oid in self.active_rigid):
                    break


def _worker_init(assets_root, task_name, use_egl, fast):
    global _env, _pool
    task = tasks.make(task_name)
    _env = Environment(assets_root=assets_root, task=task, disp=False, shared_memory=False, hz=480, use_egl=use_egl)
    _pool = ScenePool(_env, task) if fast else None


def normalize_image(color):
    """uint8 HxWx3 -> float32 (3, W, H) scaled to [0, 1] by its own min/max (the policy's input format)."""
    img = np.uint8(color).astype(np.float32)
    span = float(np.max(img) - np.min(img))
    if span > 0:
        img = (img - np.min(img)) / span
    else:  # uniform image (camera inside an object): avoid 0/0
        img = np.zeros_like(img)
    return img.transpose(2, 1, 0)


def _render(camera_pose7, scene_encoded, raw=False):
    """Worker entry point: one image of one scene from one camera pose."""
    scene = decode_scene(scene_encoded)
    config = cameras.camera_config(camera_pose7[:3], camera_pose7[3:7])
    if _pool is not None:
        _pool.place(scene)
    else:
        _env._reset_world()
        _env.task.spawn_scene(_env, scene)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        _env.step()
    color, depth, segm = _env.render_camera(config)
    if raw:
        return color, depth, segm
    return normalize_image(color)


class Renderer:
    """Pool of rendering workers. `render(scenes, poses)` returns a float32 array (N, 3, W, H)."""

    def __init__(self, task_name, num_workers=4, assets_root=ASSETS_ROOT, use_egl=None, fast=True):
        self.task_name = task_name
        self.num_workers = int(num_workers)
        self.assets_root = assets_root
        self.use_egl = (os.environ.get("SEE2ACT_USE_EGL", "1") == "1") if use_egl is None else bool(use_egl)
        self.fast = fast
        self.pool = None
        self.worker_pids = []

    def start(self):
        if self.pool is None:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(self.num_workers, initializer=_worker_init,
                                 initargs=(self.assets_root, self.task_name, self.use_egl, self.fast))
            self.worker_pids = [w.pid for w in self.pool._pool]
        return self

    def stop(self):
        """Kill the worker processes. multiprocessing.Pool would respawn killed workers (each opening a new
        PyBullet/EGL context) and its finalizer can hang joining them, so the pool is put into the TERMINATE
        state first and its finalizer is cancelled."""
        if self.pool is None:
            return
        import multiprocessing.pool as mpp
        self.pool._state = mpp.TERMINATE
        self.pool._worker_handler._state = mpp.TERMINATE
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in self.worker_pids:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            time.sleep(0.2)
        try:
            self.pool._terminate.cancel()
        except Exception:  # pragma: no cover
            pass
        self.pool = None
        self.worker_pids = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def render(self, scenes, camera_poses):
        """scenes: (N, 8k) encoded scenes (or one scene shared by all poses); camera_poses: (N, 7)."""
        camera_poses = np.asarray(camera_poses, dtype=np.float64).reshape(-1, 7)
        scenes = np.asarray(scenes, dtype=np.float64)
        if scenes.ndim == 1:
            scenes = np.repeat(scenes[None], len(camera_poses), axis=0)
        assert len(scenes) == len(camera_poses), (scenes.shape, camera_poses.shape)
        results = self.pool.starmap(_render, zip(camera_poses, scenes))
        return np.stack(results, 0)

    def render_raw(self, scene, camera_pose7):
        """(color uint8 HxWx3, depth, segmentation) of one view, for visualization."""
        return self.pool.apply(_render, (np.asarray(camera_pose7, dtype=np.float64), np.asarray(scene), True))
