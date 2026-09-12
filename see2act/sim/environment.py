# Copyright 2023 The Ravens Authors. Licensed under the Apache License, Version 2.0.
# Adapted for See2Act: no gym dependency, scene-controlled resets, observation taken before the robot is loaded.
"""PyBullet environment with a UR5 arm and a suction gripper."""
import os
import pkgutil
import sys
import tempfile

import numpy as np
import pybullet as p

from see2act.sim import cameras
from see2act.sim.pybullet_utils import load_urdf

UR5_URDF_PATH = "ur5/ur5.urdf"
UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"


class Environment:
    """Tabletop environment. Observations are RGB-D images from `agent_cams`."""

    def __init__(self, assets_root, task=None, disp=False, shared_memory=False, hz=480, use_egl=False,
                 agent_cams=None):
        if use_egl and disp:
            raise ValueError("EGL rendering cannot be used with disp=True.")
        self.pix_size = 0.003125
        self.obj_ids = {"fixed": [], "rigid": [], "deformable": []}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = cameras.AgentCameras.CONFIG if agent_cams is None else agent_cams
        self.assets_root = assets_root
        self.task = None
        self.robot_loaded = False
        self.step_hook = None   # optional callable invoked after every simulation step of a robot motion

        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option)

        file_io = p.loadPlugin("fileIOPlugin", physicsClientId=client)
        if file_io < 0:
            raise RuntimeError("pybullet: cannot load FileIO!")
        p.executePluginCommand(file_io, textArgument=assets_root, intArgs=[p.AddFileIOAction], physicsClientId=client)

        self._egl_plugin = None
        if use_egl:
            assert sys.platform == "linux", "EGL rendering is only supported on Linux."
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                self._egl_plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self._egl_plugin = p.loadPlugin("eglRendererPlugin")
            print("EGL rendering enabled.")

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1.0 / hz)

        if disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target)

        if task:
            self.set_task(task)

    # ------------------------------------------------------------------ objects
    @property
    def is_static(self):
        v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.obj_ids["rigid"]]
        return all(np.array(v) < 5e-3)

    def add_object(self, urdf, pose, category="rigid"):
        fixed_base = 1 if category == "fixed" else 0
        obj_id = load_urdf(p, os.path.join(self.assets_root, urdf), pose[0], pose[1], useFixedBase=fixed_base)
        self.obj_ids[category].append(obj_id)
        return obj_id

    def set_task(self, task):
        task.set_assets_root(self.assets_root)
        self.task = task

    def check_connection(self):
        return p.isConnected() == 1

    # ------------------------------------------------------------------ resets
    def _reset_world(self):
        if not self.task:
            raise ValueError("environment task must be set. Call set_task or pass the task arg in the constructor.")
        self.obj_ids = {"fixed": [], "rigid": [], "deformable": []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)  # reproducible contact resolution
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # faster scene load
        load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH), [0, 0, -0.001])
        load_urdf(p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])
        self.robot_loaded = False

    def _load_robot(self):
        self.ur5 = load_urdf(p, os.path.join(self.assets_root, UR5_URDF_PATH))
        self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
        self.ee_tip = 10  # link id of the suction cup
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])
        self.ee.release()
        self.robot_loaded = True

    def reset(self):
        """Sample a new scene from the task. The returned observation is taken BEFORE the robot is loaded,
        so that the recorded images show the scene only; the robot is loaded afterwards."""
        self._reset_world()
        self.task.reset(self)
        obs, _, _, _ = self.step()
        self._load_robot()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return obs

    def reset_to_scene(self, scene):
        """Rebuild a given scene (list of SceneObject) with the robot loaded, ready for execution."""
        self._reset_world()
        self._load_robot()
        self.task.spawn_scene(self, scene)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        obs, _, _, _ = self.step()
        return obs

    # ------------------------------------------------------------------ stepping
    def step(self, action=None):
        """Execute a keyframe action {'pose0': (pos, quat), 'pose1': (pos, quat)} with the pick-place primitive.
        Returns (obs, reward, done, info)."""
        if action is not None:
            timeout = self.task.primitive(self.movej, self.movep, self.ee, **action)
            if timeout:
                return self._get_obs(), 0.0, True, self.info
        while not self.is_static:
            p.stepSimulation()
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()
        info.update(self.info)
        return self._get_obs(), reward, done, info

    def close(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        p.disconnect()

    # ------------------------------------------------------------------ rendering
    def render_camera(self, config):
        """Render (color uint8 HxWx3, depth float HxW in meters, segmentation HxW) for a camera config."""
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        _, _, color, depth, segm = p.getCameraImage(
            width=config["image_size"][1], height=config["image_size"][0], viewMatrix=viewm, projectionMatrix=projm,
            shadow=1, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)[:, :, :3]
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear))
        depth = (2.0 * znear * zfar) / depth
        segm = np.uint8(segm).reshape(depth_image_size)
        return color, depth, segm

    def _get_obs(self):
        obs = {"color": (), "depth": ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs["color"] += (color,)
            obs["depth"] += (depth,)
        return obs

    @property
    def info(self):
        """Object id -> (position, rotation, dimensions)."""
        info = {}
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    # ------------------------------------------------------------------ robot motion
    def movej(self, targj, speed=0.01, max_steps=5000):
        """Move the UR5 to a joint configuration; returns True if the motion did not converge within
        `max_steps` simulation steps (e.g. an unreachable target).

        Ravens uses a 15 s wall-clock timeout here, which also fires when the machine is merely busy and
        turns evaluation outcomes into a function of the system load; a step budget is deterministic.
        """
        for _ in range(int(max_steps)):
            currj = np.array([p.getJointState(self.ur5, i)[0] for i in self.joints])
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(bodyIndex=self.ur5, jointIndices=self.joints, controlMode=p.POSITION_CONTROL,
                                        targetPositions=stepj, positionGains=gains)
            p.stepSimulation()
            if self.step_hook is not None:
                self.step_hook()
        print(f"Warning: movej did not converge within {max_steps} steps. Skipping.")
        return True

    def movep(self, pose, speed=0.01):
        return self.movej(self.solve_ik(pose), speed)

    def solve_ik(self, pose):
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5, endEffectorLinkIndex=self.ee_tip, targetPosition=pose[0], targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17], upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34], restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100, residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints


def action_from_vector(action14):
    """14-d keyframe vector [pick pos (3), pick quat (4), place pos (3), place quat (4)] -> primitive kwargs."""
    a = np.asarray(action14, dtype=np.float64)
    return {"pose0": (a[:3], a[3:7]), "pose1": (a[7:10], a[10:14])}


def vector_from_action(action):
    return np.concatenate([action["pose0"][0], action["pose0"][1], action["pose1"][0], action["pose1"][1]]).astype(np.float64)
