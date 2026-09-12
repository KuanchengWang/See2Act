# Copyright 2023 The Ravens Authors. Licensed under the Apache License, Version 2.0.
# Adapted for See2Act: scene bookkeeping, scene-controlled resets, height-aware success test.
"""Base task class."""
import collections
import os
import random
import tempfile
import uuid

import cv2
import numpy as np
import pybullet as p

from see2act.sim import cameras, utils
from see2act.sim.grippers import Suction
from see2act.sim.primitives import PickPlace
from see2act.sim.scene import SceneObject, split_name

BLOCK_URDF = "stacking/block.urdf"
BOWL_URDF = "bowl/bowl.urdf"
BOX_TEMPLATE_URDF = "container/container-template.urdf"
BLOCK_SIZE = (0.04, 0.04, 0.04)
BOWL_SIZE = (0.12, 0.12, 0.0)
WORKSPACE_XY = [0.2, 0.8, -0.4, 0.4]  # x_min, x_max, y_min, y_max used to constrain object placement


class Task:
    """A single pick-and-place keyframe task: move the red block into the green bowl.

    Every task records the objects it spawns in `self.scene` (a list of SceneObject in the order
    boxes, blocks, bowls). `spawn_scene()` rebuilds such a scene, which is how the renderer creates
    the views for the policy and how the evaluation re-creates a scene for execution.
    """

    ee = Suction
    max_steps = 1

    def __init__(self):
        self.mode = "train"
        self.primitive = PickPlace()
        self.oracle_cams = cameras.Oracle.CONFIG

        # success test: xy within pos_eps, yaw within rot_eps (if the object has a symmetry), z within z_eps
        self.pos_eps = 0.05
        self.rot_eps = np.deg2rad(15)
        self.z_eps = 0.10

        self.pix_size = 0.003125
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
        self.assets_root = None
        self.box_size = None
        self._clear()

    def _clear(self):
        self.goals = []
        self.progress = 0
        self._rewards = 0
        self._boxes, self._blocks, self._bowls = [], [], []
        self.red_block_pose = None
        self.green_bowl_pose = None

    @property
    def scene(self):
        """Objects of the current episode (boxes, then blocks, then bowls)."""
        return list(self._boxes) + list(self._blocks) + list(self._bowls)

    def set_assets_root(self, assets_root):
        self.assets_root = assets_root

    # ------------------------------------------------------------------ episode reset (to be implemented per task)
    def reset(self, env):
        if not self.assets_root:
            raise ValueError("assets_root must be set for task, call set_assets_root().")
        self._clear()

    # ------------------------------------------------------------------ object helpers
    def _record(self, group, name, pose):
        group.append(SceneObject(name, tuple(float(v) for v in pose[0]), tuple(float(v) for v in pose[1])))

    def _add_block(self, env, pose, color="red"):
        obj_id = env.add_object(BLOCK_URDF, pose)
        if color != "red":  # the block urdf is red by default
            p.changeVisualShape(obj_id, -1, rgbaColor=utils.COLORS[color] + [1])
        self._record(self._blocks, f"{color}_block", pose)
        return obj_id

    def _add_bowl(self, env, pose, color="green"):
        if color == "green":  # the goal bowl is static; the bowl urdf is green by default
            obj_id = env.add_object(BOWL_URDF, pose, "fixed")
            self.green_bowl_pose = pose
        else:
            obj_id = env.add_object(BOWL_URDF, pose)
            p.changeVisualShape(obj_id, -1, rgbaColor=utils.COLORS[color] + [1])
        self._record(self._bowls, f"{color}_bowl", pose)
        return obj_id

    def _box_urdf(self):
        replace = {"DIM": self.box_size, "HALF": np.float32(self.box_size) / 2}
        return self.fill_template(BOX_TEMPLATE_URDF, replace)

    def _spawn_box(self, env, pose):
        """Spawn the task's open-topped bin at `pose` (overridden by the shelf task)."""
        urdf = self._box_urdf()
        env.add_object(urdf, pose, "fixed")
        os.remove(urdf)

    def _add_box(self, env, pose):
        self._spawn_box(env, pose)
        self._record(self._boxes, "brown_box", pose)

    def _set_goal(self, block_ids, bowl_poses):
        blocks = [(bid, (0, None)) for bid in block_ids]
        self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))), bowl_poses, False, True, "pose", None, 1))

    def spawn_scene(self, env, scene):
        """Rebuild `scene` (list of SceneObject) in `env` and set the pick-place goal."""
        self._clear()
        block_ids, bowl_poses = [], []
        for obj in scene:
            color, otype = split_name(obj.name)
            pose = (tuple(obj.position), tuple(obj.quaternion))
            if otype == "box":
                self._add_box(env, pose)
            elif otype == "block":
                obj_id = self._add_block(env, pose, color)
                if color == "red":
                    block_ids.append(obj_id)
                    self.red_block_pose = pose
            elif otype == "bowl":
                self._add_bowl(env, pose, color)
                if color == "green":
                    bowl_poses.append(pose)
            else:
                raise ValueError(f"unknown object type {otype}")
        self._set_goal(block_ids, bowl_poses)

    def _add_distractors(self, env, n, bounding_box=None, block_colors=None, bowl_colors=None):
        """Random blocks/bowls of non-target colors. Colors are drawn with Python's `random` module
        (seed it together with numpy for reproducible scenes)."""
        block_colors = block_colors or [c for c in utils.COLORS if c != "red"]
        bowl_colors = bowl_colors or [c for c in utils.COLORS if c != "green"]
        n_added = 0
        while n_added < n:
            is_block = np.random.rand() > 0.5
            size = BLOCK_SIZE if is_block else BOWL_SIZE
            colors = block_colors if is_block else bowl_colors
            if bounding_box is None:
                pose = self.get_random_pose(env, size)
                color = colors[n_added % len(colors)]
            else:
                pose = self.get_random_constrained_pose(env, size, bounding_box=bounding_box)
                color = colors[random.randint(0, len(colors) - 1)]
            if not pose[0] or not pose[1]:
                continue
            if is_block:
                self._add_block(env, pose, color)
            else:
                self._add_bowl(env, pose, color)
            n_added += 1

    # ------------------------------------------------------------------ oracle
    def oracle(self, env):
        OracleAgent = collections.namedtuple("OracleAgent", ["act"])
        return OracleAgent(lambda obs, info: self.oracle_action(env))

    def oracle_action(self, env):
        """Scripted expert: pick the red block at its center, place it at the green bowl's center."""
        pick_pose = (np.array(self.red_block_pose[0]), np.array([0, 0, 0, 1]))
        place_pose = (np.array(self.green_bowl_pose[0]), np.array([0, 0, 0, 1]))
        return {"pose0": pick_pose, "pose1": place_pose}

    def heightmap_oracle_action(self, env):
        """Ravens' original expert: pick point sampled on the object's top-down mask."""
        _, hmap, obj_mask = self.get_true_image(env)
        objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

        if not replace:
            matches = matches.copy()
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                pose = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                for j in targets_i:
                    if self.is_match(pose, targs[j], symmetry):
                        matches[i, :] = 0
                        matches[:, j] = 0

        nn_dists, nn_targets = [], []
        for i in range(len(objs)):
            object_id, (symmetry, _) = objs[i]
            xyz, _ = p.getBasePositionAndOrientation(object_id)
            targets_i = np.argwhere(matches[i, :]).reshape(-1)
            if len(targets_i) > 0:
                targets_xyz = np.float32([targs[j][0] for j in targets_i])
                dists = np.linalg.norm(targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
                nn = np.argmin(dists)
                nn_dists.append(dists[nn])
                nn_targets.append(targets_i[nn])
            else:
                nn_dists.append(0)
                nn_targets.append(-1)
        order = np.argsort(nn_dists)[::-1]
        order = [i for i in order if nn_dists[i] > 0]

        pick_mask = None
        for pick_i in order:
            pick_mask = np.uint8(obj_mask == objs[pick_i][0])
            if np.sum(pick_mask) > 0:
                break
        if pick_mask is None or np.sum(pick_mask) == 0:
            self.goals = []
            print("Object for pick is not visible. Skipping demonstration.")
            return None

        pick_prob = np.float32(pick_mask)
        pick_pix = utils.sample_distribution(pick_prob)
        pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
        pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

        targ_pose = targs[nn_targets[pick_i]]
        obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
        obj_pose = (obj_pose[0], obj_quat)
        world_to_pick = utils.invert(pick_pose)
        obj_to_pick = utils.multiply(world_to_pick, obj_pose)
        pick_to_obj = utils.invert(obj_to_pick)
        place_pose = utils.multiply(targ_pose, pick_to_obj)
        place_pose = (np.asarray(place_pose[0]), np.asarray((0, 0, 0, 1)))
        return {"pose0": pick_pose, "pose1": place_pose}

    # ------------------------------------------------------------------ reward / success
    def reward(self):
        """Delta reward: 1 when the red block reaches the green bowl."""
        reward, info = 0, {}
        if self.goals:
            objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]
            assert metric == "pose"
            step_reward = 0
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                pose = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                for j in targets_i:
                    if self.is_match(pose, targs[j], symmetry):
                        step_reward += max_reward / len(objs)
                        break
            reward = self.progress + step_reward - self._rewards
            self._rewards = self.progress + step_reward
            if np.abs(max_reward - step_reward) < 0.01:
                self.progress += max_reward
                self.goals.pop(0)
        return reward, info

    def done(self):
        return (len(self.goals) == 0) or (self._rewards > 0.99)

    def is_match(self, pose0, pose1, symmetry):
        """Pose match test. Unlike the original Ravens test it also checks the height, so that a block
        resting on a shelf above the target bowl does not count as placed."""
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)
        diff_z = abs(float(pose0[0][2]) - float(pose1[0][2]))
        diff_rot = 0
        if symmetry > 0:
            rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
            rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
            diff_rot = np.abs(rot0 - rot1) % symmetry
            if diff_rot > (symmetry / 2):
                diff_rot = symmetry - diff_rot
        return (dist_pos < self.pos_eps) and (diff_rot < self.rot_eps) and (diff_z < self.z_eps)

    # ------------------------------------------------------------------ scene sampling helpers
    def get_true_image(self, env):
        """Orthographic top-down color map, heightmap and segmentation mask (oracle camera)."""
        color, depth, segm = env.render_camera(self.oracle_cams[0])
        color = np.concatenate((color, segm[..., None]), axis=2)
        hmaps, cmaps = utils.reconstruct_heightmaps([color], [depth], self.oracle_cams, self.bounds, self.pix_size)
        cmap = np.uint8(cmaps)[0, ..., :3]
        hmap = np.float32(hmaps)[0, ...]
        mask = np.int32(cmaps)[0, ..., 3:].squeeze()
        return cmap, hmap, mask

    def _free_space(self, env, obj_size):
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))
        _, hmap, obj_mask = self.get_true_image(env)
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        return free, hmap

    def get_random_pose(self, env, obj_size):
        """Random collision-free pose within the workspace bounds."""
        free, hmap = self._free_space(env, obj_size)
        if np.sum(free) == 0:
            return None, None
        pix = utils.sample_distribution(np.float32(free))
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def get_random_constrained_pose(self, env, obj_size, bounding_box=(0.4, 0.6, -0.1, 0.1)):
        """Random collision-free pose additionally constrained to an xy bounding box."""
        free, hmap = self._free_space(env, obj_size)
        if np.sum(free) == 0:
            return None, None
        attempt = 0
        while attempt < 10000:
            pix = utils.sample_distribution(np.float32(free))
            pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
            if bounding_box[0] <= pos[0] <= bounding_box[1] and bounding_box[2] <= pos[1] <= bounding_box[3]:
                break
            attempt += 1
        if attempt >= 10000:
            return None, None
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def fill_template(self, template, replace):
        """Instantiate a URDF template into a process-unique temporary file."""
        with open(os.path.join(self.assets_root, template), "r") as f:
            fdata = f.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f"{field}{i}", str(replace[field][i]))
        fname = os.path.join(tempfile.gettempdir(),
                             f"{os.path.split(template)[-1]}.{os.getpid()}.{uuid.uuid4().hex[:12]}")
        with open(fname, "w") as f:
            f.write(fdata)
        return fname


class BinTask(Task):
    """Shared code for tasks whose red block starts inside an open-topped bin lying on its side."""

    box_yaw_range = (0.0, np.pi)      # magnitude of the bin's yaw; the sign is drawn at random
    block_local_range = 0.045         # block position inside the bin: uniform in +/- this (m)
    n_distractors = 8

    def __init__(self):
        super().__init__()
        self.box_size = (0.17, 0.17, 0.17)

    def _random_bin_pose(self, env):
        """Random pose for a bin. Returns (pose, xy bounding box for the other objects)."""
        r_z = np.random.uniform(*self.box_yaw_range)
        sign = np.random.choice([-1, 1])
        quat = utils.eulerXYZ_to_quatXYZW((0, 0, sign * r_z))
        if sign < 0:
            bin_area, other_area = [0.2, 0.8, 0, 0.4], WORKSPACE_XY
        else:
            bin_area, other_area = [0.2, 0.8, -0.4, 0], WORKSPACE_XY
        if self.bins_anywhere:
            bin_area = WORKSPACE_XY
        pos = self.get_random_constrained_pose(env, self.box_size, bounding_box=bin_area)[0]
        return (pos, quat), other_area

    bins_anywhere = False

    def _random_block_pose_in_bin(self, box_pose):
        """Random upright block pose inside the (side-lying) bin, in the bin's frame."""
        from scipy.spatial.transform import Rotation as R
        block_height = BLOCK_SIZE[2]
        wall = 0.002
        local_x = np.random.uniform(-self.block_local_range, self.block_local_range)
        local_y = np.random.uniform(-self.block_local_range, self.block_local_range)
        local_z = 0 - (box_pose[0][-1] - block_height / 2 - wall)
        local = np.array([local_x, local_y, local_z])
        world_pos = (np.asarray(box_pose[0]) + R.from_quat(box_pose[1]).apply(local)).tolist()
        world_pos = tuple(round(v, 6) for v in world_pos)
        yaw = np.random.uniform(0, 2 * np.pi)
        local_quat = utils.eulerXYZ_to_quatXYZW((0, 0, yaw))
        world_quat = utils.legacy_quat_mult(box_pose[1], local_quat)
        return (world_pos, world_quat)
