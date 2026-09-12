"""put-within-shelf: a two-level shelf (two stacked side-lying boxes); the red block sits on the top level and
the green bowl on the bottom level, both hidden from the front view."""
import os

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from see2act.sim import utils
from see2act.sim.tasks.base import BinTask, BLOCK_SIZE

SHELF_TEXTURE = os.path.join("container", "legooo.png")


class PutWithinShelf(BinTask):
    box_yaw_range = (np.pi / 2, np.pi / 3 * 2)
    n_distractors = 5

    def __init__(self):
        super().__init__()
        self.box_size = (0.2, 0.2, 0.2)

    # the shelf is one bin template spawned twice, lying on its side (rotated 90 deg about y) and stacked
    def _shelf_quat(self, quat):
        return utils.legacy_quat_mult(utils.eulerXYZ_to_quatXYZW((0, np.pi / 2, 0)), quat)

    def _spawn_box(self, env, pose):
        urdf = self._box_urdf()
        pos, quat = pose
        shelf_quat = self._shelf_quat(quat)
        env.add_object(urdf, (pos, shelf_quat), "fixed")
        upper = env.add_object(urdf, ((pos[0], pos[1], pos[2] + self.box_size[0]), shelf_quat), "fixed")
        texture = p.loadTexture(os.path.join(env.assets_root, SHELF_TEXTURE))
        p.changeVisualShape(upper, linkIndex=4, textureUniqueId=texture)
        os.remove(urdf)

    def _shelf_local_pose(self, box_pose, is_block):
        """Fixed positions of the block (top level) and bowl (bottom level) in the shelf frame."""
        wall = 0.002
        if is_block:
            local_x, local_y = 0.03, -0.025
        else:
            local_x, local_y = 0.0, 0.0
        local_z = 0 - (box_pose[0][-1] - BLOCK_SIZE[2] / 2 - wall)
        local = np.array([local_x, local_y, local_z])
        world_pos = (np.asarray(box_pose[0]) + R.from_quat(box_pose[1]).apply(local)).tolist()
        world_pos = [round(v, 6) for v in world_pos]
        world_pos = tuple(world_pos) if is_block else (world_pos[0], world_pos[1], 0)
        np.random.uniform(0, 2 * np.pi)  # kept for random-stream compatibility with the original task
        return (world_pos, (0, 0, 0, 1))

    def reset(self, env):
        super().reset(env)
        box_pose, other_area = self._random_bin_pose(env)
        self._add_box(env, box_pose)

        block_pose = self._shelf_local_pose(box_pose, is_block=True)
        block_pose = ((block_pose[0][0], block_pose[0][1], block_pose[0][2] + self.box_size[0]), block_pose[1])
        block_id = self._add_block(env, block_pose, "red")
        self.red_block_pose = block_pose

        bowl_pose = self._shelf_local_pose(box_pose, is_block=False)
        self._add_bowl(env, bowl_pose, "green")
        self._set_goal([block_id], [bowl_pose])

        block_colors = [c for c in utils.COLORS if c not in ["red"]]
        bowl_colors = [c for c in utils.COLORS if c not in ["purple", "cyan", "blue", "green"]]
        self._add_distractors(env, self.n_distractors, bounding_box=other_area, block_colors=block_colors,
                              bowl_colors=bowl_colors)

    def oracle_action(self, env):
        """Tilted (60 deg) approach so the suction cup reaches into the shelf levels."""
        box_quat = self._boxes[0].quaternion
        ori_in_box = np.array([0, np.radians(60), 0])
        ori_in_world = R.from_quat(box_quat).apply(ori_in_box)
        quat = np.array(utils.eulerXYZ_to_quatXYZW(ori_in_world))
        pick_pose = (np.array(self.red_block_pose[0]), quat)
        bowl = self.green_bowl_pose[0]
        place_pose = (np.array([bowl[0], bowl[1], bowl[2] + 0.04]), quat)
        return {"pose0": pick_pose, "pose1": place_pose}
