"""bin-picking: the red block starts inside a single open-topped bin lying on its side, invisible from the front."""
import numpy as np

from see2act.sim.tasks.base import BinTask, BOWL_SIZE


class BinPicking(BinTask):
    box_yaw_range = (0.0, np.pi)
    block_local_range = 0.045
    n_distractors = 8

    def reset(self, env):
        super().reset(env)
        box_pose, other_area = self._random_bin_pose(env)
        self._add_box(env, box_pose)
        bowl_pose = self.get_random_constrained_pose(env, BOWL_SIZE, bounding_box=other_area)
        self._add_bowl(env, bowl_pose, "green")
        block_pose = self._random_block_pose_in_bin(box_pose)
        block_id = self._add_block(env, block_pose, "red")
        self.red_block_pose = block_pose
        self._set_goal([block_id], [bowl_pose])
        self._add_distractors(env, self.n_distractors, bounding_box=other_area)
