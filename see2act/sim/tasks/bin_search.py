"""bin-search: three open-topped bins; the red block is hidden in one of them at random (the paper's
`bin-search` / `box-search` task)."""
import numpy as np

from see2act.sim.tasks.base import BinTask, BOWL_SIZE


class BinSearch(BinTask):
    box_yaw_range = (np.pi / 4, 3 * np.pi / 4)
    bins_anywhere = True
    block_local_range = 0.035
    n_distractors = 5
    n_bins = 3

    def reset(self, env):
        super().reset(env)
        bin_poses = []
        for _ in range(self.n_bins):
            box_pose, other_area = self._random_bin_pose(env)
            self._add_box(env, box_pose)
            bin_poses.append(box_pose)
        which_bin = np.random.randint(0, self.n_bins)
        bowl_pose = self.get_random_constrained_pose(env, BOWL_SIZE, bounding_box=other_area)
        self._add_bowl(env, bowl_pose, "green")
        block_pose = self._random_block_pose_in_bin(bin_poses[which_bin])
        block_id = self._add_block(env, block_pose, "red")
        self.red_block_pose = block_pose
        self._set_goal([block_id], [bowl_pose])
        self._add_distractors(env, self.n_distractors, bounding_box=other_area)
