"""place-red-in-green: a red block and a green bowl among colored distractors on an open table."""

from see2act.sim.tasks.base import Task, BLOCK_SIZE, BOWL_SIZE


class PlaceRedInGreen(Task):
    def reset(self, env):
        super().reset(env)
        bowl_pose = self.get_random_pose(env, BOWL_SIZE)
        bowl_pose = (bowl_pose[0], (0, 0, 0, 1))
        self._add_bowl(env, bowl_pose, "green")
        block_pose = self.get_random_pose(env, BLOCK_SIZE)
        block_id = self._add_block(env, block_pose, "red")
        self.red_block_pose = block_pose
        self._set_goal([block_id], [bowl_pose])
        self._add_distractors(env, 8)

    def oracle_action(self, env):
        return self.heightmap_oracle_action(env)
