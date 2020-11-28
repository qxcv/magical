import math
import warnings

from gym.utils import EzPickle
import numpy as np

from magical import geom
from magical.base_env import BaseEnv, ez_init
import magical.entities as en


class MoveToCornerEnv(BaseEnv, EzPickle):
    @ez_init()
    def __init__(self,
                 rand_shape_colour=False,
                 rand_shape_type=False,
                 rand_poses=False,
                 debug_reward=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rand_shape_colour = rand_shape_colour
        self.rand_shape_type = rand_shape_type
        self.rand_poses = rand_poses
        self.debug_reward = debug_reward
        if self.debug_reward:
            warnings.warn(
                "DEBUG REWARD ENABLED IN MOVE-TO-CORNER ENV! This reward is "
                "ONLY intended for training RL algorithms during debugging, "
                "so don't forget to disable it when benchmarking IL")

    def on_reset(self):
        # make the robot
        robot_pos = np.asarray((0.4, -0.0))
        robot_angle = 0.55 * math.pi
        robot = self._make_robot(robot_pos, robot_angle)
        self.add_entities([robot])

        shape_pos = np.asarray((0.1, -0.65))
        shape_angle = 0.13 * math.pi
        shape_colour = 'red'
        shape_type = en.ShapeType.SQUARE
        if self.rand_shape_colour:
            shape_colour = self.rng.choice(
                np.asarray(en.SHAPE_COLOURS, dtype='object'))
        if self.rand_shape_type:
            shape_type = self.rng.choice(
                np.asarray(en.SHAPE_TYPES, dtype='object'))

        shape = self._make_shape(shape_type=shape_type,
                                 colour_name=shape_colour,
                                 init_pos=shape_pos,
                                 init_angle=shape_angle)
        self.add_entities([shape])
        self.__shape_ref = shape

        if self.rand_poses:
            geom.pm_randomise_all_poses(
                self._space, (self._robot, self.__shape_ref),
                self.ARENA_BOUNDS_LRBT,
                self.rng,
                rand_pos=True,
                rand_rot=True,
                rel_pos_linf_limits=self.JITTER_POS_BOUND,
                rel_rot_limits=self.JITTER_ROT_BOUND)

    def score_on_end_of_traj(self):
        robot_pos = np.asarray(self.__shape_ref.shape_body.position)
        # target is top left
        dist = np.linalg.norm(np.asarray([-1.0, 1.0]) - robot_pos)
        succeed_dist = np.sqrt(2) / 2
        furthest_dist = np.sqrt(2)
        drange = (furthest_dist - succeed_dist)
        score = min(1.0, max(0.0, furthest_dist - dist) / drange)
        assert 0 <= score <= 1
        return score

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        if self.debug_reward:
            # dense reward for training RL
            rew = self.debug_shaped_reward()
        return obs, rew, done, info

    def debug_shaped_reward(self):
        """Compute a heavily shaped reward. This should be sufficient to do RL
        on. It's quite helpful for tuning RL algorithms in new GAIL
        implementations."""
        shape_pos = np.asarray(self.__shape_ref.shape_body.position)
        # shape_pos[0] is meant to be 0, shape_pos[1] is meant to be 1
        target_shape_pos = np.array((0, 1))
        shape_to_corner_dist = np.linalg.norm(shape_pos - target_shape_pos)
        # encourage the robot to get close to the shape, and the shape to get
        # close to the goal
        robot_pos = np.asarray(self._robot.robot_body.position)
        robot_to_shape_dist = np.linalg.norm(robot_pos - shape_pos)
        shaping = -shape_to_corner_dist / 5 \
            - max(robot_to_shape_dist, 0.2) / 20
        return shaping + self.score_on_end_of_traj()
