import math

from gym.utils import EzPickle
import numpy as np

from milbench.base_env import BaseEnv, ez_init
import milbench.entities as en


class MoveToCornerEnv(BaseEnv, EzPickle):
    @ez_init()
    def __init__(self,
                 rand_shape_colour=False,
                 rand_shape_type=False,
                 rand_shape_pose=False,
                 rand_robot_pose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rand_shape_colour = rand_shape_colour
        self.rand_shape_type = rand_shape_type
        self.rand_shape_pose = rand_shape_pose
        self.rand_robot_pose = rand_robot_pose

    @classmethod
    def make_name(cls, suffix=None):
        base_name = 'MoveToCorner'
        return base_name + (suffix or '') + '-v0'

    def on_reset(self):
        # make the robot
        robot_pos = np.asarray((0.0, -0.0))
        robot_angle = math.pi / 9
        if self.rand_robot_pose:
            robot_jitter = np.clip(0.1 * self.rng.randn(2), 0.1, 0.1)
            robot_pos = robot_pos + robot_jitter
            robot_angle = self.rng.uniform(-math.pi, math.pi)
        robot = self._make_robot(robot_pos, robot_angle)
        self.add_entities([robot])

        shape_pos = np.asarray((0.6, -0.6))
        shape_angle = 0.13 * math.pi
        shape_colour = 'red'
        shape_type = en.ShapeType.SQUARE
        if self.rand_shape_pose:
            shape_jitter = np.clip(0.1 * self.rng.randn(2), 0.1, 0.1)
            shape_pos = shape_pos + shape_jitter
            shape_angle = self.rng.uniform(-math.pi, math.pi)
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

    def score_on_end_of_traj(self):
        robot_pos = np.asarray(self.__shape_ref.shape_body.position)
        # should be in top left corner
        shortfall_x = max(0, robot_pos[0] + 0.5)
        shortfall_y = max(0, 0.5 - robot_pos[1])
        dist = np.linalg.norm((shortfall_x, shortfall_y))
        # max reward is for being in 0.5x0.5 square in top left corner. Reward
        # decreases linearly outside of that, until we're more than 0.5 units
        # outside of that 0.5x0.5 square and & we get nothing.
        reward = max(1 - dist / 0.5, 0)
        assert reward <= 1
        return reward
