import math

import numpy as np

from milbench.base_env import BaseEnv
import milbench.entities as en


class MoveToCornerEnv(BaseEnv):
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

        shape_pos = np.asarray((0.6, -0.6))
        shape_angle = 0.13 * math.pi
        shape_colour = 'red'
        shape_type = en.ShapeType.SQUARE
        if self.rand_shape_pose:
            shape_jitter = np.clip(0.1 * self.rng.randn(2), 0.1, 0.1)
            shape_pos = shape_pos + shape_jitter
            shape_angle = self.rng.uniform(-math.pi, math.pi)
        if self.rand_shape_colour:
            shape_colour = self.rng.choice(en.SHAPE_COLOURS)
        if self.rand_shape_type:
            shape_type = self.rng.choice(en.SHAPE_TYPES)

        shape = self._make_shape(shape_type=shape_type,
                                 colour_name=shape_colour,
                                 init_pos=shape_pos,
                                 init_angle=shape_angle)
        self.__shape_ref = shape

        return robot, [shape]

    def score_on_end_of_traj(self):
        # TODO: do this goal check with invisible sensor area, not manual
        # position accesses
        robot_pos = np.asarray(self.__shape_ref.shape_body.position)
        # should be in top right corner
        shortfall_x = max(0, robot_pos[0] + 0.7)
        shortfall_y = max(0, 0.7 - robot_pos[1])
        dist = np.linalg.norm((shortfall_x, shortfall_y))
        in_corner = dist <= 0
        return 1.0 if in_corner else 0.0
