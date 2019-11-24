import math

from milbench.base_env import BaseEnv
import milbench.entities as en


class MoveToCornerEnv(BaseEnv):
    @classmethod
    def make_name(cls, suffix=None):
        base_name = 'MoveToCorner'
        return base_name + (suffix or '') + '-v0'

    def _reinit_entities(self):
        robot = en.Robot(radius=self.ROBOT_RAD, init_pos=(0.1, -0.1),
                         init_angle=math.pi / 9, mass=1.0)
        square = en.Shape(shape_type=en.ShapeType.SQUARE,
                          colour_name='red',
                          shape_size=self.SHAPE_RAD,
                          init_pos=(0.4, -0.4),
                          init_angle=0.13 * math.pi)
        # other examples:
        # square2 = en.Shape(shape_type=en.ShapeType.SQUARE,
        #                    colour_name='red',
        #                    shape_size=shape_rad,
        #                    init_pos=(0.25, -0.65),
        #                    init_angle=0.23 * math.pi)
        # circle = en.Shape(shape_type=en.ShapeType.CIRCLE,
        #                   colour_name='yellow',
        #                   shape_size=shape_rad,
        #                   init_pos=(-0.7, -0.5),
        #                   init_angle=-0.5 * math.pi)
        # triangle = en.Shape(shape_type=en.ShapeType.HEXAGON,
        #                     colour_name='green',
        #                     shape_size=shape_rad,
        #                     init_pos=(-0.5, 0.35),
        #                     init_angle=0.05 * math.pi)
        # pentagon = en.Shape(shape_type=en.ShapeType.PENTAGON,
        #                     colour_name='blue',
        #                     shape_size=shape_rad,
        #                     init_pos=(0.4, 0.35),
        #                     init_angle=0.8 * math.pi)
        return robot, [square]
