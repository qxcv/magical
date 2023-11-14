import math

from gym.utils import EzPickle
import numpy as np

from magical.base_env import BaseEnv, ez_init
import magical.entities as en
import magical.geom as geom
from typing import Any


class PushToColourRegionEnv(BaseEnv, EzPickle):
    """Push all blocks to region of a particular colour, regardless of their
    colour.
    
    Randomizes everything by default; there is no demo variant."""
    @ez_init()
    def __init__(self, target_colour: en.ShapeColour, easy_visuals: bool=False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.easy_visuals = easy_visuals
        self.target_colour = target_colour

    def on_reset(self) -> None:
        # make the robot
        robot_pos = np.asarray((-0.5, 0.1))
        robot_angle = -math.pi * 1.2
        # robot pose is randomized later
        robot = self._make_robot(robot_pos, robot_angle)

        if self.easy_visuals:
            possible_colour = np.delete(en.SHAPE_COLOURS, np.where(en.SHAPE_COLOURS == 'yellow'))
            possible_shape = np.delete(en.SHAPE_TYPES, np.where(en.SHAPE_TYPES == 'pentagon'))
        else:
            possible_colour = en.SHAPE_COLOURS
            possible_shape = en.SHAPE_TYPES

        # set up the various regions
        assert len(possible_colour) <= 4, f"too many colours {possible_colour=}"
        target_colour = self.target_colour
        assert target_colour in set(possible_colour), f"{target_colour=} not {possible_colour=}"
        # put things in the different quadrants
        # (coordates are [-1, 1] on x and y axies)
        region_positions = [
            # top left (x, y, h, w)
            (-0.95, 0.95, 0.7, 0.7),
            # top right
            (0.25, 0.95, 0.7, 0.7),
            # bottom left
            (-0.95, -0.25, 0.7, 0.7),
            # bottom right
            (0.25, -0.25, 0.7, 0.7),
        ]
        # shuffle the colours and positions
        region_positions = self.rng.permutation(region_positions)
        possible_colour = self.rng.permutation(possible_colour)
        # in the easy variant, we throw out one colour
        region_positions = region_positions[:len(possible_colour)]

        assert len(possible_colour) == len(region_positions), f"{possible_colour=} {region_positions=}"
        assert len(possible_colour) >= 1, f"{possible_colour=}"

        for region_colour, (reg_x, reg_y, reg_h, reg_w) in zip(possible_colour, region_positions):
            eps = 0.045
            reg_x += self.rng.uniform(-eps, eps)
            reg_y += self.rng.uniform(-eps, eps)
            reg_h += self.rng.uniform(-eps, eps)
            reg_w += self.rng.uniform(-eps, eps)
            sensor = en.GoalRegion(reg_x, reg_y, reg_h, reg_w, region_colour)
            self.add_entities([sensor])
            if region_colour == target_colour:
                self.__sensor_ref = sensor

        # make 2-5 blocks of random shape and colour
        block_count = self.rng.randint(2, 6)
        shape_types_np = np.asarray(possible_shape, dtype='object')
        block_shapes = [
            self.rng.choice(shape_types_np) for _ in range(block_count)
        ]
        block_colours = [
            self.rng.choice(possible_colour) for _ in range(block_count)
        ]
        base_poses = [(0, 0, 0)] * block_count

        self.__blocks = [
            self._make_shape(shape_type=block_shape,
                             colour_name=block_colour,
                             init_pos=(block_x, block_y),
                             init_angle=block_angle,
                             easy_visuals=self.easy_visuals)
            for block_shape, block_colour, (block_x, block_y, block_angle) in zip(
                block_shapes, block_colours, base_poses)
        ]
        self.add_entities(self.__blocks)

        # add this last so it shows up on top, but before layout randomisation,
        # since it needs to be added to the space before randomising
        self.add_entities([robot])

        geom.pm_randomise_all_poses(self._space,
                                    (robot, *self.__blocks),
                                    self.ARENA_BOUNDS_LRBT,
                                    rng=self.rng,
                                    rand_pos=True,
                                    rand_rot=True,
                                    rel_pos_linf_limits=None,
                                    rel_rot_limits=None)

        # set up index for lookups
        self.__ent_index = en.EntityIndex(self.__blocks)

    def score_on_end_of_traj(self):
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            com_overlap=True, ent_index=self.__ent_index)
        block_set = set(self.__blocks)
        n_overlap_blocks = len(block_set & overlap_ents)
        if len(block_set) == 0:
            return 1.0
        return n_overlap_blocks / len(block_set)
