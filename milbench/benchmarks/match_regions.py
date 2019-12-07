import math

import numpy as np

from milbench.base_env import BaseEnv
import milbench.entities as en
import milbench.geom as geom

# we don't use en.SHAPE_COLOURS because some of the constants in this module
# depend on the number of colours remaining constant
ALL_COLOURS = np.array([
    en.ShapeColour.RED,
    en.ShapeColour.GREEN,
    en.ShapeColour.BLUE,
    en.ShapeColour.YELLOW,
],
                       dtype='object')


class MatchRegionsEnv(BaseEnv):
    """Need to push blocks of a certain colour to the corresponding coloured
    region. Aim is for the robot to generalise _that_ rule instead of
    generalising others (e.g. "always move squares to this position" or
    something)."""
    def __init__(
            self,
            rand_target_colour=False,
            # shape_randomisation=ShapeRandLevel.NONE,
            rand_shape_type=False,
            rand_shape_count=False,
            rand_layout=False,
            **kwargs):
        super().__init__(**kwargs)
        self.rand_target_colour = rand_target_colour
        # self.shape_randomisation = shape_randomisation
        self.rand_shape_type = rand_shape_type
        self.rand_shape_count = rand_shape_count
        self.rand_layout = rand_layout
        if self.rand_shape_count:
            assert self.rand_layout, \
                "if shape count is randomised then layout must also be " \
                "randomised"
            assert self.rand_shape_type, \
                "if shape count is randomised then shape type must also be " \
                "randomised"

    @classmethod
    def make_name(cls, suffix=None):
        base_name = 'MatchRegions'
        return base_name + (suffix or '') + '-v0'

    def on_reset(self):
        # make the robot
        robot_pos = np.asarray((-0.5, 0.1))
        robot_angle = -math.pi * 1.2
        # if necessary, robot pose is randomised below
        robot = self._make_robot(robot_pos, robot_angle)

        # set up target colour/region/pose
        if self.rand_target_colour:
            target_colour = self.rng.choice(ALL_COLOURS)
        else:
            target_colour = en.ShapeColour.GREEN
        distractor_colours = [c for c in ALL_COLOURS if c != target_colour]
        if self.rand_layout:
            # TODO: account for arena shape and shape radius
            target_h = self.rng.uniform(0.5, 0.9)
            target_w = self.rng.uniform(0.5, 0.9)
            target_x = self.rng.uniform(-1, 1 - target_w)
            target_y = self.rng.uniform(1, -1 + target_h)
        else:
            target_h = 0.7
            target_w = 0.6
            target_x = 0.1
            target_y = 0.7
        sensor = en.GoalRegion(target_x, target_y, target_h, target_w,
                               target_colour)
        self.add_entities([sensor])
        self.__sensor_ref = sensor

        # set up spec for remaining blocks
        default_target_types = [
            en.ShapeType.HEXAGON,
            en.ShapeType.SQUARE,
        ]
        default_distractor_types = [
            [],
            [en.ShapeType.PENTAGON],
            [en.ShapeType.CIRCLE, en.ShapeType.PENTAGON],
        ]
        default_target_poses = [
            # (x, y, theta)
            (0.8, -0.7, 2.37),
            (-0.68, 0.72, 1.28),
        ]
        default_distractor_poses = [
            # (x, y, theta)
            [],
            [(-0.05, -0.2, -1.09)],
            [(-0.75, -0.55, 2.78), (0.3, -0.82, -1.15)],
        ]

        if self.rand_shape_count:
            target_count = self.rng.randint(1, 2 + 1)
            distractor_counts = [
                self.rng.randint(0, 2 + 1) for c in distractor_colours
            ]
        else:
            target_count = len(default_target_types)
            distractor_counts = [len(l) for l in default_distractor_types]

        if self.rand_shape_type:
            shape_types_np = np.asarray(en.SHAPE_TYPES, dtype='object')
            target_types = [
                self.rng.choice(shape_types_np) for _ in range(target_count)
            ]
            distractor_types = [[
                self.rng.choice(shape_types_np) for _ in range(dist_count)
            ] for dist_count in distractor_counts]
        else:
            target_types = default_target_types
            distractor_types = default_distractor_types

        if self.rand_layout:
            # will do post-hoc randomisation at the end
            target_poses = [(0, 0, 0)] * target_count
            distractor_poses = [
                [(0, 0, 0)] * dcount for dcount in distractor_counts
            ]
        else:
            target_poses = default_target_poses
            distractor_poses = default_distractor_poses

        assert len(target_types) == target_count
        assert len(target_poses) == target_count
        assert len(distractor_types) == len(distractor_counts)
        assert len(distractor_types) == len(distractor_colours)
        assert len(distractor_types) == len(distractor_poses)
        assert all(
            len(types) == dcount
            for types, dcount in zip(distractor_types, distractor_counts))
        assert all(
            len(poses) == dcount
            for poses, dcount in zip(distractor_poses, distractor_counts))

        self.__target_shapes = [
            self._make_shape(shape_type=shape_type,
                             colour_name=target_colour,
                             init_pos=(shape_x, shape_y),
                             init_angle=shape_angle)
            for shape_type, (shape_x, shape_y,
                             shape_angle) in zip(target_types, target_poses)
        ]
        self.__distractor_shapes = []
        for dist_colour, dist_types, dist_poses \
                in zip(distractor_colours, distractor_types, distractor_poses):
            for shape_type, (shape_x, shape_y, shape_angle) \
                    in zip(dist_types, dist_poses):
                dist_shape = self._make_shape(shape_type=shape_type,
                                              colour_name=dist_colour,
                                              init_pos=(shape_x, shape_y),
                                              init_angle=shape_angle)
                self.__distractor_shapes.append(dist_shape)
        shape_ents = self.__target_shapes + self.__distractor_shapes
        self.add_entities(shape_ents)

        # add this last so it shows up on top, but before layout randomisation,
        # since it needs to be added to the space before randomising
        self.add_entities([robot])

        if self.rand_layout:
            for entity in [robot, *shape_ents]:
                geom.pm_randomise_pose(space=self._space,
                                       bodies=entity.bodies,
                                       arena_lrbt=self.ARENA_BOUNDS_LRBT,
                                       rng=self.rng,
                                       rand_pos=True,
                                       rand_rot=True)

        # set up index for lookups
        self.__ent_index = en.EntityIndex(shape_ents)

    def score_on_end_of_traj(self):
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            contained=True, ent_index=self.__ent_index)
        target_set = set(self.__target_shapes)
        distractor_set = set(self.__distractor_shapes)
        n_overlap_targets = len(target_set & overlap_ents)
        n_overlap_distractors = len(distractor_set & overlap_ents)
        # what fraction of targets are in the overlap set?
        target_frac_done = n_overlap_targets / len(target_set)
        if len(overlap_ents) == 0:
            contamination_rate = 0
        else:
            # what fraction of the overlap set are distractors?
            contamination_rate = n_overlap_distractors / len(overlap_ents)
        # score guide:
        # - 1 if all target shapes and no distractors are in the target region
        # - 0 if no target shapes in the target region
        # - somewhere in between if some are all target shapes are there, but
        #   there's also contamination (more contamination = worse, fewer
        #   target shapes in target region = worse).
        return target_frac_done * (1 - contamination_rate)
