from gym.utils import EzPickle

from magical.base_env import BaseEnv, ez_init
import magical.entities as en
import magical.geom as geom

DEFAULT_QUERY_COLOUR = en.ShapeColour.YELLOW
DEFAULT_QUERY_SHAPE = en.ShapeType.PENTAGON
DEFAULT_OUT_BLOCK_SHAPES = [
    en.ShapeType.PENTAGON,
    en.ShapeType.CIRCLE,
    en.ShapeType.CIRCLE,
    en.ShapeType.SQUARE,
    en.ShapeType.STAR,
    # last one is the target shape
    DEFAULT_QUERY_SHAPE,
]
DEFAULT_OUT_BLOCK_COLOURS = [
    en.ShapeColour.GREEN,
    en.ShapeColour.RED,
    en.ShapeColour.RED,
    en.ShapeColour.YELLOW,
    en.ShapeColour.BLUE,
    # last one is the target shape
    DEFAULT_QUERY_COLOUR,
]
DEFAULT_OUT_BLOCK_POSES = [
    ((-0.066751, 0.7552), -2.9266),
    ((-0.05195, 0.31468), 1.5418),
    ((0.57528, -0.46865), -2.2141),
    ((0.40594, -0.74977), 0.24582),
    ((0.45254, 0.3681), -1.0834),
    # last one is the target shape
    ((0.76849, -0.10652), 0.10028),
]
DEFAULT_ROBOT_POSE = ((-0.57, 0.25), 3.83)
DEFAULT_TARGET_REGION_XYHW = (-0.72, -0.22, 0.67, 0.72)
DEFAULT_QUERY_BLOCK_POSE = ((-0.33, -0.49), -0.51)


class FindDupeEnv(BaseEnv, EzPickle):
    """Task where robot has to find a block that is part of a group of
    duplicate blocks and bring it to the goal region. Every instance of this
    environment starts with one element of the duplicate group already inside a
    single goal region, and at least one element outside. Moving other
    duplicates into the goal region doesn't satisfy the goal.

    Details/corner cases: there's only ever one 'target' block in the goal
    region to start with. There will be at least one corresponding block with
    the *exact same attributes* outside of the goal region. There may be more
    than one corresponding block, and if that's the case then the robot only
    needs to fetch one of them (fetching more incurs no penalty)."""
    @ez_init()
    def __init__(self,
                 rand_colours=False,
                 rand_shapes=False,
                 rand_count=False,
                 rand_layout_minor=False,
                 rand_layout_full=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rand_colours = rand_colours
        self.rand_shapes = rand_shapes
        self.rand_count = rand_count
        self.rand_layout_minor = rand_layout_minor
        self.rand_layout_full = rand_layout_full
        if self.rand_count:
            assert self.rand_layout_full and self.rand_shapes \
                and self.rand_colours, "if shape count is randomised then " \
                "layout, shapes, and colours must be fully randomised too"

    def on_reset(self):
        # make the robot (pose is randomised at end, if necessary)
        robot_pos, robot_angle = DEFAULT_ROBOT_POSE
        robot = self._make_robot(robot_pos, robot_angle)

        # sort out shapes and colours of blocks, including the target
        # shape/colour
        query_colour = DEFAULT_QUERY_COLOUR
        query_shape = DEFAULT_QUERY_SHAPE
        out_block_colours = DEFAULT_OUT_BLOCK_COLOURS
        out_block_shapes = DEFAULT_OUT_BLOCK_SHAPES
        n_out_blocks = len(DEFAULT_OUT_BLOCK_COLOURS)
        if self.rand_count:
            # have between 1 and 5 randomly generated blocks, plus one outside
            # block that always matches the query block
            n_out_blocks = self.rng.randint(1, 5 + 1) + 1
        n_distractors = n_out_blocks - 1

        if self.rand_colours:
            query_colour = self.rng.choice(en.SHAPE_COLOURS)
            out_block_colours = self.rng.choice(en.SHAPE_COLOURS,
                                                size=n_distractors).tolist()
            # last block always matches the query
            out_block_colours.append(query_colour)
        if self.rand_shapes:
            query_shape = self.rng.choice(en.SHAPE_TYPES)
            out_block_shapes = self.rng.choice(en.SHAPE_TYPES,
                                               size=n_distractors).tolist()
            out_block_shapes.append(query_shape)

        # create the target region
        target_region_xyhw = DEFAULT_TARGET_REGION_XYHW
        if self.rand_layout_minor or self.rand_layout_full:
            if self.rand_layout_minor:
                hw_bound = self.JITTER_TARGET_BOUND
            else:
                hw_bound = None
            target_hw = geom.randomise_hw(self.RAND_GOAL_MIN_SIZE,
                                          self.RAND_GOAL_MAX_SIZE,
                                          self.rng,
                                          current_hw=target_region_xyhw[2:],
                                          linf_bound=hw_bound)
            target_region_xyhw = (*target_region_xyhw[:2], *target_hw)
        sensor = en.GoalRegion(*target_region_xyhw, query_colour)
        self.add_entities([sensor])
        self.__sensor_ref = sensor

        # set up poses, as necessary
        query_block_pose = DEFAULT_QUERY_BLOCK_POSE
        out_block_poses = DEFAULT_OUT_BLOCK_POSES
        if self.rand_count:
            # this will be filled out when randomising the layout
            out_block_poses = [((0, 0), 0)] * n_out_blocks

        # The "outside blocks" are those which need to be outside the goal
        # region in the initial state. That includes the clone of the query
        # block that we intentionally inserted above.
        outside_blocks = []
        # we also keep a list of blocks that count as target blocks
        self.__target_set = set()
        for bshape, bcol, (bpos, bangle) in zip(out_block_shapes,
                                                out_block_colours,
                                                out_block_poses):
            new_block = self._make_shape(shape_type=bshape,
                                         colour_name=bcol,
                                         init_pos=bpos,
                                         init_angle=bangle)
            outside_blocks.append(new_block)
            if bcol == query_colour and bshape == query_shape:
                self.__target_set.add(new_block)
        self.add_entities(outside_blocks)
        # now add the query block
        assert len(query_block_pose) == 2
        query_block = self._make_shape(shape_type=query_shape,
                                       colour_name=query_colour,
                                       init_pos=query_block_pose[0],
                                       init_angle=query_block_pose[1])
        self.__target_set.add(query_block)
        self.add_entities([query_block])
        # these are shapes that shouldn't end up in the goal region
        self.__distractor_set = set(outside_blocks) - self.__target_set

        # add robot last, but before layout randomisation
        self.add_entities([robot])

        if self.rand_layout_minor or self.rand_layout_full:
            all_ents = (sensor, robot, *outside_blocks)
            if self.rand_layout_minor:
                # limit amount by which position and rotation can be randomised
                pos_limits = self.JITTER_POS_BOUND
                rot_limit = self.JITTER_ROT_BOUND
            else:
                # no limits, can randomise as much as needed
                assert self.rand_layout_full
                pos_limits = rot_limit = None
            # randomise rotations of all entities but goal region
            rand_rot = [False] + [True] * (len(all_ents) - 1)

            geom.pm_randomise_all_poses(self._space,
                                        all_ents,
                                        self.ARENA_BOUNDS_LRBT,
                                        rng=self.rng,
                                        rand_pos=True,
                                        rand_rot=rand_rot,
                                        rel_pos_linf_limits=pos_limits,
                                        rel_rot_limits=rot_limit,
                                        ignore_shapes=query_block.shapes)

            # randomise the query block last, since it must be mostly inside
            # the (randomly-placed) sensor region
            query_pos_limit = max(
                0,
                min(target_region_xyhw[2:]) / 2 - self.SHAPE_RAD / 2)
            if self.rand_layout_minor:
                query_pos_limit = min(self.JITTER_POS_BOUND, query_pos_limit)
            geom.pm_shift_bodies(self._space,
                                 query_block.bodies,
                                 position=sensor.bodies[0].position)
            geom.pm_randomise_pose(self._space,
                                   query_block.bodies,
                                   self.ARENA_BOUNDS_LRBT,
                                   ignore_shapes=sensor.shapes,
                                   rng=self.rng,
                                   rand_pos=True,
                                   rand_rot=True,
                                   rel_pos_linf_limit=query_pos_limit,
                                   rel_rot_limit=rot_limit)

        # block lookup index
        self.__block_index = en.EntityIndex([query_block, *outside_blocks])

    def score_on_end_of_traj(self):
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            com_overlap=True, ent_index=self.__block_index)
        n_overlap_targets = len(self.__target_set & overlap_ents)
        n_overlap_distractors = len(self.__distractor_set & overlap_ents)
        # what fraction of required targets are in the overlap set? (need 2;
        # more will not make difference)
        have_two_shapes = float(n_overlap_targets >= 2)
        if len(overlap_ents) == 0:
            contamination_rate = 0
        else:
            # what fraction of the overlap set are distractors?
            contamination_rate = n_overlap_distractors / len(overlap_ents)
        return have_two_shapes * (1 - contamination_rate)
