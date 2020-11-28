import functools

from gym.utils import EzPickle

from magical.base_env import BaseEnv, ez_init
import magical.entities as en
import magical.geom as geom

MIN_REGIONS = 2
MAX_REGIONS = 3
# we have really small goal regions in this domain so we can fit 3+ of them in
# the workspace
MIN_GOAL_SIZE = 0.4
MAX_GOAL_SIZE = 0.5
DEFAULT_ROBOT_POSE = ((0.368, 0.586), 0.718)
DEFAULT_BLOCK_COLOURS = [
    en.ShapeColour.GREEN,
    en.ShapeColour.GREEN,
    en.ShapeColour.BLUE,
]
DEFAULT_BLOCK_SHAPES = [
    en.ShapeType.PENTAGON,
    en.ShapeType.SQUARE,
    en.ShapeType.PENTAGON,
]
DEFAULT_BLOCK_POSES = [
    ((0.289, 0.030), 0.307),
    ((0.133, -0.561), 1.699),
    ((-0.336, 0.000), -1.529),
]
DEFAULT_REGION_XYHWS = [
    (-0.032, 0.348, 0.427, 0.468),
    (0.019, -0.391, 0.460, 0.458),
    (-0.681, 0.196, 0.498, 0.418),
]
DEFAULT_REGION_COLOURS = [
    en.ShapeColour.GREEN,
    en.ShapeColour.GREEN,
    en.ShapeColour.RED,
]


class FixColourEnv(BaseEnv, EzPickle):
    """There are several coloured regions, each containing a single block.
    Exactly one of the coloured regions has a block that _doesn't_ match its
    colour; the block should be removed from this region, but not the
    others."""
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
        robot_pos, robot_angle = DEFAULT_ROBOT_POSE
        robot = self._make_robot(robot_pos, robot_angle)

        block_colours = DEFAULT_BLOCK_COLOURS
        region_colours = DEFAULT_REGION_COLOURS
        block_shapes = DEFAULT_BLOCK_SHAPES
        block_poses = DEFAULT_BLOCK_POSES
        region_xyhws = DEFAULT_REGION_XYHWS

        # randomise count
        n_regions = len(block_colours)
        if self.rand_count:
            n_regions = self.rng.randint(MIN_REGIONS, MAX_REGIONS + 1)
            block_poses = block_poses[:1] * n_regions
            region_xyhws = region_xyhws[:1] * n_regions

        # randomise colours
        if self.rand_colours:
            region_colours = self.rng.choice(en.SHAPE_COLOURS,
                                             size=n_regions).tolist()
            block_colours = list(region_colours)
            # randomly choose one block to be the odd one out
            odd_idx = self.rng.randint(len(block_colours))
            new_col_idx = self.rng.randint(len(en.SHAPE_COLOURS) - 1)
            if en.SHAPE_COLOURS[new_col_idx] == block_colours[odd_idx]:
                new_col_idx += 1
            block_colours[odd_idx] = en.SHAPE_COLOURS[new_col_idx]

        # randomise shapes
        if self.rand_shapes:
            block_shapes = self.rng.choice(en.SHAPE_TYPES,
                                           size=n_regions).tolist()

        # create all regions, randomising their height/width first if necessary
        if self.rand_layout_minor or self.rand_layout_full:
            if self.rand_layout_minor:
                hw_bound = self.JITTER_TARGET_BOUND
            else:
                hw_bound = None
            rand_hw = functools.partial(geom.randomise_hw,
                                        MIN_GOAL_SIZE,
                                        MAX_GOAL_SIZE,
                                        self.rng,
                                        linf_bound=hw_bound)
            region_xyhws = [(x, y, *rand_hw(current_hw=hw))
                            for x, y, *hw in region_xyhws]
        sensors = [
            en.GoalRegion(*xyhw, colour)
            for colour, xyhw in zip(region_colours, region_xyhws)
        ]
        self.add_entities(sensors)
        self._sensors = sensors

        # set up blocks
        blocks = []
        self._target_blocks = []
        for bshape, bcol, tcol, (bpos,
                                 bangle) in zip(block_shapes, block_colours,
                                                region_colours, block_poses):
            new_block = self._make_shape(shape_type=bshape,
                                         colour_name=bcol,
                                         init_pos=bpos,
                                         init_angle=bangle)
            blocks.append(new_block)
            if bcol != tcol:
                # we don't want this region to contain any blocks
                self._target_blocks.append([])
            else:
                # we want this region to keep the original block
                self._target_blocks.append([new_block])
        self.add_entities(blocks)

        # add robot last (for draw order reasons)
        self.add_entities([robot])

        if self.rand_layout_minor or self.rand_layout_full:
            sensors_robot = (*sensors, robot)
            if self.rand_layout_minor:
                pos_limits = self.JITTER_POS_BOUND
                rot_limit = self.JITTER_ROT_BOUND
            else:
                assert self.rand_layout_full
                pos_limits = rot_limit = None
            # don't randomise rotations of goal regions
            rand_rot = [False] * n_regions + [True]
            # don't collision check with blocks
            block_pm_shapes = sum((b.shapes for b in blocks), [])

            # randomise positions of goal regions and robot
            geom.pm_randomise_all_poses(self._space,
                                        sensors_robot,
                                        self.ARENA_BOUNDS_LRBT,
                                        rng=self.rng,
                                        rand_pos=True,
                                        rand_rot=rand_rot,
                                        rel_pos_linf_limits=pos_limits,
                                        rel_rot_limits=rot_limit,
                                        ignore_shapes=block_pm_shapes)

            # shift blocks out of the way of each other
            for block, sensor in zip(blocks, sensors):
                geom.pm_shift_bodies(self._space,
                                     block.bodies,
                                     position=sensor.bodies[0].position)
            # randomise each inner block position separately
            for block, sensor, (_, _,
                                *sensor_hw) in zip(blocks, sensors,
                                                   region_xyhws):
                block_pos_limit = max(0, min(sensor_hw) / 2 - self.SHAPE_RAD)
                if self.rand_layout_minor:
                    block_pos_limit = min(self.JITTER_POS_BOUND,
                                          block_pos_limit)
                geom.pm_randomise_pose(self._space,
                                       block.bodies,
                                       self.ARENA_BOUNDS_LRBT,
                                       ignore_shapes=sensor.shapes,
                                       rng=self.rng,
                                       rand_pos=True,
                                       rand_rot=True,
                                       rel_pos_linf_limit=block_pos_limit,
                                       rel_rot_limit=rot_limit)

        # block lookup index
        self._block_index = en.EntityIndex(blocks)

    def score_on_end_of_traj(self):
        # binary scoring: give score of 1 iff we get exactly the right result
        complete = True
        for sensor, tgt_block_list in zip(self._sensors, self._target_blocks):
            overlap_ents = sensor.get_overlapping_ents(
                com_overlap=True, ent_index=self._block_index)
            if list(overlap_ents) != tgt_block_list:
                complete = False
                break
        return 1.0 if complete else 0.0
