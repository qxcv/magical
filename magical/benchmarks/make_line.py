import itertools as it

from gym.utils import EzPickle
import numpy as np

from magical import geom
from magical.base_env import BaseEnv, ez_init
import magical.entities as en

INLIER_RAD_MULT = 1.5
MAX_SEP_RADS = 3.5
MIN_BLOCKS = 3
MAX_BLOCKS = 4
DEFAULT_ROBOT_POSE = ((0.702, -0.255), 0.347)
DEFAULT_BLOCK_COLOURS = [
    en.ShapeColour.BLUE,
    en.ShapeColour.YELLOW,
    en.ShapeColour.RED,
    en.ShapeColour.GREEN,
]
DEFAULT_BLOCK_SHAPES = [
    en.ShapeType.STAR,
    en.ShapeType.CIRCLE,
    en.ShapeType.STAR,
    en.ShapeType.PENTAGON,
]
DEFAULT_BLOCK_POSES = [((0.790, -0.820), -0.721), ((-0.177, 0.383), -1.733),
                       ((-0.051, -0.128), 2.696), ((-0.292, -0.745), -0.159)]


def longest_line(points, inlier_dist, max_separation):
    """Identifies lines of blocks by performing RANSAC-style robust regression,
    but with two differences:

    1. It exhaustively searches over all pairs, instead of randomly sampling.
    2. It ensures that the distance between adjacent pairs of blocks along the
       returned line is less than max_separation.

    Returns number of points in the longest identified line."""
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 2
    npts = len(points)
    best = min(1, npts)
    for i in range(npts - 1):
        for j in range(i + 1, npts):
            # draw a line (p_i -> p_j) and find all the inliers for that line
            pi = points[i]
            offs = points - pi[None]
            pj_off = offs[j]
            pj_unit = pj_off / np.linalg.norm(pj_off)
            proj_lens = np.squeeze(offs @ pj_unit[:, None], axis=1)
            assert len(proj_lens) == len(points)
            dists = np.linalg.norm(offs - proj_lens[:, None] * pj_unit, axis=1)
            inlier_inds, = np.nonzero(dists <= inlier_dist)
            if len(inlier_inds) <= best:
                continue

            # now that we've found the inliers, find the largest subsequence of
            # inliers that are separated by a distance of at most
            # max_separation along the line
            inlier_proj_lens = proj_lens[inlier_inds]
            inlier_proj_lens.sort()
            seps = np.abs(np.diff(inlier_proj_lens))
            # find longest run of nearby shapes
            all_runs = it.groupby(seps <= max_separation)
            one_run_lens = [len(list(r)) for v, r in all_runs if v]
            max_run = max(one_run_lens, default=0) + 1
            if max_run > best:
                best = max_run

    return best


class MakeLineEnv(BaseEnv, EzPickle):
    """In this environment, the objective is to put all blocks in a straight
    line, arranged at any angle."""
    @ez_init()
    def __init__(self, rand_colours, rand_shapes, rand_count,
                 rand_layout_minor, rand_layout_full, **kwargs):
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
        self.inlier_dist = self.SHAPE_RAD * INLIER_RAD_MULT
        self.max_sep = self.SHAPE_RAD * MAX_SEP_RADS

    def on_reset(self):
        robot_pos, robot_angle = DEFAULT_ROBOT_POSE
        robot = self._make_robot(robot_pos, robot_angle)

        block_shapes = DEFAULT_BLOCK_SHAPES
        block_colours = DEFAULT_BLOCK_COLOURS
        block_poses = DEFAULT_BLOCK_POSES
        if self.rand_count:
            n_blocks = self.rng.randint(MIN_BLOCKS, MAX_BLOCKS + 1)
            block_poses = block_poses[:1] * n_blocks
        else:
            n_blocks = len(block_shapes)
        if self.rand_colours:
            block_colours = self.rng.choice(en.SHAPE_COLOURS,
                                            size=n_blocks).tolist()
        if self.rand_shapes:
            block_shapes = self.rng.choice(en.SHAPE_TYPES,
                                           size=n_blocks).tolist()

        self._blocks = []
        for bshape, bcol, (bpos, bangle) in zip(block_shapes, block_colours,
                                                block_poses):
            new_block = self._make_shape(shape_type=bshape,
                                         colour_name=bcol,
                                         init_pos=bpos,
                                         init_angle=bangle)
            self._blocks.append(new_block)

        self.add_entities(self._blocks)
        self.add_entities([robot])

        if self.rand_layout_minor or self.rand_layout_full:
            all_ents = (robot, *self._blocks)
            if self.rand_layout_minor:
                pos_limits = self.JITTER_POS_BOUND
                rot_limit = self.JITTER_ROT_BOUND
            else:
                assert self.rand_layout_full
                pos_limits = rot_limit = None

            geom.pm_randomise_all_poses(self._space,
                                        all_ents,
                                        self.ARENA_BOUNDS_LRBT,
                                        rng=self.rng,
                                        rand_pos=True,
                                        rand_rot=True,
                                        rel_pos_linf_limits=pos_limits,
                                        rel_rot_limits=rot_limit)

    def score_on_end_of_traj(self):
        points = np.asarray([(b.shape_body.position.x, b.shape_body.position.y)
                             for b in self._blocks],
                            dtype='float64')
        line_len = longest_line(points, self.inlier_dist, self.max_sep)
        max_line_len = len(points)
        # 2 outliers = score of 0; 1 outlier = score of 0.5; 0 outliers = score
        # of 1
        min_line_len = max(max_line_len - 2, 2)
        score = max(line_len - min_line_len, 0) / (max_line_len - min_line_len)
        return score
