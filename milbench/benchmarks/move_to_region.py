from gym.utils import EzPickle
import numpy as np

from milbench import geom
from milbench.base_env import BaseEnv, ez_init
import milbench.entities as en

ALL_COLOURS = np.array([
    en.ShapeColour.RED,
    en.ShapeColour.GREEN,
    en.ShapeColour.BLUE,
    en.ShapeColour.YELLOW,
],
                       dtype='object')
SMALL_POS_BOUND = 0.05
DEFAULT_ROBOT_POSE = ((0.058, 0.53), -2.13)
DEFAULT_GOAL_COLOUR = ALL_COLOURS[2]
DEFAULT_GOAL_XYHW = (-0.62, -0.17, 0.76, 0.75)
TARGET_SIZE_MIN = 0.5
TARGET_SIZE_MAX = 0.9
# In "minor pose randomisation" mode, allow deviation by plus or minus this
# amount along any dimension (x, y, height, width, theta). Expressed as
# fraction of variation allowed in full randomisation.
MINOR_DEVIATION_PCT = 0.05


class MoveToRegionEnv(BaseEnv, EzPickle):
    """Simple task where the robot merely has to move to a single coloured goal
    region."""
    @ez_init()
    def __init__(self,
                 rand_poses_minor=False,
                 rand_poses_full=False,
                 rand_goal_colour=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert not (rand_poses_minor and rand_poses_full), \
            "cannot specify both 'rand_poses_minor' and 'rand_poses_full'"
        self.rand_poses_minor = rand_poses_minor
        self.rand_poses_full = rand_poses_full
        self.rand_goal_colour = rand_goal_colour

    @classmethod
    def make_name(cls, suffix=None):
        base_name = 'MoveToRegion'
        return base_name + (suffix or '') + '-v0'

    def on_reset(self):
        # find location for goal region
        if self.rand_poses_minor or self.rand_poses_full:
            if self.rand_poses_minor:
                xyhw_default = np.asarray(DEFAULT_GOAL_XYHW)
                size_range = MINOR_DEVIATION_PCT * (TARGET_SIZE_MAX -
                                                    TARGET_SIZE_MIN)
                hw_min = np.maximum(xyhw_default[2:] - size_range,
                                    TARGET_SIZE_MIN)
                hw_max = np.minimum(xyhw_default[2:] + size_range,
                                    TARGET_SIZE_MAX)
                disp_range = MINOR_DEVIATION_PCT
                x_low, y_low = np.maximum(xyhw_default[:2] - disp_range, -1.0)
                x_high, y_high = np.minimum(xyhw_default[:2] + disp_range, 1.0)
            else:
                assert self.rand_poses_full
                hw_min = (TARGET_SIZE_MIN, TARGET_SIZE_MIN)
                hw_max = (TARGET_SIZE_MAX, TARGET_SIZE_MAX)
                x_low = y_low = -1.0
                x_high = y_high = 1.0
            sampled_hw = np.random.uniform(hw_min, hw_max)
            # reminder: x goes left -> right, but y goes top -> bottom
            sampled_xy = np.random.uniform(
                (x_low, max(y_low, -1.0 + sampled_hw[0])),
                (min(x_high, 1.0 - sampled_hw[1]), y_high))
            goal_xyhw = (*sampled_xy, *sampled_hw)
        else:
            goal_xyhw = DEFAULT_GOAL_XYHW

        # colour the goal region
        if self.rand_goal_colour:
            goal_colour = self.rng.choice(
                np.asarray(ALL_COLOURS, dtype='object'))
        else:
            goal_colour = DEFAULT_GOAL_COLOUR

        # place the goal region
        assert len(goal_xyhw) == 4, goal_xyhw
        goal = en.GoalRegion(*goal_xyhw, goal_colour)
        self.add_entities([goal])
        self.__goal_ref = goal

        # now place the robot
        default_robot_pos, default_robot_angle = DEFAULT_ROBOT_POSE
        robot = self._make_robot(default_robot_pos, default_robot_angle)
        self.add_entities([robot])
        self.__robot_ent_index = en.EntityIndex([robot])

        if self.rand_poses_minor or self.rand_poses_full:
            if self.rand_poses_minor:
                rot_bound = 1 * np.pi * MINOR_DEVIATION_PCT
                robot_rot_bounds = (default_robot_angle - rot_bound,
                                    default_robot_angle + rot_bound)
                pos_bound = 1 * MINOR_DEVIATION_PCT
                robot_pos_bounds = np.asarray(
                    ((default_robot_pos[0] - pos_bound,
                      default_robot_pos[0] + pos_bound),
                     (default_robot_pos[1] - pos_bound,
                      default_robot_pos[1] + pos_bound)))
                robot_pos_bounds = np.clip(robot_pos_bounds, -1.0, 1.0)
            else:
                assert self.rand_poses_full
                robot_rot_bounds = robot_pos_bounds = None
            geom.pm_randomise_pose(space=self._space,
                                   bodies=robot.bodies,
                                   arena_lrbt=self.ARENA_BOUNDS_LRBT,
                                   rng=self.rng,
                                   rand_pos=True,
                                   rand_rot=True,
                                   pos_bounds=robot_pos_bounds,
                                   rot_bounds=robot_rot_bounds)

    def score_on_end_of_traj(self):
        # this one just has a lazy binary reward
        dist, _ = self.__goal_ref.goal_shape.point_query(
            self._robot.robot_body.position)
        if dist <= 0:
            reward = 1.0
        else:
            reward = 0.0
        assert 0 <= reward <= 1
        return reward
