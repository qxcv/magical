from gym.utils import EzPickle
import numpy as np

from magical import geom
from magical.base_env import BaseEnv, ez_init
import magical.entities as en

SMALL_POS_BOUND = 0.05
DEFAULT_ROBOT_POSE = ((0.058, 0.53), -2.13)
DEFAULT_GOAL_COLOUR = en.ShapeColour.BLUE
DEFAULT_GOAL_XYHW = (-0.62, -0.17, 0.76, 0.75)


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

    def on_reset(self):
        goal_xyhw = DEFAULT_GOAL_XYHW
        if self.rand_poses_minor or self.rand_poses_full:
            # randomise width and height of goal region
            # (unfortunately this has to be done before pose randomisation b/c
            # I don't have an easy way of changing size later)
            if self.rand_poses_minor:
                hw_bound = self.JITTER_TARGET_BOUND
            else:
                hw_bound = None
            sampled_hw = geom.randomise_hw(self.RAND_GOAL_MIN_SIZE,
                                           self.RAND_GOAL_MAX_SIZE,
                                           self.rng,
                                           current_hw=goal_xyhw[2:],
                                           linf_bound=hw_bound)
            goal_xyhw = (*goal_xyhw[:2], *sampled_hw)

        # colour the goal region
        if self.rand_goal_colour:
            goal_colour = self.rng.choice(
                np.asarray(en.SHAPE_COLOURS, dtype='object'))
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
                # limit amount by which position and rotation can be randomised
                pos_limits = self.JITTER_POS_BOUND
                rot_limits = [None, self.JITTER_ROT_BOUND]
            else:
                # no limits, can randomise as much as needed
                assert self.rand_poses_full
                pos_limits = rot_limits = None

            geom.pm_randomise_all_poses(self._space,
                                        (self.__goal_ref, self._robot),
                                        self.ARENA_BOUNDS_LRBT,
                                        rng=self.rng,
                                        rand_pos=True,
                                        rand_rot=(False, True),
                                        rel_pos_linf_limits=pos_limits,
                                        rel_rot_limits=rot_limits)

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
