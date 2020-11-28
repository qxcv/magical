"""Gym wrapper for shape-pushing environment."""

import abc
import collections
import functools
import inspect

import gym
from gym import spaces
from gym.utils import EzPickle
import numpy as np
import pymunk as pm

import magical.entities as en
import magical.gym_render as r
from magical.phys_vars import PhysicsVariablesBase, PhysVar
from magical.style import ARENA_ZOOM_OUT, COLOURS_RGB, lighten_rgb


def ez_init(*args, **kwargs):
    """Decorator to initialise EzPickle from all arguments and keyword
    arguments. Use it like this:

        class C(…, EzPickle):
            @ez_init()
            def __init__(self, spam, ham, …):
                …"""
    assert len(args) == 0 and len(kwargs) == 0, \
        "ez_init takes no args at the moment (use `@ez_init()` only)"

    def inner_decorator(method):
        # sanity checks
        msg = f"Got function {method}; should only be used to decorate " \
              f"__init__() methods of classes"
        assert inspect.isfunction(method), msg
        assert method.__name__ == '__init__', msg

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self_var = args[0]
            EzPickle.__init__(self_var, *args[1:], **kwargs)
            return method(*args, **kwargs)

        return wrapper

    return inner_decorator


class PhysicsVariables(PhysicsVariablesBase):
    """Default values & randomisation ranges for key physical parameters of the
    environment."""
    # robot_pos_joint_max_force = PhysVar(3, (2.5, 4))
    robot_pos_joint_max_force = PhysVar(3, (2.2, 3.5))
    robot_rot_joint_max_force = PhysVar(1, (0.7, 1.5))
    robot_finger_max_force = PhysVar(4, (2.5, 4.5))
    shape_trans_joint_max_force = PhysVar(1.5, (1.0, 1.8))
    shape_rot_joint_max_force = PhysVar(0.1, (0.07, 0.15))


class BaseEnv(gym.Env, abc.ABC):
    # constants for all envs
    ROBOT_RAD = 0.2
    ROBOT_MASS = 1.0
    SHAPE_RAD = ROBOT_RAD * 0.6
    ARENA_BOUNDS_LRBT = [-1, 1, -1, 1]
    ARENA_SIZE_MAX = max(ARENA_BOUNDS_LRBT)
    # minimum and maximum size of goal regions used during randomisation
    RAND_GOAL_MIN_SIZE = 0.5
    RAND_GOAL_MAX_SIZE = 0.8
    RAND_GOAL_SIZE_RANGE = RAND_GOAL_MAX_SIZE - RAND_GOAL_MIN_SIZE
    # the following are used to standardise what "jitter" means across
    # different tasks
    JITTER_PCT = 0.05
    JITTER_POS_BOUND = ARENA_SIZE_MAX * JITTER_PCT / 2.0
    JITTER_ROT_BOUND = JITTER_PCT * np.pi
    JITTER_TARGET_BOUND = JITTER_PCT * RAND_GOAL_SIZE_RANGE / 2

    def __init__(self,
                 *,
                 res_hw=(256, 256),
                 fps=20,
                 phys_steps=10,
                 phys_iter=10,
                 max_episode_steps=None,
                 rand_dynamics=False,
                 ego_view=True,
                 allo_view=True):
        self.phys_iter = phys_iter
        self.phys_steps = phys_steps
        self.fps = fps
        self.res_hw = res_hw
        self.max_episode_steps = max_episode_steps
        self.ego_view = ego_view
        self.allo_view = allo_view
        assert self.ego_view or self.allo_view, \
            "must use egocentric view or allocentric view (or both)"
        make_image_space = functools.partial(spaces.Box,
                                             low=0.0,
                                             high=255,
                                             shape=(*res_hw, 3),
                                             dtype=np.uint8)
        space_dict = collections.OrderedDict()
        if self.allo_view:
            space_dict['allo'] = make_image_space()
        if self.ego_view:
            space_dict['ego'] = make_image_space()
        self.observation_space = spaces.Dict(space_dict)
        # action space includes every combination of those
        self.action_space = spaces.Discrete(len(en.ACTION_NUMS_FLAGS_NAMES))

        # state/rendering (see reset())
        self._entities = None
        self._space = None
        self._robot = None
        self._episode_steps = None
        self._phys_vars = None
        # this is for background rendering
        self.viewer = None
        # common randomisation option for all envs
        self.rand_dynamics = rand_dynamics

        self.seed()

    def action_to_flags(self, int_action):
        """Parse a 'flat' integer action into a combination of flags."""
        return en.ACTION_ID_TO_FLAGS[int(int_action)]

    def flags_to_action(self, flags):
        """Convert a 'structured' list of flags into a single flag integer
        action. Inverse of action_to_flags."""
        return en.FLAGS_TO_ACTION_ID[tuple(flags)]

    def seed(self, seed=None):
        """Initialise the PRNG and return seed necessary to reproduce results.

        (TODO: should I also seed action/observation spaces? Not clear.)"""
        if seed is None:
            seed = np.random.randint(0, (1 << 31) - 1)
        self.rng = np.random.RandomState(seed=seed)
        return [seed]

    def _make_robot(self, init_pos, init_angle):
        return en.Robot(radius=self.ROBOT_RAD,
                        init_pos=init_pos,
                        init_angle=init_angle,
                        mass=self.ROBOT_MASS)

    def _make_shape(self, **kwargs):
        return en.Shape(shape_size=self.SHAPE_RAD, **kwargs)

    @abc.abstractmethod
    def on_reset(self):
        """Set up entities necessary for this environment, and reset any other
        data needed for the env. Must create a robot in addition to any
        necessary entities, and return bot hthe robot and the other entities
        separately.

        Returns: a tuple with the following elements:
            robot (en.Robot): an initialised robot to be controlled by the
                user.
            ents ([en.Entity]): list of other entities necessary for this
                environment."""

    def add_entities(self, entities):
        """Adds a list of entities to the current entities list and sets it up.
        Only intended to be used from within on_reset(). Needs to be called for
        every created entity or else they will not be added to the space!

        Args:
            entities (en.Entity): the entity to add."""
        for entity in entities:
            if isinstance(entity, en.Robot):
                self._robot = entity
            self._entities.append(entity)
            entity.setup(self.viewer, self._space, self._phys_vars)

    def reset(self):
        self._episode_steps = 0
        # delete old entities/space
        self._entities = []
        self._space = None
        self._robot = None
        self._phys_vars = None
        if self.viewer is None:
            res_h, res_w = self.res_hw
            background_colour = lighten_rgb(COLOURS_RGB['grey'], times=4)
            self.viewer = r.Viewer(res_w,
                                   res_h,
                                   visible=False,
                                   background_rgb=background_colour)
        else:
            # these will get added back later
            self.viewer.reset_geoms()
        self._space = pm.Space()
        self._space.collision_slop = 0.01
        self._space.iterations = self.phys_iter

        if self.rand_dynamics:
            # Randomise the physics properties of objects and the robot a
            # little bit.
            self._phys_vars = PhysicsVariables.sample(self.rng)
        else:
            self._phys_vars = PhysicsVariables.defaults()

        # set up robot and arena
        arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
        self._arena = en.ArenaBoundaries(left=arena_l,
                                         right=arena_r,
                                         bottom=arena_b,
                                         top=arena_t)
        self._arena_w = arena_r - arena_l
        self._arena_h = arena_t - arena_b
        self.add_entities([self._arena])
        reset_rv = self.on_reset()
        assert reset_rv is None, \
            f"on_reset method of {type(self)} returned {reset_rv}, but "\
            f"should return None"
        assert isinstance(self._robot, en.Robot)
        assert len(self._entities) >= 1

        assert np.allclose(self._arena.left + self._arena.right, 0)
        assert np.allclose(self._arena.bottom + self._arena.top, 0)
        self._use_allo_cam()

        # # step forward by one second so PyMunk can recover from bad initial
        # # conditions
        # # (disabled some time in 2019; don't think it's necessary with
        # # well-designed envs)
        # forward_time = 1.0
        # forward_frames = int(math.ceil(forward_time * self.fps))
        # for _ in range(forward_frames):
        #     self._phys_steps_on_frame()

        return self.render(mode='rgb_array')

    def _phys_steps_on_frame(self):
        phys_steps = 10
        spf = 1 / self.fps
        dt = spf / phys_steps
        for i in range(phys_steps):
            for ent in self._entities:
                ent.update(dt)
            self._space.step(dt)

    @abc.abstractmethod
    def score_on_end_of_traj(self):
        """Compute the score for this trajectory. Only called at the last step
        of the trajectory.

        Returns:
           score (float): number in [0, 1] indicating the worst possible
               performance (0), the best possible performance (1) or something
               in between. Should apply to the WHOLE trajectory."""

    def step(self, action):
        # step forward physics
        ac_flag_ud, ac_flag_lr, ac_flag_grip = self.action_to_flags(action)
        action_flag = en.RobotAction.NONE
        action_flag |= ac_flag_ud
        action_flag |= ac_flag_lr
        action_flag |= ac_flag_grip
        self._robot.set_action(action_flag)
        self._phys_steps_on_frame()

        info = {}
        # always 0 reward (avoids training RL algs accidentally)
        reward = 0.0

        # check episode step count
        self._episode_steps += 1
        done = False
        eval_score = 0.0
        if self.max_episode_steps is not None:
            done = done or self._episode_steps >= self.max_episode_steps
        if done:
            eval_score = self.score_on_end_of_traj()
            assert 0 <= eval_score <= 1, \
                f'eval score {eval_score} out of range for env {self}'
            # These were my attempts at sneaking episode termination info
            # through the Monitor and SubprocVecEnv wrappers ('monitor_info'
            # was a keyword I gave to info_keywords in the Monitor
            # constructor). I don't know why, but neither approach worked.
            # info['monitor_info'] = end_ep_dict
            # info['episode'] = end_ep_dict
        # we *always* include a score, even if it's zero, because at least one
        # RL framework (rlpyt) refuses to recognise keys that aren't present at
        # the first time step.
        info.update(eval_score=eval_score)

        obs_dict_u8 = self.render(mode='rgb_array')

        return obs_dict_u8, reward, done, info

    def _use_ego_cam(self):
        self.viewer.set_cam_follow(
            source_xy_world=(self._robot.robot_body.position.x,
                             self._robot.robot_body.position.y),
            target_xy_01=(0.5, 0.15),
            viewport_hw_world=(self._arena_h * ARENA_ZOOM_OUT,
                               self._arena_w * ARENA_ZOOM_OUT),
            rotation=self._robot.robot_body.angle)

    def _use_allo_cam(self):
        self.viewer.set_bounds(left=self._arena.left * ARENA_ZOOM_OUT,
                               right=self._arena.right * ARENA_ZOOM_OUT,
                               bottom=self._arena.bottom * ARENA_ZOOM_OUT,
                               top=self._arena.top * ARENA_ZOOM_OUT)

    def render(self, mode='human'):
        # FIXME: would be simpler for this to special-case to mode='human',
        # mode='ego', and mode='allo', then handle the logic for combining ego
        # + allo in .step().
        if self.viewer is None:
            return None

        for ent in self._entities:
            ent.pre_draw()

        if mode == 'human':
            self.viewer.window.set_visible(True)
        else:
            assert mode == 'rgb_array'

        view_dict = collections.OrderedDict()
        is_human = mode == 'human'
        if self.allo_view:
            self._use_allo_cam()
            view_dict['allo'] = self.viewer.render(return_rgb_array=True,
                                                   update_foreground=is_human)
        if self.ego_view:
            self._use_ego_cam()
            # allo view is the default foreground view; we only show ego view
            # in foreground if it's the only thing available
            view_dict['ego'] = self.viewer.render(return_rgb_array=True,
                                                  update_foreground=is_human
                                                  and not self.allo_view)

        return view_dict

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def debug_print_entity_spec(self):
        """Hacky function to extract specifications for locations & types of
        entities, etc., from the current state of the environment. Good for
        constructing new demo configs from world states."""
        robot_pose = None
        block_colours = []
        block_poses = []
        block_shapes = []
        goal_region_xyhws = []
        goal_region_colours = []
        for entity in self._entities:
            if isinstance(entity, en.Robot):
                _, sig = entity.reconstruct_signature()
                robot_pose = (sig['init_pos'], sig['init_angle'])
            elif isinstance(entity, en.Shape):
                _, sig = entity.reconstruct_signature()
                block_colours.append(sig['colour_name'])
                block_poses.append((sig['init_pos'], sig['init_angle']))
                block_shapes.append(sig['shape_type'])
            elif isinstance(entity, en.GoalRegion):
                _, sig = entity.reconstruct_signature()
                goal_region_xyhws.append(
                    (sig['x'], sig['y'], sig['h'], sig['w']))
                goal_region_colours.append(sig['colour_name'])

        def f_pose(pose):
            # string-format a pose
            (x, y), angle = pose
            return '((%.3f, %.3f), %.3f)' % (x, y, angle)

        def f_xyhw(xyhw):
            return '(%.3f, %.3f, %.3f, %.3f)' % xyhw

        def f_colour(colour):
            # strong-format a colour
            return f'en.ShapeColour.{colour.name.upper()}'

        def f_shape(shape):
            # strong-format a shape
            return f'en.ShapeType.{shape.name.upper()}'

        def lst(f, lst):
            # apply function f to each element of lst, convert the results to
            # string, then present as string-formatted list
            return '[' + ', '.join(str(f(e)) for e in lst) + ']'

        if robot_pose:
            print(f'ROBOT_POSE = {f_pose(robot_pose)}')
        if block_colours:
            print(f'BLOCK_COLOURS = {lst(f_colour, block_colours)}')
            print(f'BLOCK_SHAPES = {lst(f_shape, block_shapes)}')
            print(f'BLOCK_POSES = {lst(f_pose, block_poses)}')
        if goal_region_xyhws:
            print(f'GOAL_REGION_XYHWS = {lst(f_xyhw, goal_region_xyhws)}')
            print(
                f'GOAL_REGION_COLOURS = {lst(f_colour, goal_region_colours)}')
