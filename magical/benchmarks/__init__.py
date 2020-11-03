"""Register available envs and create index of train/test mappings."""

import collections
import functools
import re

import cv2
import gym
from gym.spaces import Box, Dict
from gym.wrappers import ResizeObservation
import numpy as np

from magical.benchmarks.cluster import ClusterColourEnv, ClusterShapeEnv
from magical.benchmarks.find_dupe import FindDupeEnv
from magical.benchmarks.fix_colour import FixColourEnv
from magical.benchmarks.make_line import MakeLineEnv
from magical.benchmarks.match_regions import MatchRegionsEnv
from magical.benchmarks.move_to_corner import MoveToCornerEnv
from magical.benchmarks.move_to_region import MoveToRegionEnv

__all__ = [
    'DEMO_ENVS_TO_TEST_ENVS_MAP',
    'MoveToCornerEnv',
    'MatchRegionsEnv',
    'ClusterColourEnv',
    'ClusterShapeEnv',
    'FindDupeEnv',
    'MakeLineEnv',
    'register_envs',
    'EnvName',
]

DEFAULT_RES = (384, 384)


def _gym_tree_map(f, *structures):
    """Apply a function f to the given structures. If the structures are
    dictionaries or Gym dictionary observation spaces, then it recursively
    applies the function to their values and returns a new dict (or new Dict
    observation space), as appropriate."""
    s0 = structures[0]
    if isinstance(s0, Dict):
        return Dict(
            collections.OrderedDict([
                (k,
                 _gym_tree_map(f, *(struct.spaces[k]
                                    for struct in structures)))
                for k in s0.spaces.keys()
            ]))
    elif isinstance(s0, (dict, collections.OrderedDict)):
        return type(s0)((k, _gym_tree_map(f, *(s[k] for s in structures)))
                        for k in s0.keys())
    return f(*structures)


class EagerDictFrameStack(gym.Wrapper):
    """Version of Gym's frame stack wrapper that is (1) totally eager, (2)
    stacks along channels axis instead of a separate leading axis, and (3)
    supports nested Dict observation spaces with Box values at the leaves."""
    def __init__(self, env, depth):
        super().__init__(env)
        self.depth = depth
        self.frames = collections.deque(maxlen=depth)

        def box_map(box):
            low = np.repeat(box.low, depth, axis=-1)
            high = np.repeat(box.high, depth, axis=-1)
            return Box(low=low, high=high, dtype=box.dtype)

        self.observation_space = _gym_tree_map(box_map, env.observation_space)

    def _get_observation(self):
        assert len(self.frames) == self.depth, \
            (len(self.frames), self.depth)
        return _gym_tree_map(lambda *frames: np.concatenate(frames, axis=-1),
                             *self.frames)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for _ in range(self.depth):
            self.frames.append(observation)
        return self._get_observation()


class FlattenFrameStack(gym.Wrapper):
    """Like EagerFrameStack, except it stacks all images into one big frame.
    Supports variable lookback for each key."""
    def __init__(self, env, depth_by_key):
        super().__init__(env)
        self.depth_by_key = depth_by_key
        self.frames_by_key = collections.OrderedDict([
            (k, collections.deque(maxlen=k_depth))
            for k, k_depth in depth_by_key.items()
        ])

        orig_space = None

        def box_map(box):
            nonlocal orig_space
            if orig_space is None:
                orig_space = box
            else:
                assert np.all(box.low == orig_space.low)
                assert np.all(box.high == orig_space.high)
                assert box.dtype == orig_space.dtype
            return box  # this is never used

        # figure out what the space is inside the dict
        assert isinstance(env.observation_space, Dict)
        _gym_tree_map(box_map, env.observation_space)
        self.depth_sum = sum(self.depth_by_key.values())
        new_low = np.repeat(orig_space.low, self.depth_sum, axis=-1)
        new_high = np.repeat(orig_space.high, self.depth_sum, axis=-1)
        self.observation_space = Box(low=new_low,
                                     high=new_high,
                                     dtype=orig_space.dtype)

    def _get_observation(self):
        # assume depth 1 dict
        all_frames = []
        for frames in self.frames_by_key.values():
            all_frames.extend(frames)
        len(all_frames) == self.depth_sum
        stacked_image = np.concatenate(all_frames, axis=-1)
        return stacked_image

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        for key, frame in observation.items():
            self.frames_by_key[key].append(frame)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for key, frame in observation.items():
            depth = self.depth_by_key[key]
            for _ in range(depth):
                self.frames_by_key[key].append(frame)
        return self._get_observation()


class ResizeDictObservation(gym.ObservationWrapper):
    """Version of Gym's ResizeObservation wrapper that supports dict
    observations."""
    def __init__(self, env, res):
        super().__init__(env)

        if isinstance(res, int):
            res = (res, res)
        assert all(x > 0 for x in res), res
        self.res = tuple(res)

        def space_mapper(box):
            # copy channels from the box, but replace height/width (assumed to
            # be leading two dims)
            new_shape = self.res + box.shape[2:]
            return Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        self.observation_space = _gym_tree_map(space_mapper,
                                               env.observation_space)

    def observation(self, observation):
        # use exactly the same resizing method as ResizeObservation in Gym
        def obs_mapper(box_obs):
            box_obs = cv2.resize(box_obs,
                                 self.res[::-1],
                                 interpolation=cv2.INTER_AREA)
            if box_obs.ndim == 2:
                box_obs = np.expand_dims(box_obs, -1)
            return box_obs

        return _gym_tree_map(obs_mapper, observation)


class ChannelsFirst(gym.ObservationWrapper):
    """Moves last axis around to first position. Useful for going from
    channels-last to channels-first representation."""
    def __init__(self, env):
        super().__init__(env)

        def space_mapper(box):
            assert isinstance(box, gym.spaces.Box), box
            return gym.spaces.Box(low=self._rotate(box.low),
                                  high=self._rotate(box.high),
                                  dtype=box.dtype)

        self.observation_space = _gym_tree_map(space_mapper,
                                               env.observation_space)

    @staticmethod
    def _rotate(array):
        return np.moveaxis(array, -1, 0)

    def observation(self, observation):
        return _gym_tree_map(self._rotate, observation)


def lores_stack_entry_point(env_cls, small_res, frames=4):
    def make_lores_stack(**kwargs):
        base_env = env_cls(**kwargs)
        resize_env = ResizeDictObservation(base_env, small_res)
        stack_env = EagerDictFrameStack(resize_env, frames)
        return stack_env

    return make_lores_stack


def lores_ea_entry_point(env_cls,
                         small_res,
                         allo_frames=1,
                         ego_frames=3,
                         channels_first=False):
    """For stacking ego/allo frames together."""
    def make_lores_ea(**kwargs):
        base_env = env_cls(**kwargs)
        stack_env = FlattenFrameStack(
            base_env,
            collections.OrderedDict([
                ('allo', allo_frames),
                ('ego', ego_frames),
            ]))
        resize_env = ResizeObservation(stack_env, small_res)
        if channels_first:
            return ChannelsFirst(resize_env)
        return resize_env

    return make_lores_ea


DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS = collections.OrderedDict([
    # stacking latest allo view with three most recent ego views (output is
    # just a stacked array, not a 3D array)
    ('LoRes3EA',
     functools.partial(lores_ea_entry_point,
                       small_res=(96, 96),
                       allo_frames=1,
                       ego_frames=3)),
    # stacking egocentric views from four ego/allo dicts & stacking them
    # together
    ('LoRes4E',
     functools.partial(lores_ea_entry_point,
                       small_res=(96, 96),
                       allo_frames=0,
                       ego_frames=4)),
    # stacking four allocentric views instead
    ('LoRes4A',
     functools.partial(lores_ea_entry_point,
                       small_res=(96, 96),
                       allo_frames=4,
                       ego_frames=0)),
    # stacking values of four dicts together to produce a single new dict
    ('LoResStack',
     functools.partial(lores_stack_entry_point, small_res=(96, 96), frames=4)),
    # stacking egocentric views from four ego/allo dicts & stacking them
    # together
    ('LoResCHW4E',
     functools.partial(lores_ea_entry_point,
                       small_res=(96, 96),
                       allo_frames=0,
                       ego_frames=4,
                       channels_first=True)),
])
_ENV_NAME_RE = re.compile(
    r'^(?P<name_prefix>[^-]+)(?P<demo_test_spec>-(Demo|Test[^-]*))'
    r'(?P<env_name_suffix>(-[^-]+)*)(?P<version_suffix>-v\d+)$')
_REGISTERED = False
# this will be filled in later
DEMO_ENVS_TO_TEST_ENVS_MAP = collections.OrderedDict()
AVAILABLE_PREPROCESSORS = [key for key in DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS]


class EnvName:
    """Convenience class for parsing environment names. All environment names
    look like this (per _ENV_NAME_RE):

        <name_prefix>-<demo_test_spec>[-<suffix>]-<version_suffix>

    Where:
        - name_prefix identifies the environment class (e.g. MoveToCorner).
        - demo_test_spec is either 'demo' for the demonstration environment, or
          'Test<description>' for test environments, where the description is
          something like 'Pose' or 'Colour' or 'All', depending on what aspects
          of the environment are randomised.
        - suffix is usually for indicating a type of postprocessing (e.g.
          -LoResStack for stacked, scaled-down frames). Not always present.
        - version_suffix is -v0, -v1, etc. etc."""
    def __init__(self, env_name):
        match = _ENV_NAME_RE.match(env_name)
        if match is None:
            raise ValueError(
                "env name '{env_name}' does not match _ENV_NAME_RE spec")
        groups = match.groupdict()
        self.env_name = env_name
        self.name_prefix = groups['name_prefix']
        self.demo_test_spec = groups['demo_test_spec']
        self.env_name_suffix = groups['env_name_suffix']
        self.version_suffix = groups['version_suffix']
        self.demo_env_name = self.name_prefix + '-Demo' \
            + self.env_name_suffix + self.version_suffix
        self.is_test = self.demo_test_spec.startswith('-Test')
        if not self.is_test:
            assert self.demo_env_name == self.env_name, \
                (self.demo_env_name, self.env_name)


def register_envs():
    """Register all default environments for this benchmark suite."""
    global _REGISTERED, DEMO_ENVS_TO_TEST_ENVS_MAP
    if _REGISTERED:
        return False
    _REGISTERED = True

    common_kwargs = dict(res_hw=DEFAULT_RES,
                         fps=8,
                         phys_steps=10,
                         phys_iter=10)

    # remember 100 frames is ~12.5s at 8fps
    mtc_ep_len = 80
    move_to_corner_variants = [
        (MoveToCornerEnv, mtc_ep_len, '-Demo', {
            'rand_shape_colour': False,
            'rand_shape_type': False,
            'rand_poses': False,
            'rand_dynamics': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestColour', {
            'rand_shape_colour': True,
            'rand_shape_type': False,
            'rand_poses': False,
            'rand_dynamics': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestShape', {
            'rand_shape_colour': False,
            'rand_shape_type': True,
            'rand_poses': False,
            'rand_dynamics': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestJitter', {
            'rand_shape_colour': False,
            'rand_shape_type': False,
            'rand_poses': True,
            'rand_dynamics': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestDynamics', {
            'rand_shape_colour': False,
            'rand_shape_type': False,
            'rand_poses': False,
            'rand_dynamics': True,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestAll', {
            'rand_shape_colour': True,
            'rand_shape_type': True,
            'rand_poses': True,
            'rand_dynamics': True,
        }),
    ]

    mtr_ep_len = 40
    move_to_region_variants = [
        (MoveToRegionEnv, mtr_ep_len, '-Demo', {
            'rand_poses_minor': False,
            'rand_poses_full': False,
            'rand_goal_colour': False,
            'rand_dynamics': False,
        }),
        (MoveToRegionEnv, mtr_ep_len, '-TestJitter', {
            'rand_poses_minor': True,
            'rand_poses_full': False,
            'rand_goal_colour': False,
            'rand_dynamics': False,
        }),
        (MoveToRegionEnv, mtr_ep_len, '-TestColour', {
            'rand_poses_minor': False,
            'rand_poses_full': False,
            'rand_goal_colour': True,
            'rand_dynamics': False,
        }),
        (MoveToRegionEnv, mtr_ep_len, '-TestLayout', {
            'rand_poses_minor': False,
            'rand_poses_full': True,
            'rand_goal_colour': False,
            'rand_dynamics': False,
        }),
        (MoveToRegionEnv, mtr_ep_len, '-TestDynamics', {
            'rand_poses_minor': False,
            'rand_poses_full': False,
            'rand_goal_colour': False,
            'rand_dynamics': True,
        }),
        (MoveToRegionEnv, mtr_ep_len, '-TestAll', {  # to stop yapf
            'rand_poses_minor': False,
            # rand_poses_full subsumes rand_poses_minor
            'rand_poses_full': True,
            'rand_goal_colour': True,
            'rand_dynamics': True,
        }),
        # egocentric view shift (probably a bad idea to add this as a test env,
        # rather than a postprocessor or something)
        # (MoveToRegionEnv, mtr_ep_len, '-TestEgo', {
        #     'rand_poses_minor': False,
        #     'rand_poses_full': False,
        #     'rand_goal_colour': False,
        #     'rand_dynamics': False,
        #     'egocentric': True,
        # }),
    ]

    mr_ep_len = 120
    match_regions_variants = [
        (MatchRegionsEnv, mr_ep_len, '-Demo', {
            'rand_target_colour': False,
            'rand_shape_type': False,
            'rand_shape_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MatchRegionsEnv, mr_ep_len, '-TestJitter', {
            'rand_target_colour': False,
            'rand_shape_type': False,
            'rand_shape_count': False,
            'rand_layout_minor': True,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MatchRegionsEnv, mr_ep_len, '-TestColour', {
            'rand_target_colour': True,
            'rand_shape_type': False,
            'rand_shape_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MatchRegionsEnv, mr_ep_len, '-TestShape', {
            'rand_target_colour': False,
            'rand_shape_type': True,
            'rand_shape_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MatchRegionsEnv, mr_ep_len, '-TestLayout', {
            'rand_target_colour': False,
            'rand_shape_type': False,
            'rand_shape_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        # test everything except dynamics
        (MatchRegionsEnv, mr_ep_len, '-TestCountPlus', {
            'rand_target_colour': True,
            'rand_shape_type': True,
            'rand_shape_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        (MatchRegionsEnv, mr_ep_len, '-TestDynamics', {
            'rand_target_colour': False,
            'rand_shape_type': False,
            'rand_shape_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': True,
        }),
        (MatchRegionsEnv, mr_ep_len, '-TestAll', {
            'rand_target_colour': True,
            'rand_shape_type': True,
            'rand_shape_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': True,
        }),
    ]

    ml_ep_len = 180
    make_line_variants = [
        (MakeLineEnv, ml_ep_len, '-Demo', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MakeLineEnv, ml_ep_len, '-TestJitter', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': True,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MakeLineEnv, ml_ep_len, '-TestColour', {
            'rand_colours': True,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MakeLineEnv, ml_ep_len, '-TestShape', {
            'rand_colours': False,
            'rand_shapes': True,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (MakeLineEnv, ml_ep_len, '-TestLayout', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        # test everything except dynamics
        (MakeLineEnv, ml_ep_len, '-TestCountPlus', {
            'rand_colours': True,
            'rand_shapes': True,
            'rand_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        (MakeLineEnv, ml_ep_len, '-TestDynamics', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': True,
        }),
        (MakeLineEnv, ml_ep_len, '-TestAll', {
            'rand_colours': True,
            'rand_shapes': True,
            'rand_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': True,
        }),
    ]

    fd_ep_len = 100
    find_dupe_variants = [
        (FindDupeEnv, fd_ep_len, '-Demo', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FindDupeEnv, fd_ep_len, '-TestJitter', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': True,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FindDupeEnv, fd_ep_len, '-TestColour', {
            'rand_colours': True,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FindDupeEnv, fd_ep_len, '-TestShape', {
            'rand_colours': False,
            'rand_shapes': True,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FindDupeEnv, fd_ep_len, '-TestLayout', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        (FindDupeEnv, fd_ep_len, '-TestCountPlus', {
            'rand_colours': True,
            'rand_shapes': True,
            'rand_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        (FindDupeEnv, fd_ep_len, '-TestDynamics', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': True,
        }),
        (FindDupeEnv, fd_ep_len, '-TestAll', {
            'rand_colours': True,
            'rand_shapes': True,
            'rand_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': True,
        }),
    ]

    fc_ep_len = 60
    fix_colour_variants = [
        (FixColourEnv, fc_ep_len, '-Demo', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FixColourEnv, fc_ep_len, '-TestJitter', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': True,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FixColourEnv, fc_ep_len, '-TestColour', {
            'rand_colours': True,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FixColourEnv, fc_ep_len, '-TestShape', {
            'rand_colours': False,
            'rand_shapes': True,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': False,
        }),
        (FixColourEnv, fc_ep_len, '-TestLayout', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        (FixColourEnv, fc_ep_len, '-TestCountPlus', {
            'rand_colours': True,
            'rand_shapes': True,
            'rand_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': False,
        }),
        (FixColourEnv, fc_ep_len, '-TestDynamics', {
            'rand_colours': False,
            'rand_shapes': False,
            'rand_count': False,
            'rand_layout_minor': False,
            'rand_layout_full': False,
            'rand_dynamics': True,
        }),
        (FixColourEnv, fc_ep_len, '-TestAll', {
            'rand_colours': True,
            'rand_shapes': True,
            'rand_count': True,
            'rand_layout_minor': False,
            'rand_layout_full': True,
            'rand_dynamics': True,
        }),
    ]

    # Long episodes because this is a hard environment. You can have up to 10
    # blocks when doing random layouts, and that takes a human 20-30s to
    # process (so 240/8=30s is just enough time to finish a 10-block run if
    # you know what you're doing).
    cluster_ep_len = 240
    cluster_variants = []
    for cluster_cls in (ClusterColourEnv, ClusterShapeEnv):
        cluster_variants.extend([
            (cluster_cls, cluster_ep_len, '-Demo', {
                'rand_shape_colour': False,
                'rand_shape_type': False,
                'rand_layout_minor': False,
                'rand_layout_full': False,
                'rand_shape_count': False,
                'rand_dynamics': False,
            }),
            (cluster_cls, cluster_ep_len, '-TestJitter', {
                'rand_shape_colour': False,
                'rand_shape_type': False,
                'rand_layout_minor': True,
                'rand_layout_full': False,
                'rand_shape_count': False,
                'rand_dynamics': False,
            }),
            (cluster_cls, cluster_ep_len, '-TestColour', {
                'rand_shape_colour': True,
                'rand_shape_type': False,
                'rand_layout_minor': False,
                'rand_layout_full': False,
                'rand_shape_count': False,
                'rand_dynamics': False,
            }),
            (cluster_cls, cluster_ep_len, '-TestShape', {
                'rand_shape_colour': False,
                'rand_shape_type': True,
                'rand_layout_minor': False,
                'rand_layout_full': False,
                'rand_shape_count': False,
                'rand_dynamics': False,
            }),
            (cluster_cls, cluster_ep_len, '-TestLayout', {
                'rand_shape_colour': False,
                'rand_shape_type': False,
                'rand_layout_minor': False,
                'rand_layout_full': True,
                'rand_shape_count': False,
                'rand_dynamics': False,
            }),
            (cluster_cls, cluster_ep_len, '-TestCountPlus', {
                'rand_shape_colour': True,
                'rand_shape_type': True,
                'rand_layout_minor': False,
                'rand_layout_full': True,
                'rand_shape_count': True,
                'rand_dynamics': False,
            }),
            (cluster_cls, cluster_ep_len, '-TestDynamics', {
                'rand_shape_colour': False,
                'rand_shape_type': False,
                'rand_layout_minor': False,
                'rand_layout_full': False,
                'rand_shape_count': False,
                'rand_dynamics': True,
            }),
            (cluster_cls, cluster_ep_len, '-TestAll', {
                'rand_shape_colour': True,
                'rand_shape_type': True,
                'rand_layout_minor': False,
                'rand_layout_full': True,
                'rand_shape_count': True,
                'rand_dynamics': True,
            }),
        ])

    # collection of ALL env specifications
    env_cls_suffix_kwargs = [
        *cluster_variants,
        *find_dupe_variants,
        *fix_colour_variants,
        *make_line_variants,
        *match_regions_variants,
        *move_to_corner_variants,
        *move_to_region_variants,
    ]

    # register all the envs and record their names
    # TODO: make registration lazy, so that I can do it automatically without
    # importing all the benchmark code (including Pyglet code, etc.).
    registered_env_names = []
    for env_class, env_ep_len, env_suffix, env_kwargs in env_cls_suffix_kwargs:
        base_env_name = env_class.make_name(env_suffix)
        registered_env_names.append(base_env_name)
        gym.register(base_env_name,
                     entry_point=env_class,
                     max_episode_steps=env_ep_len,
                     kwargs={
                         'max_episode_steps': env_ep_len,
                         **common_kwargs,
                         **env_kwargs,
                     })

        for preproc_str, constructor in \
                DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS.items():
            new_name = env_class.make_name(env_suffix + f'-{preproc_str}')
            registered_env_names.append(new_name)
            gym.register(new_name,
                         entry_point=constructor(env_class),
                         max_episode_steps=env_ep_len,
                         kwargs={
                             'max_episode_steps': env_ep_len,
                             **common_kwargs,
                             **env_kwargs,
                         })

    train_to_test_map = {}
    observed_demo_envs = set()
    for name in registered_env_names:
        parsed = EnvName(name)
        if parsed.is_test:
            test_l = train_to_test_map.setdefault(parsed.demo_env_name, [])
            test_l.append(parsed.env_name)
        else:
            observed_demo_envs.add(parsed.env_name)

    # use immutable values
    train_to_test_map = {k: tuple(v) for k, v in train_to_test_map.items()}

    envs_with_test_variants = train_to_test_map.keys()
    assert observed_demo_envs == envs_with_test_variants, \
        "there are some train envs without test envs, or test envs without " \
        "train envs"
    sorted_items = sorted(train_to_test_map.items())
    DEMO_ENVS_TO_TEST_ENVS_MAP.update(sorted_items)

    # Debugging environment: MoveToCorner with a nicely shaped reward function
    # that you can simply do RL on.
    debug_mtc_kwargs = dict(max_episode_steps=mtc_ep_len,
                            kwargs={
                                'debug_reward': True,
                                'max_episode_steps': mtc_ep_len,
                                'rand_shape_colour': False,
                                'rand_shape_type': False,
                                'rand_shape_pose': False,
                                'rand_robot_pose': False,
                                **common_kwargs,
                            })
    debug_mtc_suffix = '-DebugReward'
    gym.register(MoveToCornerEnv.make_name(debug_mtc_suffix),
                 entry_point=MoveToCornerEnv,
                 **debug_mtc_kwargs)
    for preproc_str, constructor in \
            DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS.items():
        gym.register(
            MoveToCornerEnv.make_name(f'{debug_mtc_suffix}-{preproc_str}'),
            entry_point=constructor(MoveToCornerEnv),
            **debug_mtc_kwargs)

    return True
