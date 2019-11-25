"""Register available envs and create index of train/test mappings."""

import collections
import re

import gym
from gym.wrappers import FrameStack, ResizeObservation

from milbench.benchmarks.move_to_corner import MoveToCornerEnv

__all__ = [
    'DEMO_ENVS_TO_TEST_ENVS_MAP',
    'MoveToCornerEnv',
    'register_envs',
]


def lores_stack_entry_point(env_cls, small_res, frames=4):
    def make_lores_stack(**kwargs):
        base_env = env_cls(**kwargs)
        resize_env = ResizeObservation(base_env, small_res)
        stack_env = FrameStack(resize_env, frames)
        return stack_env

    return make_lores_stack


_ENV_NAME_RE = re.compile(
    r'^(?P<name_prefix>[^-]+)(?P<demo_test_spec>-(Demo|Test[^-]*))'
    r'(?P<env_name_suffix>(-[^-]+)*)(?P<version_suffix>-v\d+)$')
_REGISTERED = False
# this will be filled in later
DEMO_ENVS_TO_TEST_ENVS_MAP = collections.OrderedDict()


class _EnvName:
    def __init__(self, env_name):
        match = _ENV_NAME_RE.match(env_name)
        if match is None:
            raise ValueError(
                "env name '{env_name}' does not match _ENV_NAME_RE spec")
        groups = match.groupdict()
        self.env_name = env_name
        name_prefix = groups['name_prefix']
        demo_test_spec = groups['demo_test_spec']
        env_name_suffix = groups['env_name_suffix']
        version_suffix = groups['version_suffix']
        self.demo_env_name = name_prefix + '-Demo' + env_name_suffix \
            + version_suffix
        self.is_test = demo_test_spec.startswith('-Test')
        if not self.is_test:
            assert self.demo_env_name == self.env_name, \
                (self.demo_env_name, self.env_name)


def register_envs():
    """Register all default environments for this benchmark suite."""
    global _REGISTERED, DEMO_ENVS_TO_TEST_ENVS_MAP
    if _REGISTERED:
        return False
    _REGISTERED = True

    default_res = (256, 256)
    small_res = (96, 96)
    common_kwargs = dict(res_hw=default_res,
                         fps=15,
                         phys_steps=10,
                         phys_iter=10)

    # note on episode lengths: 250 frames is ~17s at 15fps; 200 frames is ~13s
    mtc_ep_len = 200
    move_to_corner_variants = [
        (MoveToCornerEnv, mtc_ep_len, '-Demo', {
            'rand_shape_colour': False,
            'rand_shape_type': False,
            'rand_shape_pose': False,
            'rand_robot_pose': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestShapeColour', {
            'rand_shape_colour': True,
            'rand_shape_type': False,
            'rand_shape_pose': False,
            'rand_robot_pose': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestShapeType', {
            'rand_shape_colour': False,
            'rand_shape_type': True,
            'rand_shape_pose': False,
            'rand_robot_pose': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestShapePose', {
            'rand_shape_colour': False,
            'rand_shape_type': False,
            'rand_shape_pose': True,
            'rand_robot_pose': False,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestRobotPose', {
            'rand_shape_colour': False,
            'rand_shape_type': False,
            'rand_shape_pose': False,
            'rand_robot_pose': True,
        }),
        (MoveToCornerEnv, mtc_ep_len, '-TestAll', {
            'rand_shape_colour': True,
            'rand_shape_type': True,
            'rand_shape_pose': True,
            'rand_robot_pose': True,
        }),
    ]

    # collection of ALL env specifications
    env_cls_suffix_kwargs = [
        *move_to_corner_variants,
    ]

    # register all the envs and record their names
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

        # images downsampled to 128x128, four adjacent frames stacked together
        lores_env_name = env_class.make_name(env_suffix + '-LoResStack')
        registered_env_names.append(lores_env_name)
        gym.register(lores_env_name,
                     entry_point=lores_stack_entry_point(env_class, small_res),
                     max_episode_steps=env_ep_len,
                     kwargs={
                         'max_episode_steps': env_ep_len,
                         **common_kwargs,
                         **env_kwargs,
                     })

    train_to_test_map = {}
    observed_demo_envs = set()
    for name in registered_env_names:
        parsed = _EnvName(name)
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

    return True
