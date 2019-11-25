"""Register available envs and create index of train/test mappings."""

import gym
from gym.wrappers import FrameStack, ResizeObservation

from milbench.benchmarks.move_to_corner import MoveToCornerEnv


def lores_stack_entry_point(env_cls, small_res, frames=4):
    def make_lores_stack(**kwargs):
        base_env = env_cls(**kwargs)
        resize_env = ResizeObservation(base_env, small_res)
        stack_env = FrameStack(resize_env, frames)
        return stack_env

    return make_lores_stack


_REGISTERED = False


def register_envs():
    """Register all default environments for this benchmark suite."""
    global _REGISTERED
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

    return True
