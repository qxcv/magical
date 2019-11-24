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

    # 250 frames is 20s at 15fps
    ep_len = 200
    env_classes = [
        MoveToCornerEnv,
    ]
    for env_class in env_classes:
        gym.register(env_class.make_name(),
                     entry_point=env_class,
                     max_episode_steps=ep_len,
                     kwargs=common_kwargs)

        # images downsampled to 128x128, four adjacent frames stacked together
        gym.register(env_class.make_name('LoResStack'),
                     entry_point=lores_stack_entry_point(env_class, small_res),
                     max_episode_steps=ep_len,
                     kwargs=common_kwargs)

    return True
