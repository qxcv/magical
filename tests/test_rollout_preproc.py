"""Test rollouts in every environment."""
import gym
import pytest

import magical

N_ROLLOUTS = 2

magical.register_envs()


def test_registered_envs():
    # make sure we registered at least some environments
    assert len(magical.ALL_REGISTERED_ENVS) > 8


@pytest.mark.parametrize('env_name', magical.ALL_REGISTERED_ENVS)
def test_rollouts(env_name):
    """Simple smoke test to make sure environments can roll out trajectories of
    the right length."""
    env = gym.make(env_name)
    try:
        env.seed(7)
        env.action_space.seed(42)
        env.reset()
        for _ in range(N_ROLLOUTS):
            done = False
            traj_len = 0
            while not done:
                action = env.action_space.sample()
                obs, rew, done, info = env.step(action)
                traj_len += 1
            assert traj_len == env.max_episode_steps
            env.reset()
    finally:
        env.close()
