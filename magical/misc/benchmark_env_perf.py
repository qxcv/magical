#!/usr/bin/env python3

import cProfile
import datetime

import click
import gym

import magical


def do_eval(env, ntraj):
    for _ in range(ntraj):
        env.reset()
        done = False
        while not done:
            fake_action = env.action_space.sample()
            _, _, done, _ = env.step(fake_action)


@click.command()
@click.option("--ntraj",
              default=100,
              help="number of trajectories to roll out")
@click.option("--seed", default=42, help="env seed")
@click.argument('env_name')
def main(ntraj, env_name, seed):
    """Very simple script to benchmark performance of a particular
    environment. Will simply run a ~hundred trajectories or so and record
    performance."""
    magical.register_envs()
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)

    dtime = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_filename = f'profile-{env_name.lower()}-{dtime}.cprofile'
    print(f"Will write profile to '{out_filename}'")

    cProfile.runctx('do_eval(env, ntraj)',
                    globals(),
                    locals(),
                    filename=out_filename)

    print("Done")


if __name__ == '__main__':
    main()
