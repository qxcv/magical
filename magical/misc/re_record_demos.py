#!/usr/bin/env python3
"""Convert my old demonstrations so that they use the new action format."""
import gzip
import os
import sys

import click
import cloudpickle
import gym

from magical.baselines.saved_trajectories import (  # this appeases isort
    MAGICALTrajectory, load_demos)
from magical.benchmarks import register_envs

SUFFIX = '.pkl.gz'


@click.command()
@click.option('--out-dir',
              default='demos-converted',
              help='out directory to write to')
@click.argument('target_env_name')
@click.argument('demo_paths', nargs=-1, required=True)
def main(out_dir, target_env_name, demo_paths):
    demo_itr = load_demos(demo_paths)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    register_envs()
    # don't bother seeding env; we'll just hope it's deterministic (which is
    # true of demo variants of magical envs)
    env = gym.make(target_env_name)
    try:
        for old_path, demo_dict in zip(demo_paths, demo_itr):
            print(f"Working on demo at '{old_path}'")
            actions = demo_dict['trajectory'].acts
            observations = [env.reset()]
            rews = []
            infos = []
            for act_num, act in enumerate(actions, 1):
                obs, rew, done, info = env.step(act)
                observations.append(obs)
                rews.append(rew)
                infos.append(info)
                if done:
                    if act_num != len(actions):
                        print(f"Got 'done' at action {act_num}/{len(actions)}")
                    break
            if not done:
                print("Ran out of actions, but env still isn't done (???)")
                print("Going to pad with noops")
                pad_acts = 0
                while not done:
                    obs, rew, done, info = env.step(0)
                    observations.append(obs)
                    rews.append(rew)
                    infos.append(info)
                    pad_acts += 1
                print(f"Padded to {pad_acts} noops")
            score = info.get('eval_score')
            print(f"Traj done. sum(rew) is {sum(rews)}, score is {score}")
            new_traj = MAGICALTrajectory(acts=actions,
                                         obs=observations,
                                         rews=rews,
                                         infos=infos)
            new_dict = {
                'trajectory': new_traj,
                'score': score,
                'env_name': target_env_name,
            }
            old_bn = os.path.basename(old_path)
            new_bn = old_bn.replace(demo_dict['env_name'], target_env_name)
            new_bn = new_bn.replace(' ', '0')  # HACK (e.g. for 'T 9:30:12')
            new_path = os.path.join(out_dir, new_bn)
            print(f"Saving trajectory to '{new_path}'")
            with gzip.GzipFile(new_path, 'wb') as fp:
                cloudpickle.dump(new_dict, fp)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        with main.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = main.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.exit_code)
