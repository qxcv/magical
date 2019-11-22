"""Tool for demonstrating MILBench and collecting demos."""

import datetime
import gzip
import os
import sys
import time

import click
import dill
import gym
from pyglet.window import key

from milbench import envs


def get_unique_fn(env_name):
    now = datetime.datetime.now()
    time_str = now.strftime('%FT%k:%M:%S')
    return f"demo-{env_name}-{time_str}.dill.gz"


@click.command()
@click.option("--record",
              type=str,
              default=None,
              help="directory to record demos to, if any")
def main(record):
    if record:
        record_dir = os.path.abspath(record)
        print(f"Will record demos to '{record_dir}'")
        os.makedirs(record_dir, exist_ok=True)
        # naughty :-)
        from imitation.util.rollout import _TrajectoryAccumulator
        traj_accum = _TrajectoryAccumulator()

    envs.register()
    env_name = 'ShapePushLoResStack-v0'
    env = gym.make(env_name)

    obs = env.reset()
    if record:
        traj_accum.add_step(0, {"obs": obs})

    # keys that are depressed will end up in key_map
    key_map = key.KeyStateHandler()
    env.viewer.window.push_handlers(key_map)

    # render loop
    spf = 1.0 / env.fps
    saved = False
    try:
        while env.viewer.isopen:
            if key_map[key.R]:
                # drop traj and don't save
                obs = env.reset()
                if record:
                    traj_accum = _TrajectoryAccumulator()
                    traj_accum.add_step(0, {"obs": obs})
                time.sleep(0.5)
                saved = False

            # for limiting FPS
            frame_start = time.time()

            # movement and gripper keys
            action = [0, 0, 0]
            if key_map[key.UP] and not key_map[key.DOWN]:
                action[0] = 1
            elif key_map[key.DOWN] and not key_map[key.UP]:
                action[0] = 2
            if key_map[key.LEFT] and not key_map[key.RIGHT]:
                action[1] = 1
            elif key_map[key.RIGHT] and not key_map[key.LEFT]:
                action[1] = 2
            if key_map[key.SPACE]:
                # close gripper (otherwise it's open)
                action[2] = 1

            obs, rew, done, info = env.step(action)

            if record:
                traj_accum.add_step(0, {
                    "rews": rew,
                    "obs": obs,
                    "acts": action,
                    "infos": info
                })
                if done and not saved:
                    traj = traj_accum.finish_trajectory(0)
                    new_path = os.path.join(record_dir,
                                            get_unique_fn(env_name))
                    pickle_data = {
                        'env_name': env_name,
                        'trajectory': traj,
                    }
                    print(f"Saving trajectory ({len(traj.obs)} obs, "
                          f"{len(traj.acts)} actions, {len(traj.rews)} rews) "
                          f"to '{new_path}'")
                    with gzip.GzipFile(new_path, 'wb') as fp:
                        dill.dump(pickle_data, fp, protocol=4)
                    saved = True

            # render to screen
            env.render(mode='human')

            # wait for next frame
            elapsed = time.time() - frame_start
            if elapsed < spf:
                time.sleep(spf - elapsed)
    finally:
        # once done, close the window
        env.viewer.close()


if __name__ == '__main__':
    try:
        with main.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = main.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
