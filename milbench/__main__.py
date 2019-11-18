"""Interactive demo for MILBench."""

import sys
import time
import gym

import click
from pyglet.window import key

from milbench import envs


@click.command()
def main():
    envs.register()
    env = gym.make('ShapePushingEnv-v0')
    env.reset()

    # keys that are depressed will end up in key_map
    key_map = key.KeyStateHandler()
    env.viewer.window.push_handlers(key_map)

    # render loop
    spf = 1.0 / env.fps
    try:
        while env.viewer.isopen:
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
