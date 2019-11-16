"""Interactive demo for MILBench."""

import sys
import time

import click
from pyglet.window import key
import pymunk

import milbench.gym_render as r
import milbench.entities as en


@click.command()
@click.option("--width", default=500, help="width of screen")
@click.option("--height", default=500, help="height of screen")
@click.option("--fps", default=30, help="render framerate")
def main(width, height, fps):
    # set up physics
    space = pymunk.Space()
    # no gravity, this is top-down
    space.gravity = (0.0, 0.0)
    # TODO: remove damping. Seems kind of hacky.
    space.damping = 0.95

    # set up robot and arena
    robot = en.Robot(radius=0.1, init_pos=(0, 0), init_angle=0.0, mass=1.0)
    arena = en.ArenaBoundaries(left=-1.0, right=1.0, bottom=-1.0, top=1.0)
    entities = [robot, arena]

    # set up graphics
    viewer = r.Viewer(width, height)
    viewer.set_bounds(left=arena.left,
                      right=arena.right,
                      bottom=arena.bottom,
                      top=arena.top)

    for ent in entities:
        ent.setup(viewer, space)

    # keys that are depressed will end up in key_map
    key_map = key.KeyStateHandler()
    viewer.window.push_handlers(key_map)

    # render loop
    spf = 1.0 / fps
    while viewer.isopen:
        # for limiting FPS
        frame_start = time.time()

        # handle keyboard input for actions
        action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        action_flag = en.RobotAction.NONE
        for action_name in action_names:
            if key_map[getattr(key, action_name)]:
                action_flag |= getattr(en.RobotAction, action_name)
        robot.set_action(action_flag)

        # step forward physics
        phys_steps = 10
        dt = spf / phys_steps
        for i in range(phys_steps):
            for ent in entities:
                ent.update(dt)
            space.step(dt)

        # render
        for ent in entities:
            ent.pre_draw()
        viewer.render()

        # wait for next frame
        elapsed = time.time() - frame_start
        if elapsed < spf:
            time.sleep(spf - elapsed)

    # once done, close the window
    viewer.close()


if __name__ == '__main__':
    try:
        with main.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = main.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
