"""Interactive demo for MILBench."""

import sys
import time
import math

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
    space.iterations = 10

    # set up robot and arena
    robot_rad = 0.18
    shape_rad = robot_rad * 2 / 3
    robot = en.Robot(radius=robot_rad,
                     init_pos=(0, 0),
                     init_angle=math.pi / 9,
                     mass=1.0)
    arena = en.ArenaBoundaries(left=-1.0, right=1.0, bottom=-1.0, top=1.0)
    square = en.Shape(shape_type=en.ShapeType.SQUARE,
                      colour_name='red',
                      shape_size=shape_rad,
                      init_pos=(0.4, -0.3),
                      init_angle=0.13 * math.pi)
    circle = en.Shape(shape_type=en.ShapeType.CIRCLE,
                      colour_name='yellow',
                      shape_size=shape_rad,
                      init_pos=(-0.7, -0.5),
                      init_angle=-0.5 * math.pi)
    triangle = en.Shape(shape_type=en.ShapeType.TRIANGLE,
                        colour_name='green',
                        shape_size=shape_rad,
                        init_pos=(-0.5, 0.25),
                        init_angle=0.05 * math.pi)
    pentagon = en.Shape(shape_type=en.ShapeType.PENTAGON,
                        colour_name='blue',
                        shape_size=shape_rad,
                        init_pos=(0.4, 0.3),
                        init_angle=0.8 * math.pi)
    entities = [circle, square, triangle, pentagon, robot, arena]

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
        action_flag = en.RobotAction.NONE
        if key_map[key.LEFT]:
            action_flag |= en.RobotAction.LEFT
        if key_map[key.RIGHT]:
            action_flag |= en.RobotAction.RIGHT
        if key_map[key.UP]:
            action_flag |= en.RobotAction.UP
        if key_map[key.DOWN]:
            action_flag |= en.RobotAction.DOWN
        # holding down the space key closes the gripper, and releasing it opens
        # the gripper (maybe I can improve these ergonomics somehow?)
        if key_map[key.SPACE]:
            action_flag |= en.RobotAction.CLOSE
        else:
            action_flag |= en.RobotAction.OPEN
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
