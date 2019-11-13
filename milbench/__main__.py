"""Interactive demo for MILBench."""

import sys
import time

import click

import milbench.gym_render as r

# Add options by including decorators like this after click.command:


@click.command()
@click.option("--width", default=500, help="width of screen")
@click.option("--height", default=500, help="height of screen")
@click.option("--fps", default=30, help="render framerate")
def main(width, height, fps):
    viewer = r.Viewer(width, height)
    viewer.set_bounds(left=-1, right=1, bottom=-1, top=1)
    circle = r.make_circle(radius=0.1, res=100)
    circle.set_color(0.2, 0.2, 0.2)
    viewer.add_geom(circle)
    spf = 1.0 / fps
    while True:
        frame_start = time.time()
        viewer.render()
        elapsed = time.time() - frame_start
        if elapsed < spf:
            time.sleep(spf - elapsed)


if __name__ == '__main__':
    try:
        with main.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = main.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
