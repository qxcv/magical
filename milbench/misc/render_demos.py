#!/usr/bin/env python3
"""Render a list of demonstrations as a long, contiguous video."""

import os
import sys

import click
import numpy as np
import skvideo.io as vidio

from milbench.baselines.saved_trajectories import load_demos


@click.command()
@click.option('--out-path', default='demo-video.mp4', help='output video path')
@click.option('--fps', default=32, help='frame rate for rendering')
@click.argument('demo_paths', nargs=-1, required=True)
def main(out_path, fps, demo_paths):
    all_demos = load_demos(demo_paths)
    frames = np.concatenate([d['trajectory'].obs for d in all_demos], axis=0)
    del all_demos
    out_dir = os.path.dirname(out_path)
    if out_dir:
        print(f"Will make dir '{out_dir}' if it does not exist yet")
        os.makedirs(out_dir, exist_ok=True)
    print(f"Writing video to '{out_path}'")
    vidio.vwrite(out_path,
                 frames,
                 outputdict={
                     '-r': str(fps),
                     '-vcodec': 'libx264',
                     '-pix_fmt': 'yuv420p',
                 })
    print("Done!")


if __name__ == '__main__':
    try:
        with main.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = main.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.exit_code)
