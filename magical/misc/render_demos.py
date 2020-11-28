#!/usr/bin/env python3
"""Render a list of demonstrations as a long, contiguous video."""

import collections
import os
import sys

import click
import numpy as np
import skvideo.io as vidio

from magical.saved_trajectories import load_demos


def get_frames(demo):
    frame_dicts = demo['trajectory'].obs
    keys = sorted(frame_dicts[0].keys())
    by_key = collections.defaultdict(list)
    for frame_dict in frame_dicts:
        for key in keys:
            by_key[key].append(frame_dict[key])
    # concatenate along time axis
    by_key = {k: np.stack(l, axis=0) for k, l in by_key.items()}
    # concatenate along width axis to show everything
    frames = np.concatenate([by_key[key] for key in keys], axis=2)
    return frames


@click.command()
@click.option('--out-path', default='demo-video.mp4', help='output video path')
@click.option('--fps', default=32, help='frame rate for rendering')
@click.argument('demo_paths', nargs=-1, required=True)
def main(out_path, fps, demo_paths):
    all_demos = load_demos(demo_paths)
    frame_segments = map(get_frames, all_demos)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        print(f"Will make dir '{out_dir}' if it does not exist yet")
        os.makedirs(out_dir, exist_ok=True)
    print(f"Writing video to '{out_path}'")
    writer = vidio.FFmpegWriter(out_path,
                                outputdict={
                                    '-r': str(fps),
                                    '-vcodec': 'libx264',
                                    '-pix_fmt': 'yuv420p',
                                })
    nframes = 0
    for frame_segment in frame_segments:
        print(f"Writing frame segment ({nframes} frames written so far)")
        for frame in frame_segment:
            writer.writeFrame(frame)
            nframes += 1
    print(f"Done! ({nframes} frames written in total)")


if __name__ == '__main__':
    try:
        with main.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = main.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.exit_code)
