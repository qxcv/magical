#!/usr/bin/env python3
"""Convert my old demonstrations so that they use the new action format."""
import gzip
import os

import cloudpickle
import numpy as np

from magical.baselines.saved_trajectories import (  # this appeases isort
    MAGICALTrajectory, load_demos)
import magical.entities as en

SOURCE_TREE = "demos/"
DEST_TREE = "demos-new/"
SUFFIX = '.pkl.gz'
AC_FLAGS_UD = [
    en.RobotAction.NONE,
    en.RobotAction.UP,
    en.RobotAction.DOWN,
]
AC_FLAGS_LR = [
    en.RobotAction.NONE,
    en.RobotAction.LEFT,
    en.RobotAction.RIGHT,
]
AC_FLAGS_GRIP = [en.RobotAction.OPEN, en.RobotAction.CLOSE]
AC_FLAGLISTS = [AC_FLAGS_UD, AC_FLAGS_LR, AC_FLAGS_GRIP]


def main():
    # dest_tree is where we save things, source_tree is where we read things
    dest_tree = os.path.abspath(DEST_TREE)
    source_tree = os.path.abspath(SOURCE_TREE)
    os.makedirs(dest_tree, exist_ok=True)

    for dirpath, _, filenames in os.walk(source_tree):
        for filename in filenames:
            if not filename.endswith(SUFFIX):
                continue

            # load old trajectory
            source_path = os.path.join(dirpath, filename)
            demo_dict, = load_demos([source_path])
            demo_traj = demo_dict['trajectory']

            # convert actions
            old_acts = demo_traj.acts
            new_acts_list = []
            for act in old_acts:
                assert len(act) == len(AC_FLAGLISTS)
                flags = tuple(flaglist[flag_num]
                              for flag_num, flaglist in zip(act, AC_FLAGLISTS))
                action_id = en.FLAGS_TO_ACTION_ID[flags]
                new_acts_list.append(action_id)
            new_act_tensor = np.asarray(new_acts_list, dtype=np.int32)

            # splice the new actions into a new demonstration with the right
            # trajectory
            new_demo_traj = MAGICALTrajectory(acts=new_act_tensor,
                                              obs=demo_traj.obs,
                                              rews=demo_traj.rews,
                                              infos=demo_traj.infos)
            new_demo_dict = {k: v for k, v in demo_dict.items()}
            new_demo_dict['trajectory'] = new_demo_traj

            # save it in the appropriate place
            prefix_dir = os.path.relpath(dirpath, start=source_tree)
            dest_dir = os.path.join(dest_tree, prefix_dir)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
            print(f"'{source_path}' -> '{dest_path}'")
            with gzip.GzipFile(dest_path, 'wb') as fp:
                cloudpickle.dump(new_demo_dict, fp)


if __name__ == '__main__':
    main()
