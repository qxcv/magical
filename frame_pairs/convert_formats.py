#!/usr/bin/env python3
"""Convert all the different annotation formats into the one that Alex was
using.

Alex's format has a single metadat a file per task and horizon. For example, in
Task1/20-frame-pairs/alex_ground_truth.json, the JSON file looks like this:

```json
{
  "1": "Move green circle B4 to green special area SA1",
  "2": "Move green circle B1 to green special area SA1",
  "3": "Move green square B3 to green special area SA1",
  "4": "Move blue circle B2 to blue special area SA1",
  "5": "Move blue circle B1 to blue special area SA1",
  "6": "Move pink square B3 to pink special area SA1",
  "7": "Release pink square B1 inside the pink special area SA1 and back away",
  "8": "Move green square B4 to green special area SA1",
  "9": "Move green circle B4 to green special area SA1",
  "10": "Move green square B5 to green special area SA1"
}
```

The keys are the frame pair numbers, and the values are the instructions.

Elvis and Sam instead had one file per frame pair. e.g.
Task1/20-frame-pairs/pair1/elvis_ground_truth.json looks like this:

```json
{
  "subgoal": "Move to bottom and pick up block B1"
}
```

(similar for sam_ground_truth.json)

Olivia uses a different non-JSON format at the horizon level. e.g.
Task1/20-frame-pairs/olivia-ground-truth.json looks like this:

```
SUBGOAL: Pick up block B1.
SUBGOAL: Move block B1 to the top-left corner.
SUBGOAL: Move block B1 to the top-left corner.
SUBGOAL: Move block B1 to the left edge of the arena.
SUBGOAL: Carry block B1 to the top-left corner.
SUBGOAL: Carry block B1 to the center of the arena.
SUBGOAL: Move block B1 to the top-left corner of the arena.
SUBGOAL: Move to the center of the arena.
SUBGOAL: Release block B1 and move to the center.
SUBGOAL: Move to block B1 and pick it up.
```

Each line is for a different pair.
"""
import json
import os
from typing import Literal
import re


def convert_olivia_format(olivia_src: str, olivia_dist: str) -> None:
    """Convert Olivia's non-JSON format to Alex's JSON format."""
    print(f"Converting {olivia_src} to {olivia_dist}")
    lines = []
    with open(olivia_src, "r") as src:
        for line in src:
            line = line.strip()
            if line.startswith("SUBGOAL:"):
                lines.append(line[len("SUBGOAL:") :].strip())
    if len(lines) != 10:
        raise ValueError(f"Expected 10 lines, got {len(lines)} in {olivia_src}")

    with open(olivia_dist, "w") as dist:
        json.dump({i + 1: line for i, line in enumerate(lines)}, dist, indent=2, sort_keys=True)

    # remove the original
    os.remove(olivia_src)


def elvis_sam_convert_format(
    task_horizon_dir: str, person: Literal["elvis", "sam"]
) -> None:
    """Iterate through all the pair<N>s in the task/horizon dir and convert the
    annos for the given person to Alex's format."""
    pair_dir_re = re.compile(r"^pair(?P<num>\d+)$")
    subgoals = {}
    for pair_dir in os.listdir(task_horizon_dir):
        # look at the pair<N> dirs
        match = pair_dir_re.match(pair_dir)
        if not match:
            continue
        num_marker = match.group("num")

        pair_dir = os.path.join(task_horizon_dir, pair_dir)
        person_src = os.path.join(pair_dir, f"{person}_ground_truth.json")
        with open(person_src, "r") as src:
            subgoals[num_marker] = json.load(src)["subgoal"]

        # remove the original
        print(f"Removing {person_src}")
        os.remove(person_src)

    dst_path = os.path.join(task_horizon_dir, f"{person}_ground_truth.json")
    print(f"Writing to {dst_path}")
    with open(dst_path, "w") as dst:
        json.dump(subgoals, dst, indent=2, sort_keys=True)


def main():
    """Convert all the different annotation formats into the one that Alex was
    using."""
    task_dirs = [
        "Task1/10-frame-pairs",
        "Task1/20-frame-pairs",
        "Task1/40-frame-pairs",
        "Task1/80-frame-pairs",
        "Task2/10-frame-pairs",
        "Task2/20-frame-pairs",
        "Task2/50-frame-pairs",
        "Task2/100-frame-pairs",
        "Task3/10-frame-pairs",
        "Task3/20-frame-pairs",
        "Task3/60-frame-pairs",
        "Task3/180-frame-pairs",
    ]
    for task_dir in task_dirs:
        # convert Olivia's file
        olivia_src = os.path.join(task_dir, "olivia-ground-truth.json")
        olivia_dst = os.path.join(task_dir, "olivia_ground_truth.json")
        convert_olivia_format(olivia_src, olivia_dst)

        # convert Elvis's file
        elvis_sam_convert_format(task_dir, "elvis")

        # convert Sam's file
        elvis_sam_convert_format(task_dir, "sam")

    print("Done!")

if __name__ == "__main__":
    main()
