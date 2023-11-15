import os
import glob
from magical.saved_trajectories import load_demos

def render(dir):
    # CHANGE THIS PATH TO YOUR OWN
    demos = os.path.join(dir, "demo-*.pkl.gz")

    # Load demos
    demo_trajs = list(load_demos(glob.glob(demos)))

    # base names for saving frames and videos, e.g. demo-0.pkl.gz -> demo-0
    base_names = [os.path.basename(x).split('.')[0] for x in glob.glob(demos)]


def render_grid():

def render

if __name__ == "__main__":
    directory = <YOUR DIR>
    render(directory)