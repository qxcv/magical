# General code cleanup tasks I want to do at some point

- Abstract all of the environment interaction code to use a common main loop
  class which can handle keyboard I/O (if desired), resetting, saving
  demonstrations, recording videos, etc. This should be useful for all the
  baselines.
- Refactor the viewer so that it can do a separate render at a higher
  resolution, but for the human demonstrator only.
- Refactor state preprocessors so that demos aren't tied to a particular
  preprocessing of the environment.
- Move the test() and testall() functions in bc.py into their own rollout script
  in the baselines directory. Those two things shouldn't be immutably attached
  to BC.
- Fix bug in `dagger.py` (or maybe `DAggerTrainer` in imitation) that is causing
  it not to reload policies correctly. When I do `collect()`, it just seems to
  roll out randomly, which is not at all what I want. Can I verify that it's
  loading all of the correct weights? `save_policy()` and `reconstruct_policy()`
  seem to work, so it's evidently just something up with
  `reconstruct_trainer()` and/or `save_trainer()`.
- In `Cluster*` environments, consider changing the random layout function to
  avoid placing blocks of similar colour or type too close to one another (e.g.
  within ~4 shape radii). That should minimise the number of accidental clusters
  that the algorithm builds, at the cost of making placement expensive when
  there are many shapes.
