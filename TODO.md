# General code cleanup tasks I want to do at some point

- Abstract all of the environment interaction code to use a common main loop
  class which can handle keyboard I/O (if desired), resetting, saving
  demonstrations, recording videos, etc. This should be useful for all the
  baselines.
- Write a `Sensor` entity that can detect when certain objects enter a specified
  region of the space, and refactor the move-to-corner env to use that entity
  for scoring.
- Replace Dill with Cloudpickle (like stable_baselines) or numpy's
  `savez_compressed` throughout the codebase, then remove Dill from deps.
- Move the test() and testall() functions in bc.py into their own rollout script
  in the baselines directory. Those two things shouldn't be immutably attached
  to BC.
- Fix bug in `dagger.py` (or maybe `DAggerTrainer` in imitation) that is causing
  it not to reload policies correctly. When I do `collect()`, it just seems to
  roll out randomly, which is not at all what I want. Can I verify that it's
  loading all of the correct weights? `save_policy()` and `reconstruct_policy()`
  seem to work, so it's evidently just something up with
  `reconstruct_trainer()` and/or `save_trainer()`.
