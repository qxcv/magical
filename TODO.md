# General code cleanup tasks I want to do at some point

- Abstract all of the environment interaction code to use a common main loop
  class which can handle keyboard I/O (if desired), resetting, saving
  demonstrations, recording videos, etc. This should be useful for all the
  baselines.
- Write a `Sensor` entity that can detect when certain objects enter a specified
  region of the space, and refactor the move-to-corner env to use that entity
  for scoring.
