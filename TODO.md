# General code cleanup tasks I want to do before release

Urgent things that must be done before running more baselines:

- Port to EGL.
- Make image resizing run in a shader. Ideally it should be possible to render
  at half size with roughly ~0 added cost (right now PIL resize calls are a
  major bottleneck when doing rollouts).
- Benchmark to make sure new code is actually faster. Ultimately I should be
  getting a few hundred frames a second with one core on my laptop (note that
  glxgears gets ~2,500 FPS with geometry of similar complexity, but no physics).

Cleanliness things that should be done before release:

- Remove all baselines and other code that I can get away with omitting.
- Use American spellings of everything. In particular, 'colour' -> 'color'.
- Ensure that environments aren't imported when MAGICAL is. In particular, make
  sure that MAGICAL can be imported without loading Pyglet (some parts of Pyglet
  make new windows when imported, which is annoying). Should be at least
  possible to register environments and load demonstrations without touching any
  graphics code.
- Add docstrings to just about everything.
- Make the eval code easier to use. Should not need to do any inversion of
  control just to compute confidence intervals (that's kind of ridiculous).
- Write a more detailed README that includes examples of how to use different
  parts of the code. Particular attention should be paid to loading
  demonstrations; instantiating different variants of the environments; and
  automatically evaluating environments.
- Write a demo that shows how to use MAGICAL with imitation (both BC and GAIL,
  ideally).
- In `Cluster*` environments, consider changing the random layout function to
  avoid placing blocks of similar colour or type too close to one another (e.g.
  within ~4 shape radii). That should minimise the number of accidental clusters
  that the algorithm builds, at the cost of making placement expensive when
  there are many shapes.

Optional, but nice to have:

- Abstract all of the environment interaction code to use a common main loop
  class which can handle keyboard I/O (if desired), resetting, saving
  demonstrations, recording videos, etc. This should be useful for all the
  baselines.
- Refactor the viewer so that it can do a separate render at a higher
  resolution, but for the human demonstrator only.
