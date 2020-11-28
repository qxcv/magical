# noqa: F401
from magical.benchmarks import (  # noqa: F401
    ALL_REGISTERED_ENVS, AVAILABLE_PREPROCESSORS, DEMO_ENVS_TO_TEST_ENVS_MAP,
    register_envs)
from magical.reference_demos import try_download_demos  # noqa: F401
from magical.saved_trajectories import (  # noqa: F401
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from magical.version import __version__  # noqa: F401
