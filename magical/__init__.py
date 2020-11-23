# flake8: noqa
from magical.benchmarks import (AVAILABLE_PREPROCESSORS,
                                DEMO_ENVS_TO_TEST_ENVS_MAP, register_envs)
from magical.reference_demos import try_download_demos  # noqa: F401
from magical.saved_trajectories import (load_demos,
                                        preprocess_demos_with_wrapper,
                                        splice_in_preproc_name)
from magical.version import __version__
