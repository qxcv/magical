import abc
import collections
import io
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

from magical.benchmarks import DEMO_ENVS_TO_TEST_ENVS_MAP


class EvaluationProtocol(abc.ABC):
    """Common elements of the multi-task evaluation protocol. Users of this
    class provide a way to roll out K trajectories on a Gym environment with a
    given name. The class does aggregation of statistics in a sensible way."""
    _called_init = False

    def __init__(self, demo_env_name, n_rollouts):
        self.n_rollouts = n_rollouts
        self.demo_env_name = demo_env_name
        self.test_env_names = [
            demo_env_name,
            *DEMO_ENVS_TO_TEST_ENVS_MAP[demo_env_name],
        ]
        self._called_init = True

    @property
    @abc.abstractmethod
    def run_id(self):
        """Should produce a string identifier distinguishing the current
        model/algorithm/etc. This might be, e.g., a model snapshot path, or
        even just a name like "Multi-task GAIL". This identifier will be
        available in a `run_id` column of the DataFrame returned by
        `do_eval`."""

    @abc.abstractmethod
    def obtain_scores(self, env_name):
        """This method should perform `self.n_rollouts` rollouts on the
        environment named `env_name`, then return the `eval_score`s from each
        rollout. Should never return _fewer_ than `self.n_rollouts` scores; if
        it produces more scores, then the later ones will be ignored."""

    def do_eval(self, verbose=False):
        if not self._called_init:
            raise ValueError(
                "EvaluationProtocol.__init__() was not called. Did you "
                "include a super().__init__(…) call in your subclass?")

        stats_tuples = []

        for env_name in self.test_env_names:
            scores = self.obtain_scores(env_name)
            if len(scores) < self.n_rollouts:
                raise ValueError(f".obtain_scores() returned only "
                                 f"{len(scores)} scores, but we asked for "
                                 f"{self.n_rollouts} scores")
            if len(scores) > self.n_rollouts:
                # We want to be consistent. Extra scores will reduce width of
                # CI, but if some runs have more scores than others then user
                # might get confused and think that the configurations with
                # more runs have lower population variance (they may not).
                warnings.warn(
                    f"Asked for {self.n_rollouts} scores but got "
                    f"{len(scores)} scores instead. Will truncate to only "
                    f"consider the first {self.n_rollouts} scores.")
                scores = scores[:self.n_rollouts]
            mean = np.mean(scores)
            interval = DescrStatsW(scores).tconfint_mean(0.05, 'two-sided')
            std = np.std(scores, ddof=1)
            stats_tuples.append(
                (env_name, mean, interval[0], interval[1], std))

        records = [
            collections.OrderedDict([
                ('demo_env', self.demo_env_name),
                ('test_env', env_name),
                ('mean_score', mean_score),
                ('ci95_lower', ci95_lower),
                ('ci95_upper', ci95_upper),
                ('std_score', std),
                ('run_id', self.run_id),
            ]) for env_name, mean_score, ci95_lower, ci95_upper, std in
            stats_tuples
        ]
        frame = pd.DataFrame.from_records(records)

        if verbose:
            print(f"Final mean scores for '{self.run_id}':")
            relevant_stats = [
                'test_env',
                'mean_score',
                'ci95_lower',
                'ci95_upper',
            ]
            print(frame[relevant_stats])

        return frame


def latexify_results(eval_data, id_column='run_id'):
    """Take a data frame produced by `EvaluationProtocol.eval_data()` and
    produce a LaTeX table of results for this method. Will use the `run_id`
    column as an algorithm name (or to get algorithm names, if there's more
    than one algorithm present in the given data). You can override that by
    specifying the `id_column` keyword argument."""

    # Each column of the LaTeX table corresponds to a particular evaluation
    # environment, while each row corresponds to an algorithm. In contrast,
    # each row of the given Pandas frame is represents a series of rollouts by
    # one particular algorithm on one particular test configuration.

    test_envs = eval_data['test_env'].unique()
    col_names = [r'\textbf{%s}' % e for e in test_envs]
    alg_names = eval_data[id_column].unique()

    # write to buffer so we can use print()
    fp = io.StringIO()

    # prefix is just LaTeX table setup
    print(r"\centering", file=fp)
    print(r"\begin{tabular}{l@{\hspace{1em}}%s}" % ("c" * len(col_names)),
          file=fp)
    print(r"\toprule", file=fp)

    # first line: env names
    print(r'\textbf{Randomisation} & ', end='', file=fp)
    print(' & '.join(col_names), end='', file=fp)
    print('\\\\', file=fp)
    print(r'\midrule', file=fp)

    # next lines: actual results
    for alg_name in alg_names:
        alg_mask = eval_data[id_column] == alg_name
        stat_parts = []
        for env_name in test_envs:
            full_mask = alg_mask & (eval_data['test_env'] == env_name)
            relevant_rows = list(eval_data[full_mask].iterrows())
            if len(relevant_rows) != 1:
                raise ValueError(
                    f"got {len(relevant_rows)} rows corresponding to "
                    f"{id_column}={alg_name} and test_env={env_name}, but "
                    f"expected one (maybe IDs in column {id_column} aren't "
                    f"unique?)")
            (_, row), = relevant_rows
            std = row['std_score']
            stat_parts.append(f'{row["mean_score"]:.2f} ($\\pm$ {std:.2f})')
        print(r'\textbf{%s} & ' % alg_name, end='', file=fp)
        print(' & '.join(stat_parts), end='', file=fp)
        print('\\\\', file=fp)
        print(r'\bottomrule', file=fp)
        print(r'\end{tabular}', file=fp)

    return fp.getvalue()
