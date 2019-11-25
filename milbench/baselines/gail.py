"""Generative Adversarial Imitation Learning (GAIL)."""
import sys

import click
from imitation.algorithms.adversarial import AdversarialTrainer

from milbench.benchmarks import register_envs


@click.group()
def cli():
    # TODO: move the scratch dir up here so I can have an actual training dir
    # for each training run I do.
    pass


@cli.command()
@click.option("--scratch",
              default="scratch/gail",
              help="directory to save snapshots in")
@click.argument("demos", nargs=-1, required=True)
def train():
    # Constructor signature:
    # AdversarialTrainer(
    #     venv, gen_policy, discrim, expert_demos,
    #     disc_opt_cls=tf.train.AdamOptimizer, disc_opt_kwargs={},
    #     n_disc_samples_per_buffer=200, gen_replay_buffer_capacity=None,
    #     init_tensorboard=False, init_tensorboard_graph=False,
    #     debug_use_ground_truth=False)
    trainer = AdversarialTrainer()  # broken, need to fix

    # Rough sketch of things I have to do here:
    #   - Determine appropriate architectures for discriminator and generator.
    #   - Figure out how to create vecenv from my plain old env.
    #   - Figure out how to use DiscrimNetGAIL.
    #   - Implement serialisation for both reward model and policy.
    #   - Tune optim hyperparams until it works :-)
    #   - (Maybe) write a test function that can evaluate reward function as
    #     well as policy (e.g. do better trajectories, according to the scoring
    #     function, obtain higher reward?)

    raise NotImplementedError()


if __name__ == '__main__':
    register_envs()
    try:
        with cli.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = cli.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        if e.exit_code == 0:
            sys.exit(e.exit_code)
        raise
