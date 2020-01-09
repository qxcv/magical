"""Uses DAgger to interactively collect data."""

import os
import time

import click
import gym
from imitation.algorithms.dagger import (DAggerTrainer, NeedsDemosException,
                                         linear_beta_schedule)
from imitation.util.util import make_session
from pyglet.window import key

from milbench.baselines.common import SimpleCNNPolicy
from milbench.benchmarks import register_envs


@click.group()
@click.option('--scratch', help='directory to save snapshots etc. in')
@click.pass_context
def cli(ctx, scratch):
    register_envs()
    ctx.ensure_object(dict)
    ctx.obj['scratch'] = scratch


@cli.command()
@click.option('--env-name',
              default='MoveToCorner-Demo-LoResStack-v0',
              help='name of environment to train on')
@click.option(
    '--nepochs',
    default=50,
    help='number of epochs (passes through the full demo dataset) of '
    'training to do')
@click.pass_context
@make_session()
def train(ctx, env_name, nepochs):
    # TODO: refactor this into a 'create' command that creates a new DAgger
    # training session, and a 'train' command that does a round of training.
    scratch = ctx.obj['scratch']
    if os.path.exists(scratch):
        print(f"Scratch directory '{scratch}' already exists, reloading")
        trainer = DAggerTrainer.reconstruct_trainer(scratch)
    else:
        print(f"Scratch directory '{scratch}' does not exist, constructing "
              f"trainer anew")
        trainer = DAggerTrainer(env=gym.make(env_name),
                                scratch_dir=scratch,
                                policy_class=SimpleCNNPolicy,
                                beta_schedule=linear_beta_schedule(10))
        # initial save, just in case
        trainer.save_trainer()
    try:
        print(f"Training for round {trainer.round_num}")
        trainer.extend_and_update(n_epochs=nepochs)
    except NeedsDemosException as ex:
        raise NeedsDemosException(
            "No demos for this round yet. Try running 'collect' command. "
            "Original error: " + str(ex))
    trainer.save_trainer()
    print(f"Training done, new round is {trainer.round_num}. "
          "Remember to run the .collect() command to get new demos")


@cli.command()
@click.pass_context
@make_session()
def collect(ctx):
    # FIXME: refactor to dedupe code with __main__
    scratch = ctx.obj['scratch']
    print(f"Attempting to reload existing trainer from '{scratch}'")
    trainer = DAggerTrainer.reconstruct_trainer(scratch)

    print(f"Running data collection for round {trainer.round_num}")
    collector = trainer.get_trajectory_collector()
    print(f"Beta for this round is {collector.beta}")

    try:
        collector.reset()
        was_done_on_prev_step = False
        collector.render(mode='human')
        # keys that are depressed will end up in key_map
        key_map = key.KeyStateHandler()
        collector.env.viewer.window.push_handlers(key_map)

        # render loop
        spf = 1.0 / collector.env.fps
        while collector.env.viewer.isopen:
            if key_map[key.R]:
                collector.reset()
                time.sleep(0.5)
                was_done_on_prev_step = False

            frame_start = time.time()

            action = [0, 0, 0]
            if key_map[key.UP] and not key_map[key.DOWN]:
                action[0] = 1
            elif key_map[key.DOWN] and not key_map[key.UP]:
                action[0] = 2
            if key_map[key.LEFT] and not key_map[key.RIGHT]:
                action[1] = 1
            elif key_map[key.RIGHT] and not key_map[key.LEFT]:
                action[1] = 2
            if key_map[key.SPACE]:
                action[2] = 1

            _, _, done, info = collector.step(action)

            if done and not was_done_on_prev_step:
                print(f"Done, score {info['eval_score']:.4g}/1.0")
                print("Hit R to save this demo and record another")

            collector.render(mode='human')

            was_done_on_prev_step = done

            # wait for next frame
            elapsed = time.time() - frame_start
            if elapsed < spf:
                time.sleep(spf - elapsed)
    finally:
        # once done, close the window
        collector.env.close()

    print(f"Collection for round {trainer.round_num} done")


if __name__ == '__main__':
    cli()
