"""Generative Adversarial Imitation Learning (GAIL)."""
import multiprocessing
import os
import sys
import time

import click
from imitation.algorithms.adversarial import init_trainer
from imitation.scripts.train_adversarial import save as save_adv_trainer
from imitation.util import rollout, util
import numpy as np
from stable_baselines import logger
import tensorflow as tf

from milbench.baselines.common import (SimpleCNNPolicy, load_demos,
                                       make_convnet_builder)
from milbench.benchmarks import register_envs


@click.group()
def cli():
    # TODO: move the scratch dir option up here so I can have an actual
    # training dir for each training run I do.
    pass


@cli.command()
@click.option("--scratch",
              default="scratch/adversarial/",
              help="directory to save snapshots in")
@click.option("--nenvs",
              default=None,
              type=int,
              help="number of parallel envs to use")
@click.option("--test-every",
              default=1,
              help="score policy every `--test-every` updates")
@click.option("--save-every",
              default=10,
              help="save policy every `--save-every` updates")
@click.option("--seed", default=42, help="random seed to use")
@click.option("--nepochs", default=100, help="number of epochs to train for")
@click.argument("demos", nargs=-1, required=True)
@util.make_session()
def train(scratch, demos, seed, nenvs, nepochs, test_every, save_every):
    demo_dicts = load_demos(demos)
    env_name = demo_dicts[0]['env_name']
    demo_trajectories = [d['trajectory'] for d in demo_dicts]
    if nenvs is None:
        nenvs = min(32, max(1, multiprocessing.cpu_count()))
    os.makedirs(scratch, exist_ok=True)
    # Constructor signature:
    trainer = init_trainer(
        env_name,
        demo_trajectories,
        use_gail=True,
        num_vec=nenvs,
        parallel=True,
        log_dir=scratch,
        trainer_kwargs={
            'init_tensorboard': True,
            'disc_opt_cls': tf.train.AdamOptimizer,
            'disc_opt_kwargs': dict(learning_rate=1e-3),
        },
        init_rl_kwargs={
            'policy_class': SimpleCNNPolicy,
        },
        discrim_kwargs={
            'build_discrim_net': make_convnet_builder(),
        },
    )
    checkpoint_dir = os.path.join(trainer._log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training for {nepochs} epochs")

    def test_policy(when):
        print(f"Testing policy, {when}")
        start_time = time.time()
        trajectories = rollout.generate_trajectories(
            trainer.gen_policy,
            trainer.venv,
            sample_until=rollout.min_episodes(10))
        scores = []
        for traj in trajectories:
            scores.append(traj.infos[-1]['eval_score'])
        mean_score = np.mean(scores)
        ntraj = len(trajectories)
        elapsed = time.time() - start_time
        print(f"Mean score over {ntraj} trajs: {mean_score:.4g} "
              f"(in {elapsed:.4g}s)")

    test_policy('before training')

    # this is just for writing out stats in a nice table
    human_writer = logger.HumanOutputFormat(sys.stdout)

    for epoch in range(1, nepochs + 1):
        print("\n\n\n\n")
        print(f"Training discriminator ({epoch}/{nepochs})")
        disc_stats = trainer.train_disc(10)
        human_writer.writekvs(disc_stats)

        print(f"Training generator ({epoch}/{nepochs})")
        # FIXME: the fact that train_gen and train_disc take totally different
        # types of arguments is weird. Should fix (?) imitation project so that
        # they both measure progress in either "updates performed on the model"
        # or "steps of interaction data generated/consumed", not a mix of both.
        n_updates = 1
        trainer.train_gen(n_updates * trainer._gen_policy.n_batch)

        if (epoch % test_every) == 0:
            test_policy(f'epoch {epoch}')

        if (epoch % save_every) == 0:
            # save policy for convenience
            policy_path = os.path.join(checkpoint_dir,
                                       f"policy-{epoch:03d}.pkl")
            policy_link = os.path.join(checkpoint_dir, "policy-latest.pkl")
            print(f"Saving policy to '{policy_path}' "
                  f"(linked via '{policy_link}')")
            with trainer.gen_policy.graph.as_default():
                with trainer.gen_policy.sess.as_default():
                    trainer.gen_policy.act_model.scnn_save_policy(policy_path)
            try:
                os.unlink(policy_link)
            except FileNotFoundError:
                pass
            os.symlink(policy_path, policy_link)

            # save the entire trainer
            trainer_path = os.path.join(checkpoint_dir, f"trainer-{epoch:03d}")
            trainer_link = os.path.join(checkpoint_dir, "trainer-latest")
            print(f"Saving trainer to '{trainer_path}' "
                  f"(linked via '{trainer_link}')")
            save_adv_trainer(trainer, trainer_path)
            try:
                os.unlink(trainer_link)
            except FileNotFoundError:
                pass
            os.symlink(trainer_path, trainer_link, target_is_directory=True)

    # TODO Rough sketch of things I left have to do here:
    #
    # - Implement serialisation for both reward model and policy. Probably I
    #   can just serialise the trainer and it will work fine. Serialising the
    #   policy itself will be a bit trickier; perhaps I can hack my
    #   SimpleCNNPolicy to make it work?
    # - Figure out how to do intermediate evaluations to track how well the
    #   model is doing.
    # - Figure out how to make TensorBoard work.
    # - Tune optim hyperparams until it works :-)
    # - (Maybe) write a test function that can evaluate reward function as well
    #   as policy (e.g. do better trajectories, according to the scoring
    #   function, obtain higher reward?)

    raise NotImplementedError()


# putting this up here ensures it is executed on import; otherwise never get
# registered in teh subprocesses created by SubprocVecEnv.
register_envs()

if __name__ == '__main__':
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
