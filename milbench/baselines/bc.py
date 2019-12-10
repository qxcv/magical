"""Train a policy with behavioural cloning."""
import collections
import multiprocessing
import os
import random
import sys
import time

import click
import gym
from imitation.algorithms.bc import BCTrainer
from imitation.util import rollout as roll_util
from imitation.util import util
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from milbench.baselines.common import (SimpleCNNPolicy, load_demos,
                                       preprocess_demos_with_wrapper,
                                       splice_in_preproc_name)
from milbench.benchmarks import DEMO_ENVS_TO_TEST_ENVS_MAP, register_envs

# put this here so that it happens even in subprocesses spawned for vecenvs
register_envs()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--scratch",
              default="scratch",
              help="directory to save snapshots in")
@click.option("--batch-size", default=32, help="batch size for training")
@click.option("--nholdout",
              default=0,
              help="use the first HOLDOUT demos only for cross-validation")
@click.option("--nepochs", default=100, help="number of epochs of training")
@click.option(
    "--add-preproc",
    default=None,
    type=str,
    help="add preprocessor to the demos and test env (e.g. LoResStack)")
@click.argument("demos", nargs=-1, required=True)
def train(demos, scratch, batch_size, nholdout, nepochs, add_preproc):
    """Use behavioural cloning to train a convnet policy on DEMOS."""
    demo_dicts = load_demos(demos)
    orig_env_name = demo_dicts[0]['env_name']
    if add_preproc:
        env_name = splice_in_preproc_name(orig_env_name, add_preproc)
        print(f"Splicing preprocessor '{add_preproc}' into environment "
              f"'{orig_env_name}'. New environment is {env_name}")
    else:
        env_name = orig_env_name
    env = gym.make(env_name)
    env.reset()

    # split into train/validate
    demo_trajs = [d['trajectory'] for d in demo_dicts]
    if add_preproc:
        demo_trajs = preprocess_demos_with_wrapper(demo_trajs, orig_env_name,
                                                   add_preproc)
    if nholdout > 0:
        # TODO: do validation occasionally (currently this is unused)
        val_transitions = roll_util.flatten_trajectories(demo_trajs[:nholdout])
    train_transitions = roll_util.flatten_trajectories(demo_trajs[nholdout:])

    # train for a while
    policy_class = SimpleCNNPolicy
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        trainer = BCTrainer(env,
                            expert_demos=train_transitions,
                            policy_class=policy_class,
                            batch_size=batch_size)

        def save_snapshot(trainer_locals):
            # intermediate policy, can delete later
            save_path = os.path.join(scratch, "policy-intermediate.pkl")
            trainer.save_policy(save_path)

        trainer.train(n_epochs=nepochs, on_epoch_end=save_snapshot)

        # save policy
        save_path = os.path.join(scratch, "policy.pkl")
        print(f"Saving a model to '{save_path}'")
        trainer.save_policy(save_path)


@cli.command()
@click.option("--snapshot",
              default="scratch/policy.pkl",
              help="path to saved policy")
@click.option("--env-name",
              default="MoveToCorner-Demo-LoResStack-v0",
              help="name of env to instantiate")
@click.option("--det-pol/--no-det-pol",
              default=False,
              help="should actions be sampled deterministically?")
def test(snapshot, env_name, det_pol):
    """Roll out the given SNAPSHOT in the environment."""
    env = gym.make(env_name)
    obs = env.reset()

    with tf.Session():
        policy = BCTrainer.reconstruct_policy(snapshot)

        spf = 1.0 / env.fps
        try:
            while env.viewer.isopen:
                # for limiting FPS
                frame_start = time.time()
                # return value is actions, values, states, neglogp
                (action, ), _, _, _ = policy.step(obs[None],
                                                  deterministic=det_pol)
                obs, rew, done, info = env.step(action)
                obs = np.asarray(obs)
                env.render(mode='human')
                if done:
                    print(f"Done, score {info['eval_score']:.4g}/1.0")
                    obs = env.reset()
                elapsed = time.time() - frame_start
                if elapsed < spf:
                    time.sleep(spf - elapsed)
        finally:
            env.viewer.close()


@cli.command()
@click.option("--snapshot",
              default="scratch/policy.pkl",
              help="path to saved policy")
@click.option('--nrollout', default=30, help="number of rollouts to perform")
@click.option("--demo-env-name",
              default="MoveToCorner-Demo-LoResStack-v0",
              help="name of env to instantiate")
@click.option("--seed", default=42, help="seed for TF etc.")
@click.option("--nenvs", default=None, help="number of parallel envs to use")
def testall(snapshot, demo_env_name, nrollout, seed, nenvs):
    """Compute completion statistics on an entire family of benchmark tasks."""
    test_env_names = [
        demo_env_name,
        *DEMO_ENVS_TO_TEST_ENVS_MAP[demo_env_name],
    ]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

    if nenvs is None:
        nenvs = max(1, multiprocessing.cpu_count())
    assert nenvs > 0

    with tf.Session():
        policy = BCTrainer.reconstruct_policy(snapshot)
        mean_scores = []

        for env_name in test_env_names:
            print(f"Testing on {env_name}")
            # env = gym.make(env_name)
            vec_env = util.make_vec_env(env_name,
                                        nenvs,
                                        seed=seed,
                                        parallel=nenvs > 1)
            scores = []
            for _ in tqdm.trange(int(np.ceil(float(nrollout) / nenvs))):
                # obs = env.reset()
                obses = vec_env.reset()
                while True:
                    # (action, ), _, _, _ = policy.step(obs[None])
                    # obs, rew, done, info = env.step(action)
                    # obs = np.asarray(obs)
                    actions, _, _, _ = policy.step(obses)
                    obses, rews, dones, infos = vec_env.step(actions)
                    if np.any(dones):
                        scores.extend(d['eval_score'] for d in infos)
                        break
            # drop the last few rollouts if we have a vec env that's too big
            scores = scores[:nrollout]
            # TODO: it would be nice to make a t-dist CI instead of doing this
            mean_scores.append((env_name, np.mean(scores)))

    records = [
        collections.OrderedDict([
            ('demo_env', demo_env_name),
            ('test_env', env_name),
            ('mean_score', mean_score),
            ('snapshot', snapshot),
        ]) for env_name, mean_score in mean_scores
    ]
    frame = pd.DataFrame.from_records(records)
    print(f"Final mean scores for '{snapshot}':")
    print(frame[['test_env', 'mean_score']])

    return frame


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
