"""Train a policy with behavioural cloning."""
import collections
import os
import sys
import time

# TODO: replace dill with cloudpickle (that's what SB uses, so I don't have to
# introduce another dep)
import click
import gym
from imitation.algorithms.bc import BCTrainer
from imitation.util import rollout as roll_util
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from milbench.baselines.common import SimpleCNNPolicy, load_demos
from milbench.benchmarks import DEMO_ENVS_TO_TEST_ENVS_MAP, register_envs


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
@click.argument("demos", nargs=-1, required=True)
def train(demos, scratch, batch_size, nholdout, nepochs):
    """Use behavioural cloning to train a convnet policy on DEMOS."""
    demo_dicts = load_demos(demos)
    env_name = demo_dicts[0]['env_name']
    env = gym.make(env_name)
    env.reset()

    # split into train/validate
    demo_trajs = [d['trajectory'] for d in demo_dicts]
    if nholdout > 0:
        # TODO: do validation occasionally
        val_transitions = roll_util.flatten_trajectories(demo_trajs[:nholdout])
    train_transitions = roll_util.flatten_trajectories(demo_trajs[nholdout:])

    # train for a while
    policy_class = SimpleCNNPolicy
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
def test(snapshot, env_name):
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
                (action, ), _, _, _ = policy.step(obs[None])
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
@click.option("--demo-env-name",
              default="MoveToCorner-Demo-LoResStack-v0",
              help="name of env to instantiate")
def testall(snapshot, demo_env_name):
    """Compute completion statistics on an entire family of benchmark tasks."""
    test_env_names = [
        demo_env_name,
        *DEMO_ENVS_TO_TEST_ENVS_MAP[demo_env_name],
    ]

    with tf.Session():
        policy = BCTrainer.reconstruct_policy(snapshot)
        mean_scores = []

        for env_name in test_env_names:
            print(f"Testing on {env_name}")
            env = gym.make(env_name)
            scores = []
            # TODO: refactor this to use imitation rollout utils once I figure
            # out how to create vecenvs (also use appropriately large vecenv)
            for _ in tqdm.trange(30):
                obs = env.reset()
                while True:
                    (action, ), _, _, _ = policy.step(obs[None])
                    obs, rew, done, info = env.step(action)
                    obs = np.asarray(obs)
                    if done:
                        scores.append(info['eval_score'])
                        break
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
