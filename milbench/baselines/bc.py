"""Train a policy with behavioural cloning."""
import functools
import gzip
import os
import sys
import time

# TODO: replace dill with cloudpickle (that's what SB uses, so I don't have to
# introduce another dep)
import click
import dill
import gym
from imitation.algorithms.bc import BCTrainer
from imitation.util import rollout as roll_util
import numpy as np
from stable_baselines.a2c.utils import conv, conv_to_fc, linear
from stable_baselines.common.policies import CnnPolicy
import tensorflow as tf

from milbench.benchmarks import register_envs


@click.group()
def cli():
    pass


def simple_cnn(scaled_images, **kwargs):
    """Simple CNN made to play nicely with my input shapes (and be a bit
    deeper, since this is not RL).

    :param scaled_images: (TensorFlow Tensor) Image input placeholder.
    :param kwargs: (dict) Extra keywords parameters for the convolutional
        layers of the CNN.
    :return: (TensorFlow Tensor) The CNN output layer."""
    # FIXME: make sure input is appropriately scaled to [-1, 1] or something.
    # TODO: make this a beefy resnet and figure out how to give it batch norm.

    # input shape is [batch_size, num_steps, h, w, num_channels]
    # we transpose to [batch_size, h, w, num_channels * num_steps]
    scaled_images_nc_ns = tf.transpose(scaled_images, (0, 2, 3, 1, 4))
    orig_shape_list = scaled_images_nc_ns.shape.as_list()
    orig_shape = tf.shape(scaled_images_nc_ns)
    final_chans = np.prod(orig_shape_list[3:])
    new_shape = tf.concat((orig_shape[:3], (final_chans, )), axis=0)
    scaled_images_ncs = tf.reshape(scaled_images_nc_ns, new_shape)
    activ = tf.nn.relu
    layer_1 = activ(
        conv(scaled_images_ncs,
             'c1',
             n_filters=32,
             filter_size=8,
             stride=4,
             init_scale=np.sqrt(2),
             **kwargs))
    layer_2 = activ(
        conv(layer_1,
             'c2',
             n_filters=64,
             filter_size=4,
             stride=2,
             init_scale=np.sqrt(2),
             **kwargs))
    layer_3 = activ(
        conv(layer_2,
             'c3',
             n_filters=64,
             filter_size=3,
             stride=2,
             init_scale=np.sqrt(2),
             **kwargs))
    layer_4 = activ(
        conv(layer_3,
             'c4',
             n_filters=64,
             filter_size=3,
             stride=2,
             init_scale=np.sqrt(2),
             **kwargs))
    layer_5 = conv_to_fc(layer_4)
    return activ(linear(layer_5, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


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
    demo_dicts = []
    for d_num, d_path in enumerate(demos, start=1):
        print(f"Loading '{d_path}' ({d_num}/{len(demos)})")
        with gzip.GzipFile(d_path, 'rb') as fp:
            demo_dicts.append(dill.load(fp))
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
    policy_class = functools.partial(CnnPolicy, cnn_extractor=simple_cnn)
    trainer = BCTrainer(env,
                        expert_demos=train_transitions,
                        policy_class=policy_class,
                        batch_size=batch_size)
    trainer.train(n_epochs=nepochs)

    # save policy
    save_path = os.path.join(scratch, "policy.pkl")
    print(f"Saving a model to '{save_path}'")
    trainer.save_policy(save_path)


@cli.command()
@click.option("--snapshot",
              default="scratch/policy.pkl",
              help="path to saved policy")
@click.option("--env",
              "env_name",
              default="MoveToCornerLoResStack-v0",
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
                    obs = env.reset()
                elapsed = time.time() - frame_start
                if elapsed < spf:
                    time.sleep(spf - elapsed)
        finally:
            env.viewer.close()


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
