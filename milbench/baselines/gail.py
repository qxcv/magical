"""Generative Adversarial Imitation Learning (GAIL)."""
import collections
import multiprocessing
import sys

import click
from imitation.algorithms.adversarial import init_trainer
from imitation.util import util, rollout
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tqdm

from milbench.baselines.bc import SimpleCNNPolicy, load_demos
from milbench.benchmarks import register_envs


class make_convnet_builder:
    def __init__(self, activation=tf.nn.relu):
        self.activation = activation

    def __call__(self, obs_tensor, act_tensor):
        # FIXME: make sure input is appropriately scaled to [-1, 1] or
        # something. TODO: make this a beefy resnet and figure out how to give
        # it batch norm.

        # obs_tensor signature:
        #     <tf.Tensor 'truediv:0' shape=(?, 4, 96, 96, 3) dtype=float32>
        # act_tensor signature:
        #     <tf.Tensor 'concat:0' shape=(?, 1, 8) dtype=float32>

        # flat_acts should be 3-hot vector of shape [B, 8]
        flat_acts = tf.squeeze(act_tensor, axis=1)
        act_shape = tuple(flat_acts.shape.as_list()[1:])
        assert None not in act_shape, \
            f"act shape {act_shape} not fully specified"

        # going to flatten first axis into channels for observation tensor
        scaled_images = obs_tensor
        scaled_images_nc_ns = tf.transpose(scaled_images, (0, 2, 3, 1, 4))
        orig_shape_list = scaled_images_nc_ns.shape.as_list()
        orig_shape = tf.shape(scaled_images_nc_ns)
        final_chans = np.prod(orig_shape_list[3:])
        new_shape = tf.concat((orig_shape[:3], (final_chans, )), axis=0)
        scaled_images_ncs = tf.reshape(scaled_images_nc_ns, new_shape)
        obs_shape = tuple(scaled_images_ncs.shape.as_list()[1:])
        assert None not in obs_shape, \
            f"obs shape {obs_shape} not fully specified"

        activ = self.activation

        # vision preprocessing stage
        vis_input = layers.Input(obs_shape)
        vis_rep = layers.Conv2D(32, 8, strides=4, activation=activ)(vis_input)
        vis_rep = layers.Conv2D(64, 4, strides=2, activation=activ)(vis_rep)
        vis_rep = layers.Conv2D(64, 3, strides=2, activation=activ)(vis_rep)
        vis_rep = layers.Conv2D(64, 3, strides=2, activation=activ)(vis_rep)
        vis_rep = layers.Flatten()(vis_rep)

        # now merge with actions
        act_input = layers.Input(act_shape)
        merged_rep = layers.Concatenate()([vis_rep, act_input])
        merged_rep = layers.Dense(128, activation=activ)(merged_rep)
        merged_rep = layers.Dense(64, activation=activ)(merged_rep)
        logits_rep = layers.Dense(1)(merged_rep)

        # combine & apply parts of the model
        model = keras.Model(inputs=[vis_input, act_input], outputs=logits_rep)
        logits_tensor_w_extra_dim = model([scaled_images_ncs, flat_acts])
        logits_tensor = tf.squeeze(logits_tensor_w_extra_dim, axis=1)

        # this is only needed for serialisation
        layers_obj = collections.OrderedDict([('model', model)])

        return layers_obj, logits_tensor


@click.group()
def cli():
    # TODO: move the scratch dir option up here so I can have an actual
    # training dir for each training run I do.
    pass


@cli.command()
@click.option("--scratch",
              default="scratch/gail",
              help="directory to save snapshots in")
@click.option("--nenvs",
              default=None,
              type=int,
              help="number of parallel envs to use")
@click.option("--seed", default=42, help="random seed to use")
@click.option("--nepochs", default=100, help="number of epochs to train for")
@click.argument("demos", nargs=-1, required=True)
@util.make_session()
def train(scratch, demos, seed, nenvs, nepochs):
    demo_dicts = load_demos(demos)
    env_name = demo_dicts[0]['env_name']
    demo_trajectories = [d['trajectory'] for d in demo_dicts]
    if nenvs is None:
        nenvs = min(32, max(1, multiprocessing.cpu_count()))
    # Constructor signature:
    trainer = init_trainer(
        env_name,
        demo_trajectories,
        use_gail=True,
        num_vec=nenvs,
        parallel=True,
        trainer_kwargs={
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
    print(f"Training for {nepochs} epochs")
    for epoch in tqdm.tqdm(range(1, nepochs+1), desc="epoch"):
        trainer.train_disc(50)
        trainer.train_gen(1)

    # TODO Rough sketch of things I left have to do here:
    #
    # - Implement serialisation for both reward model and policy.
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
