import collections
import os

import cloudpickle
import numpy as np
from stable_baselines.a2c.utils import conv, conv_to_fc, linear
from stable_baselines.common.policies import CnnPolicy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class make_convnet_builder:
    """Convnet builder for `GAILDiscrimNetwork` in `imitation`."""
    def __init__(self, activation=tf.nn.relu):
        self.activation = activation

    def __call__(self, obs_tensor, act_tensor):
        # FIXME: make sure input is appropriately scaled to [-1, 1] or
        # something. TODO: make this a beefy resnet and figure out how to give
        # it batch norm.

        # obs_tensor signature:
        #     <tf.Tensor 'truediv:0' shape=(?, 4, 96, 96, 3) dtype=float32>
        # act_tensor signature:
        #     <tf.Tensor 'concat:0' shape=(?, 1, 18) dtype=float32>

        # act_tensor should be one-hot vector of shape [B, 18]
        at_shape_list = act_tensor.shape.as_list()
        assert len(at_shape_list) == 2, at_shape_list
        act_shape = tuple(at_shape_list[1:])
        assert None not in act_shape, \
            f"act shape {act_shape} not fully specified"
        assert act_shape == (18,), act_shape

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
        # TODO: consider decreasing kernel size to 5x5 or so here
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
        logits_tensor_w_extra_dim = model([scaled_images_ncs, act_tensor])
        logits_tensor = tf.squeeze(logits_tensor_w_extra_dim, axis=1)

        # this is only needed for serialisation
        layers_obj = collections.OrderedDict([('model', model)])

        return layers_obj, logits_tensor


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


class SimpleCNNPolicy(CnnPolicy):
    """CNN-based Stable Baselines policy with feature extractor given by
    simple_cnn. This class also records constructor arguments, variable
    handles, etc. to make saving/reloading easier."""
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 **kwargs):
        # save args/kwargs so we can pickle this policy later
        self._scnn_args = []
        save_arg_names = {
            # omit sess (which can't be pickled), but include all other args
            'ob_space',
            'ac_space',
            'n_env',
            'n_steps',
            'n_batch',
        }
        locs = locals()
        extra_kwargs = {n: locs[n] for n in save_arg_names}
        all_kwargs = dict(kwargs)
        all_kwargs.update(extra_kwargs)
        self._scnn_kwargs = all_kwargs

        # call parent constructor that does actual graph-building
        super().__init__(sess, cnn_extractor=simple_cnn, **all_kwargs)

        # also save newly-created variables by making use of the 'model' scope
        # employed by parent
        if kwargs.get('reuse'):
            # I don't think we can do anything here; the code that constructs
            # the "train_model" policy uses a custom getter for its train_model
            # variable scope that makes it really hard to use
            # tf.get_collection() in the normal way. Hopefully (?) we never
            # have to serialise this. Look at PPO2.setup_model() in
            # stable_baselines to see what the issue is (specifically the
            # second call to `self.policy` within
            # `tf.variable_scope("train_model", …)`).
            return

        inner_scope = tf.get_variable_scope().name
        if not inner_scope:
            model_scope = 'model'
        else:
            model_scope = inner_scope + '/model'
        self._scnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=model_scope)
        assert len(self._scnn_vars) > 0

    def scnn_save_policy(self, policy_path):
        """Save policy to given path. Note that this policy can be
        reconstructed by BCTrainer.reconstruct_policy() in `imitation`."""
        sess = tf.get_default_session()
        policy_params = sess.run(self._scnn_vars)
        data = {
            'class': SimpleCNNPolicy,
            'args': self._scnn_args,
            'kwargs': self._scnn_kwargs,
            'params': policy_params,
        }
        dirname = os.path.dirname(policy_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(policy_path, 'wb') as fp:
            cloudpickle.dump(data, fp)
