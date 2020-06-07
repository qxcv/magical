# Multitask Assessment of Generalisation in Imitative Control Algorithms (MAGICAL)

MAGICAL is a benchmark suite to evaluate the generalisation capabilities of
imitation learning algorithms.



## Installing and using MAGICAL

You can install MAGICAL using `pip`:

```#!sh
cd /path/to/this/directory
pip install -e .
```

MAGICAL tasks and variants are exposed as Gym environments. Once you've
installed MAGICAL, you can use the Gym environments as follows:

```python
import magical
import gym

# magical.register_envs() must be called before making any Gym envs
magical.register_envs()

# creating a demo variant for one task
env = gym.make('FindDupe-Demo-v0')
env.reset()
env.render(mode='human')
env.close()

# We can also make the test variant of the same environment, or add a
# preprocessor to the environment. In this case, we are creating a
# TestShape variant of the original environment, and applying the
# LoRes4E preprocessor to observations. LoRes4E stacks four
# egocentric frames together and downsamples them to 96x96.
env = gym.make('FindDupe-TestShape-LoRes4E-v0')
init_obs = env.reset()
print('Observation type:', type(obs))  # np.ndarray
print('Observation shape:', obs.shape)  # (96, 96, 3)
env.close()
```

Keep reading to see a list of all available tasks and variants, as well as all
the builtin observation preprocessors that ship with MAGICAL.

## Available tasks and variants



## Preprocessors

