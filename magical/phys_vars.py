"""Tools for defining and tracking 'physics variables' for the environment,
such as the mass of shapes, the 'motive force' of the robot, the power of its
grippers, and so on. The list of physics-related variables tracked here is not
exhaustive; variables have been added here primarily so that they can be
subject to dynamics randomisation in a consistent way. For instance, we want
all shapes to have the same mass in any given state, even if some dynamics
randomisation variants of a task might randomise that single number on
environment reset.

See `base_env.py` for actual values of the real physics variables used in the
MAGICAL environments.

(in hindsight this is wayyyyy more complicated than it will to be for the
foreseeable future; can probably replace with dict or something)"""

import collections

# names that can't be used for
RESERVED_NAMES = set(['defaults', 'sample', 'variables'])
RESERVED_PREFIXES = ['_', 'rand_']


def _is_var_name(name):
    return name not in RESERVED_NAMES \
        and all(not name.startswith(p) for p in RESERVED_PREFIXES)


class _PhysicsVariablesMeta(type):
    """Metaclass for classes containing physics variables. When a new attribute
    of type `PhysVar` (and an appropriate name---see `_is_var_name()`) is added
    to a class, this metaclass records it in a `variables` dictionary, but
    removes it from the final attributes list. Note that when an actual
    instance of a physics variables class is constructed, it will still have
    attributes with the same names, but whose values are filled in with either
    defaults or randomly sampled values, as appropriate."""
    def __new__(cls, name, bases, classdict, _pbm_skip=False):
        if _pbm_skip:
            return super().__new__(cls, name, bases, classdict)

        new_classdict = dict()
        variables = collections.OrderedDict()
        for k, v in classdict.items():
            if not _is_var_name(k):
                new_classdict[k] = v
                continue
            if not isinstance(v, PhysVar):
                raise TypeError("variable '{k}' should be of type PhysVar")
            variables[k] = v
        rv = super().__new__(cls, name, bases, new_classdict)
        rv.variables = variables
        return rv


class PhysicsVariablesBase(metaclass=_PhysicsVariablesMeta, _pbm_skip=True):
    """Base class for physics variables with convenience methods for sampling
    random variables, instantiating from defaults, etc."""
    variables: collections.OrderedDict

    def __init__(self, *, _var_values):
        """Constructor for classes holding physics variable values. This should
        NOT be called directly. Instead, call the sample() or defaults()
        methods, as appropriate."""
        if _var_values.keys() != self.variables.keys():
            raise ValueError("must supply all & only given variable names")
        for k, v in _var_values.items():
            assert not isinstance(v, PhysVar), \
                f"are you sure '{k}' is meant to be a PhysVar?"
            setattr(self, k, v)

    @classmethod
    def defaults(cls):
        """Return a new instance of the class with default variable values."""
        return cls(
            _var_values={k: v.default
                         for k, v in cls.variables.items()})

    @classmethod
    def sample(cls, rng):
        """Return a new instance of the class with randomly sampled variable
        values."""
        return cls(
            _var_values={k: v.sample(rng)
                         for k, v in cls.variables.items()})

    def __repr__(self):
        pairs = ', '.join(f'{k}={getattr(self, k)}' for k in self.variables)
        return f'{self.__class__.__name__}({pairs})'


class PhysVar:
    """Container for the default value and randomisation bounds of a physical
    variable. Randomisation is always uniform."""
    def __init__(self, default, bounds):
        assert len(bounds) == 2, bounds
        lower, upper = bounds
        assert lower <= default <= upper, (lower, default, upper)
        self.default = default
        self.lower = lower
        self.upper = upper

    def sample(self, rng):
        result = rng.uniform(self.lower, self.upper)
        return result
