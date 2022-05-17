#!/usr/bin/env python3

"""Tools to help with random controls for analyses.
"""
from collections.abc import Mapping
from copy import deepcopy
from lazy import lazy

import pandas as pd

from connsense.io import logging
from connsense.io.write_results import LazyMatrix, read_toc_plus_payload
from connsense.randomize_connectivity.algorithm import SingleMethodAlgorithmFromSource

from connsense.analyze_connectivity import BATCHED_SUBTARGETS

STEP = "analyze-connectivity"
LOG = logging.get_logger(STEP)


class LazyRandomMatrix:
    """Randomize an adjacenciy matrix, but lazily.
    """
    def __init__(self, matrix, node_properties, using_shuffling, name=None, tapping=None,
                 log_info=None, **kwargs):
        """Lazily randomize a matrix using a specified shuffling algorithm,
        which may use further keyword arguments such as graph node properties
        that can be passed among `kwargs`.

        index :: to name the matrix
        """
        self._original = matrix
        self._shuffle = using_shuffling
        self._node_properties = node_properties
        self._name = name
        self._tap = tapping
        self._log_info = log_info
        self._params = kwargs

    @lazy
    def original(self):
        """..."""
        try:
            return self._original.matrix
        except AttributeError:
            return self._original
        raise RuntimeError("Python should not even execute this line.")

    @lazy
    def matrix(self):
        """Wake up.
        """
        LOG.info("Wake up lazy random matrix of shape %s", self.original.shape[0])
        LOG.info("\t [REFERENCE]%s", self._log_info)
        return self._shuffle.apply(self.original, self._node_properties, subtarget=self._name,
                                   tap=self._tap, **self._params)

    @staticmethod
    def dump_toc(of_subtargets, to_dir):
        """Write a toc of lazy-random matrices to a destination.
        A `LazyRandom` object carries a module inside it and hence cannot be pickled.
        We need to save the origical toc with index that of original adjacencies.
        A loader can be used ot load ...
        """
        h5, hdf_group = BATCHED_SUBTARGETS
        path_h5 = to_dir / h5
        if path_h5.exists():
            LOG.warning("OVERRIDE existing data for batched subtargets exists at %s\n"
                        "Consider providing flags,"
                        " for example allow override only in pipeline `mode=resume`", h5)
        return (of_subtargets.apply(lambda lazy_random: lazy_random._original)
                .to_hdf(path_h5, key=hdf_group, mode='w'))

    @staticmethod
    def load_toc(of_subtargets):
        """Load a toc of subtargets containing the original connectivity dumped as above.
        and wrap in shuffler.
        """
        raise NotImplementedError


def lazy_random(control):
    """..."""
    def apply(adjacency, node_properties=None):
        """..."""
        return LazyRandomMatrix(adjacency, control, node_properties)
    return apply


class RandomControls:
    """A random control to help with connectivity measurements.

    As an example config section,

        "erdos-renyi": {
          "source": "/gpfs/bbp.cscs.ch/project/proj83/analyses/topological-analysis-subvolumes/proj83/connectome_analysis/library/randomization.py",
          "method": "ER_shuffle",
          "samples": [0, 1, 2, 3, 4, 5],
          "kwargs": {}
        }
    """
    @staticmethod
    def read_seeds(description):
        """..."""
        seeds = description.get("seeds", [])
        assert isinstance(seeds, list),(
            "Provide a list of random seeds --"
            " we may allow an integer representing number of controls in a future version.")
        return seeds

    def __init__(self, name, description, be_lazy=True):
        """..."""
        self._name = name
        self._description = description
        self._seeds = self.read_seeds(description)
        self._lazily = be_lazy

        self._index_algo = None

    def name_sample(self, s):
        """...name a sample...
        """
        return f"{self._name}-variant-{s}"

    def seed_algorithm(self, s):
        """Set the seed to prepare an algorithm method for this random control.
        """
        description = deepcopy(self._description)
        _=description.pop("seeds", None)
        description["seed"] = s
        return SingleMethodAlgorithmFromSource(self.name_sample(s), description)

    @lazy
    def algorithms(self):
        """..."""
        names = [self.name_sample(s) for s in self._seeds]
        algorithms = pd.Series([self.seed_algorithm(s) for s in  self._seeds],
                               index=pd.Index(names, name="algorithm"))
        self._index_algo = {a: i for i, a in enumerate(algorithms)}
        return algorithms

    def __getitem__(self, key):
        """..."""
        return self.algorithms[key]

    def __iter__(self):
        """Iter throught the random control algorithms.
        """
        return self.algorithms.index.values

    def __len__(self):
        """..."""
        return len(self.algorithms)

    def __call__(self, adjacency, node_properties=None, log_info=None):
        """Apply random controls...
        """
        n = len(self.algorithms)

        def to_inputs_algorithm(a):
            """..."""
            LOG.info("Apply %s / %s control %s.", self._index_algo[a] + 1, n, a.name)

            if not self._lazily:
                return a.apply(adjacency, node_properties, log_info)

            return LazyRandomMatrix(matrix=adjacency, using_shuffling=a,
                                    node_properties=node_properties, log_info=log_info)

        return self.algorithms.apply(to_inputs_algorithm)


def read_randomization(configured, argument=None):
    """...
    TODO: improve exception handling.
    """
    if isinstance(argument, Mapping):
        return argument

    assert not argument or isinstance(argument, str)

    parameters = configured["parameters"]
    randomize = parameters["connectivity-controls"]
    algorithms = randomize["algorithms"]
    return algorithms[argument] if argument else algorithms


def read_seeds(configured, for_algorithm):
    """..."""
    a = read_randomization(configured, for_algorithm)
    return a["seeds"]


def read_random_controls(argued, in_config, lazily=True):
    """..."""
    algorithm = read_randomization(in_config, argued)
    return RandomControls(name=argued, description=algorithm, be_lazy=lazily)
