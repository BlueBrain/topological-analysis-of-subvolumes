"""Load a `randomization.Algorihtm from source code.`"""
from copy import deepcopy
from pprint import pformat

import pandas as pd

from randomization import Algorithm
from ..plugins import import_module
from ..io.logging import get_logger

LOG = get_logger("Topology Pipeline Analysis", "INFO")


class AlgorithmFromSource(Algorithm):
    """...
    NOTE: This is underdeveloped and not used.
    ~     May be we want to have more than `SingleMethodAlgorithmFromSource`,
    ~     But a `MultiMethodAlgorithmFromSource` can also be useful.
    ~     So keep this one as a baseclass.
    """
    @staticmethod
    def read_source(description):
        """"..."""
        return description["source"]

    @staticmethod
    def read_functions(description):
        """..."""
        return description["functions"]

    @staticmethod
    def load_args(functions):
        """..."""
        return [description.get("args", []) for description in functions]

    @staticmethod
    def load_kwargs(functions):
        """..."""
        return [description.get("kwargs", {}) for description in functions]

    def load_application(self, source, functions):
        """..."""
        self._functions = [f["name"] for f in functions]
        unique = set(self._functions)

        module, methods = get_module(from_object=source, with_function=unique)
        self._module = module
        self._methods = methods

        self._args = self.load_args(functions)
        self._kwargs = self.load_kwargs(functions)
        return

    def __init__(self, name, description):
        """..."""
        self._name = name
        source = self.read_source(description)
        functions = self.read_functions(description)
        self.load_application(source, functions)

    @property
    def name(self):
        """..."""
        return self._name

    def apply(self, adjacency, node_properties=None):
        """...Apply the algorithm
        This implementation has not been tested, having mostly used it's subclass...
        """
        def apply_method_indexed(i):
            """..."""
            function = self._functions[i]
            method = self._methods[function]
            args = self._args[i]
            kwargs = self._kwargs[i]
            label = self._label(function, args, kwargs)
            return (label, method(adjacency, node_properties, *args, **kwargs))

        N = len(self._functions)
        labels, matrices = zip(*[apply_method_indexed(i) for i in range(N)])

        return pd.Series(matrices, name="matrix", index=pd.Index(labels, names="algorithm"))


def get_algorithms(in_config):
    """..."""
    algorithms = in_config["algorithms"]
    return collect_plugins_of_type(AlgorithmFromSource, in_config=algorithms)

def collect_plugins_of_type(T, in_config):
    """..."""
    return {T(name, description) for name, description in items()}


class SingleMethodAlgorithmFromSource(Algorithm):
    """Algorithms defined as such in the config:

    algorithms : {'erin': {'source': '/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topological-analysis-of-subvolumes/randomization/library/rewire.py',
                            'kwargs': {'invariant_degree': 'IN'},
                            'name': 'connections-rewired-controlling-in-degree'},
                 'erout': {'source': '/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topological-analysis-of-subvolumes/randomization/library/rewire.py',
                            'kwargs': {'invariant_degree': 'OUT'},
                            'name': 'connections-rewired-controlling-out-degree'},
                 'erdos_renyi': {'source': '/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/randomization/ER_shuffle.py',
                                 'method': 'ER_shuffle',
                                 'kwargs': {},
                                 'name': 'erdos-renyi'}}
    """
    @staticmethod
    def read_method(description):
        """..."""
        return description.get("method", "shuffle")

    @staticmethod
    def read_source(description):
        """..."""
        return description["source"]

    @staticmethod
    def read_seed(description):
        """It is assumed that the randomization method wrapped here needs a seed.
        """
        return description.get("seed", None)

    @staticmethod
    def read_args(description):
        """..."""
        return description.get("args", [])

    @staticmethod
    def read_kwargs(description):
        """..."""
        return description.get("kwargs", {})

    def __init__(self, name, description, cannot_be_analyses=None):
        """..."""
        self._name = name
        self._description = description
        self._source = self.read_source(description)
        self._seed = self.read_seed(description)
        self._args = self.read_args(description)
        self._kwargs = self.read_kwargs(description)
        self._method = self.read_method(description)
        self._shuffle = self.load(description)
        self._cannot_be_analyses = cannot_be_analyses or (
            "adjacency", "adj", "nodes", "node_properties", "controls", "args", "kwargs")

    @property
    def name(self):
        """..."""
        return self._name

    @property
    def description(self):
        """..."""
        return self._description

    def load(self, description):
        """..."""
        source = self.read_source(description)
        method = self.read_method(description)

        try:
           run = getattr(source, method)
        except AttributeError:
            pass
        else:
            self._module = source
            return run

        if callable(source):
            #TODO: inspect source
            return source

        module, method = import_module(from_path=source, with_method=method)
        self._module = module
        return method

    def _input_analyses(self, tap, subtarget):
        """Get other analyses from the TAP that are needed for this one.

        To do so, lookup the index entry subtarget in each analysis stored in the TAP instance tap.

        TODO: Provide better documnetation, and log messages to help the user
        TODO: This is a copy from SingleMethodAnalysisFromSource --- find a common place...
        """
        import inspect
        signat = inspect.signature(self._shuffle)

        index_entry = tuple(subtarget)

        def check_toc(of_analysis):
            for a, toc in tap.analyses.items():
                if a.replace('-', '_') == of_analysis:
                    return toc

            LOG.error("No such analysis %s in store at %s", of_analysis, tap._root)
            return None

            try:
                toc = tap.analyses[of_analysis]
            except KeyError:
                LOG.error("No such analysis %s in store at %s", of_analysis, tap._root)
                return None
            return toc

        def lookup_analysis(a, in_toc):
            try:
                return in_toc.loc[index_entry]
            except KeyError:
                LOG.error("No analyses %a value in store for %s ", a, index_entry)
                return None

            LOG.error("Could not find a stored tap value for analysis %s of subtarget %s",
                      a, index_entry)
            return None

        may_be_analysis_name = lambda n: n not in self._cannot_be_analyses
        possibly_analysis = (p for p in signat.parameters if may_be_analysis_name(p))
        tocs = (check_toc(of_analysis=a) for a in possibly_analysis)
        available = [(analysis, toc) for analysis, toc in zip(possibly_analysis, tocs) if toc]

        requested = {a: lookup_analysis(a, in_toc=t) for a, t in available}
        LOG.info("Available inputs for analysis %s available: \n%s", self.name, pformat(requested))

        return requested

    def apply(self, adjacency, node_properties=None, tap=None, subtarget=None,
              log_info=None, **kwargs):
        """..."""
        try:
            matrix = adjacency.matrix
        except AttributeError:
            matrix = adjacency

        if log_info:
            LOG.info("Apply analysis %s %s\t\t of shape %s",
                     self.name, log_info, matrix.shape)

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        input_analyses = self._input_analyses(tap, subtarget) if tap else {}
        LOG.info("input analysis to subtarget %s: \n%s", subtarget, pformat(input_analyses))

        args = deepcopy(self._args)
        kwargs = deepcopy(self._kwargs)
        if self._seed is not None:
            kwargs["seed"] = self._seed

        try:
            result = self._shuffle(matrix, node_properties, *args, **input_analyses, **kwargs)
        except RuntimeError as runt:
            called_by = log_info if log_info else "unknown"

            LOG.error("FAILURE in analysis %s of adj mat %s called by %s: "
                      "RuntimeError caught %s.", self.name, matrix.shape, called_by, runt)
            result = None

        if log_info:
            LOG.info("Done analysis %s of adjacency matrix of shape %s from %s",
                     self.name, matrix.shape, log_info)

        return result

    def __call__(self, adjacency, node_properties=None):
        """..."""
        return self.apply(adjacency, node_properties)
