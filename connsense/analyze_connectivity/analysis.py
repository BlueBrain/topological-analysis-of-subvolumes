"""What is an analysis?"""
from ..plugins import import_module


from ..io.utils import widen_by_index
from ..io.logging import get_logger

LOG = get_logger("Topology Pipeline Analysis", "INFO")

def get_analyses(in_config):
    """..."""
    return collect_plugins_of_type(SingleMethodAnalysisFromSource,
                                   for_analyses=in_config["analyses"])

def collect_plugins_of_type(T, for_analyses):
    """..."""
    return {name: T(name, description) for name, description in for_analyses.items()}


class SingleMethodAnalysisFromSource:
    """Algorithms defined as such in the config:

    "analyze-connectivity": {
      "analyses": {
        "simplex_counts": {
          "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/analysis/simplex_counts.py",
          "args": [],
          "kwargs": {},
          "method": "simplex-counts",
          "output": "scalar"
        }
      }
    }

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
    def read_args(description):
        """..."""
        return description.get("args", [])

    @staticmethod
    def read_kwargs(description):
        """..."""
        return description.get("kwargs", {})

    @staticmethod
    def read_output_type(description):
        """..."""
        return description["output"]

    @staticmethod
    def read_collection(description):
        """..."""
        policy = description.get("collect", None)
        if not policy:
            return None
        return policy

    def __init__(self, name, description):
        """..."""
        self._name = name
        self._description = description
        self._source = self.read_source(description)
        self._args = self.read_args(description)
        self._kwargs = self.read_kwargs(description)
        self._method = self.read_method(description)
        self._analysis = self.load(description)
        self._output_type = self.read_output_type(description)
        self._collection_policy = self.read_collection(description)

    @property
    def name(self):
        """..."""
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def quantity(self):
        """To name the column in a dataframe, or an item in a series."""
        return self._description.get("quantity",
                                     self._description.get("method",
                                                           self._analysis.__name__))

    @property
    def output_type(self):
        return self._output_type

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

    def apply(self, adjacency, node_properties=None, log_info=None,
              **kwargs):
        """Use keyword arguments to test interactively,
        instead of reloading a config.
        """
        try:
            matrix = adjacency.matrix
        except AttributeError:
            matrix = adjacency

        if log_info:
            LOG.info("Apply analysis %s %s\n\t\t of shape %s",
                     self.name, log_info, matrix.shape)

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        result = self._analysis(matrix, node_properties,
                                *self._args, **self._kwargs, **kwargs)

        if log_info:
            LOG.info("Done analsysis %s of %s\n to matrix of shape %s",
                     self.name, log_info, matrix.shape)

        return result

    @staticmethod
    def collect(data):
        """collect data...
        TODO: We could have the scientist provide a collect method.
        """
        return data
