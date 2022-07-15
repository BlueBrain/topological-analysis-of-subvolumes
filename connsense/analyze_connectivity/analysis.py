"""What is an analysis?"""
from abc import ABC, abstractmethod
from pprint import pformat

import pandas as pd

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


class ApplicableFromSource(ABC):
    """Generate subgraphs of an adjacency matrix.
    """
    @property
    @abstractmethod
    def apptype(self):
        """For example: analysis, randomization, control, subgraph...
        """
        raise NotImplementedError(f"Provide what this {self.__class__.__name__} will do to inputs.\n",
                                  "For example: analysis, randomization, control, subgraph...")

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
        self._apply = self.load(description)
        self._output_type = self.read_output_type(description)
        self._collection_policy = self.read_collection(description)

    def load(self, description):
        """...Load the described apply method...
        Override in a subclass...
        """
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

    @property
    def name(self):
        """..."""
        return self._name

    @property
    @abstractmethod
    def hdf_group(self):
        """An HDf group name.
        A concrete implementation of `AnalysisFromSource` will have to decide
        its HDf group name.
        """
        raise NotImplementedError("Concrete class must decide what HDF group to save its data in.")

    @property
    def description(self):
        return self._description

    @property
    def output_type(self):
        return self._output_type

    @property
    def method(self):
        return self._method

    @staticmethod
    def read_matrix(adjacency, with_control=None):
        """..."""
        controlled = with_control or (lambda m: m)
        try:
            matrix = adjacency.matrix
        except AttributeError as aberration:
            LOG.error("Could not get adjancency.matrix: %s", aberration.args)
            matrix = adjacency
        return controlled(matrix)

    @abstractmethod
    def apply(self, adjacency, node_properties,*, with_control=None, tap=None, subtarget=None,
              log_info=None, **kwargs):
        """..."""
        raise NotImplementedError("Delegated to the concrete implementation...")


class SingleMethodAnalysisFromSource(ApplicableFromSource):
    """Algorithm that can be configured as:

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
    apptype = "analysis"

    def __init__(self, name, description, cannot_be_analyses=None):
        """...
        cannot_be_analyses :: a list of names that cannot be analyses used in the input config.
        ~                        this list will free the scientist to use the listed words as
        ~                        argument names in their input libraries.
        ~                     A default is provided.
        """
        super().__init__(name, description)
        self._cannot_be_analyses = cannot_be_analyses or (
            "adjacency", "adj", "nodes", "node_properties", "controls", "args", "kwargs")

    @property
    def quantity(self):
        """To name the column in a dataframe, or an item in a series."""
        method =  self._description.get("method", self._apply.__name__)
        return self._description.get("quantity", method)

    @property
    def hdf_group(self):
        """..."""
        return self.name

    def _input_analyses(self, tap, subtarget):
        """Get other analyses from the TAP that are needed for this one.

        To do so, lookup the index entry subtarget in each analysis stored in the TAP instance tap.

        TODO: Provide better documnetation, and log messages to help the user
        """
        import inspect
        signat = inspect.signature(self._apply)

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
                return in_toc.loc[index_entry].get_value()
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

    def apply(self, adjacency, node_properties=None, *, with_control=None, tap=None, subtarget=None,
              log_info=None, **kwargs):
        """Use keyword arguments to test interactively, instead of reloading a config.
        """
        LOG.info("Apply analysis %s to (%s) %s-type matrix, and %s-type nodes with kwargs: \n%s",
                 self.name, log_info or "", type(adjacency), type(node_properties), pformat(self._kwargs))

        LOG.info("Apply analysis %s to (%s) %s-shaped matrix, and %s nodes with kwargs: \n%s",
                 self.name, log_info or "", matrix.shape,
                 len(node_properties) if node_properties is not None else 0,
                 pformat(self._kwargs))
        matrix = self.read_matrix(adjacency, with_control)

        if log_info:
            LOG.info("Apply analysis %s %s\t\t of shape %s", self.name, log_info, matrix.shape)

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        input_analyses = self._input_analyses(tap, subtarget) if tap else {}
        LOG.info("Input analyses to subtarget \n%s: \n%s", subtarget, pformat(input_analyses or "None"))
        try:
            result = self._apply(matrix, node_properties, *self._args,
                                 **input_analyses, **self._kwargs, **kwargs)
        except RuntimeError as runt:
            called_by = log_info if log_info else "unknown"
            LOG.error("FAILURE in analysis %s of adj mat %s called by %s: "
                      "RuntimeError caught %s.", self.name, matrix.shape, called_by, runt)
            return None

        LOG.info("Done analysis %s of %s adjacency (%s)", self.name, matrix.shape, log_info or "...")

        return result


    @staticmethod
    def collect(data):
        """collect data...
        TODO: We could have the scientist provide a collect method.
        """
        return data


class AdjacencySubgraphs(ApplicableFromSource):
    """..."""
    apptype = "subgraph"

    @property
    def hdf_group(self):
        """..."""
        self.name

    def apply(self, adjacency, node_properties, with_control=None, log_info=None, **kwargs):
        """..."""
        matrix = self.read_matrix(adjacency, with_control)

        if log_info:
            LOG.info("Generate subgraphs %s (%s) to adjacency %s", self.name, log_info, matrix.shape)

        if node_properties is not None:
            assert matrix.shape[0] == node_properties.shape[0]

        try:
            result = self._apply(adjacency, node_properties, **kwargs)
        except Exception as failure:
            called_by = log_info if log_info else "NA"
            LOG.error("FAILURE in subgraphs %s of adj mat %s called by %s. Exception caught: \n%s",
                      self.name, matrix.shape, called_by, failure)
            return None

        if log_info:
            LOG.info("Done subgraphs %s of adjacency of shape %s from %s", self.name, matrix.shape,
                     log_info)

        return result


class SubgraphAnalysisFromSource(SingleMethodAnalysisFromSource):
    """Run an analysis on the subgraphs of an adjacency matrix.
    """
    class MissingRequiredConfiguration(KeyError): pass

    def __init__(self, name, description, subgraphs, **kwargs):
        """..."""
        LOG.info("Initialize SubgraphAnalysisFromSource with name %s, description %s, subgraphs %s",
                 name, description, subgraphs)
        label, algorithm = subgraphs

        try:
            subgraphs_to_apply = description["subgraphs-to-apply"]
        except KeyError as keharr:
            raise SubgraphAnalysisFromSource.MissingRequiredConfiguration(
                "Analysis description should also describe the subgraphs to apply."
            ) from keharr

        LOG.info("SubgraphAnalysisFromSource got subgraphs to apply:\n%s", subgraphs_to_apply)

        try:
            subgraph_description = subgraphs_to_apply[label]
        except KeyError as keharr:
            raise SubgraphAnalysisFromSource.MissingRequiredConfiguration(
                f"Subgraph descriptions should also contain informationl about {label}."
            ) from keharr

        LOG.info("SubgraphAnalysisFromSource got subgraph description:\n%s", subgraph_description)

        try:
            output_type = subgraph_description["output"]
        except KeyError as keharr:
            raise SubgraphAnalysisFromSource.MissingRequiredConfiguration(
                f"Subgraph {label}'s description should specify the resulting subgraph analysis' "
                "output type as entry `output`"
            ) from keharr
        description["output"] = output_type

        self._subgraphs = AdjacencySubgraphs(label, algorithm)

        super().__init__(name, description, **kwargs)
        #super().__init__(f"{name}_{self._subgraphs.label}", description, **kwargs)

    @property
    def hdf_group(self):
        """..."""
        return f"{self.name}/{self._subgraphs.name}"

    def read_subgraphs(self, ss, with_control=None):
        """..."""
        controlled = self.read_matrix(ss.adjacency, with_control)
        return pd.Series({"adjacency": controlled, "node_properties": ss.node_properties},
                         name=s.name)

    def generate_subgraphs(self, adjacency, node_properties, with_control=None):
        """..."""
        return (self._subgraphs.apply(adjacency, node_properties, with_control=False)
                .apply(lambda ss: self.read_subgraphs(ss, with_control), axis=1))

    def apply(self, adjacency, node_properties, with_control=None, **kwargs):
        """..."""
        subgraphs = self.generate_subgraphs(adjacency, node_properties, with_control)

        an_apply = super().apply
        return subgraphs.apply(lambda s: an_apply(s.adjacency, s.node_properties, **kwargs),
                               axis=1)
