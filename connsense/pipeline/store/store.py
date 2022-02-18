"""Interface to the HDFStore where the pipeline stores its data."""
from collections import OrderedDict
from lazy import lazy
from pathlib import Path

import pandas as pd

from connsense import analyze_connectivity as anzconn
from connsense.io.write_results import (read_subtargets,
                                        read_node_properties,
                                        read_toc_plus_payload)
from connsense.io import logging


LOG = logging.get_logger(__name__)


def locate_store(config):
    """..."""
    return Path(config["paths"]["input"]["store"])


def group_steps(config):
    """..."""
    inputs = config["paths"]["input"]["steps"]
    return {step: group for step, (_, group) in inputs.items()}


class HDFStore:
    """Handle the pipeline's data.
    """
    def __init__(self, config):
        """..."""
        self._config = config
        self._root = locate_store(config)
        self._groups = group_steps(config)
        self._analyses = anzconn.get_analyses(config, as_dict=True)

    def get_path(self, step):
        """..."""
        return (self._root, self._groups[step])

    @lazy
    def subtarget_gids(self):
        """lists of gids for each subtarget column in the database."""
        try:
            return read_subtargets(self.get_path("define-subtargets"))
        except (KeyError, FileNotFoundError):
            return None

    @lazy
    def nodes(self):
        """Subtarget nodes that have been saved to the HDf store."""
        try:
            return read_node_properties(self.get_path("extract-neurons"))
        except (KeyError, FileNotFoundError):
            return None

    def _read_matrix_toc(self, step):
        """Only for the steps that store connectivity matrices."""
        return read_toc_plus_payload(self.get_path(step), step)

    @lazy
    def adjacency(self):
        """Original connectivity of subtargets that have been saved to the HDF store."""
        try:
            return self._read_matrix_toc("extract-connectivity")
        except (KeyError, FileNotFoundError):
            return None

    @lazy
    def randomizations(self):
        """Read randomizations."""
        try:
            return self._read_matrix_toc("randomize-connectivity")
        except (KeyError, FileNotFoundError):
            return None

    @lazy
    def analyses(self):
        """A TOC for analyses results available in the HDF store.
        """
        def toc(analysis):
            """..."""
            agroup = self._groups[anzconn.STEP]
            astore = anzconn.get_value_store(analysis, at_path=(self._root, agroup))
            try:
                return astore.toc
            except FileNotFoundError as error:
                LOG.error("Analysis %s NO TOC found: %s", analysis.name, error)
                return None
            return None

        return {name: toc(analysis) for name, analysis in self._analyses.items()}

    @lazy
    def circuits(self):
        """Available circuits for which subtargets have been computed."""
        return self.subtarget_gids.index.get_level_values("circuit").unique().to_list()

    def pour_subtargets(self, circuit):
        """All subtargets defined for a circuit."""
        if self.subtarget_gids is None:
            return None

        gids = self.subtarget_gids.xs(circuit, level="circuit")
        return pd.Series(gids.index.get_level_values("subtarget").unique().to_list(),
                         name="subtarget")

    def pour_nodes(self, circuit, subtarget):
        """..."""
        if self.nodes is None:
            return None

        level = ("circuit", "subtarget")
        query = (circuit, subtarget)
        return self.nodes.xs(query, level=level)

    def pour_adjacency(self, circuit, subtarget, connectome):
        """..."""
        if self.adjacency is None:
            return None

        if connectome:
            level = ["circuit", "connectome", "subtarget"]
            query = [circuit, connectome, subtarget]
        else:
            level = ["circuit",  "subtarget"]
            query = [circuit, subtarget]

        adj = self.adjacency.xs(query, level=level)
        if adj.shape[0] == 1:
            return adj.iloc[0].matrix
        return adj

    def pour_randomizations(self, circuit, subtarget, connectome, algorithms):
        """..."""
        if self.randomizations is None:
            return None

        if connectome:
            level = ["circuit", "connectome", "subtarget"]
            query = [circuit, connectome, subtarget]
        else:
            level = ["circuit",  "subtarget"]
            query = [circuit, subtarget]

        randomizations = self.randomizations.xs(query, level=level)

        if not algorithms:
            return randomizations

        return randomizations.loc[algorithms]

    def pour_data(self, circuit, subtarget, connectome=None, randomizations=None):
        """Get available data for a subtarget."""
        args = (circuit, subtarget)
        return OrderedDict([("nodes", self.get_nodes(*args)),
                            ("adjacency", self.get_adjacency(*args, connectome or "local")),
                            ("randomizations", self.get_randomizations(*args, connectome, randomizations))])
