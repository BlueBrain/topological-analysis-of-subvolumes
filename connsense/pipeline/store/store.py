"""Interface to the HDFStore where the pipeline stores its data."""
from collections import OrderedDict, defaultdict
from lazy import lazy
from pathlib import Path
import h5py

import pandas as pd

from connsense import analyze_connectivity as anzconn
from connsense.analyze_connectivity import matrices
from connsense import randomize_connectivity as ranconn
from connsense.io.write_results import (read as read_dataset,
                                        read_subtargets,
                                        read_node_properties,
                                        read_toc_plus_payload)
from connsense.io import logging


LOG = logging.get_logger(__name__)


def locate_store(config, in_connsense_h5=None):
    """..."""
    if not in_connsense_h5:
        return Path(config["paths"]["input"]["store"])
    return Path(in_connsense_h5)


def group_steps(config):
    """..."""
    inputs = config["paths"]["input"]["steps"]
    return {step: group for step, (_, group) in inputs.items()}


class HDFStore:
    """Handle the pipeline's data.
    """
    def load_analyses(self, config):
        """Load analyses from `anzconn` and cast them for the needs of the `HDFStore`.
        """
        configured = anzconn.get_analyses(self._config, as_dict=True)

        def load_configured(quantity, analyses):
            """..."""
            fullgraph = (analyses["fullgraph"].hdf_group, analyses["fullgraph"])
            subgraphs = [(a.hdf_group, a) for _,a in analyses["subgraphs"].items()]
            return [fullgraph] + subgraphs

        return {name: analysis for quantity, analyses in configured.items()
                for name, analysis in load_configured(quantity, analyses)}

    def __init__(self, config, in_connsense_h5=None):
        """..."""
        self._config = config
        self._root = locate_store(config, in_connsense_h5)
        self._groups = group_steps(config)
        self._analyses = self.load_analyses(config)
        self._controls = ranconn.get_controls(config)

    def get_path(self, step):
        """..."""
        return (self._root, self._groups[step])


    @lazy
    def datasets(self):
        """..."""
        return defaultdict(lambda: {})

    def read_dataset(self, d):
        """..."""
        step, dset = d
        h5, group = self.get_path(step)
        if dset not in self.datasets:
            self.datasets[dset] = read_dataset((h5, group+"/"+dset), step)
        return pd.concat([self.subtargets, self.datasets[dset]], axis=1).set_index("subtarget")

    @lazy
    def subtargets(self):
        """..."""
        h5, group = self.get_path("define-subtargets")
        return read_dataset((h5, group+"/index"), "define-subtargets")

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
        populations = list(self._config["parameters"]["extract-node-populations"]["populations"])
        hdf, grp = self.get_path("extract-node-populations")
        nodes = {p: read_node_properties((hdf, grp+'/'+p)) for p in populations}
        if len(nodes) == 1:
            return nodes[populations[0]]
        return nodes
        #try:
            #return read_node_properties(self.get_path("extract-node-populations"))
        #except (KeyError, FileNotFoundError):
            #return None

    def _read_matrix_toc(self, step, dset=None):
        """Only for the steps that store connectivity matrices."""
        root, group = self.get_path(step)
        if dset:
            group += f"/{dset}"
        return read_toc_plus_payload((root, group), step)

    @lazy
    def adjacency(self):
        """Original connectivity of subtargets that have been saved to the HDF store."""
        def get_population(p):
            LOG.info("Look for extracted edges in population %s", p)
            try:
                return self._read_matrix_toc("extract-edge-populations", p+"/adj")
            except (KeyError, FileNotFoundError):
                LOG.warning("Nothing found for extract-edge-populations %s", p+"/adj")
                return None
        populations = list(self._config["parameters"]["extract-edge-populations"]["populations"])
        return {p: get_population(p) for p in populations}

    def _read_edge_properties(self, population):
        """..."""
        root, edges = self.get_path("extract-edge-populations")
        store = matrices.get_store(root, edges+"/"+population+"/props",  "pandas.DataFrame")
        return store.toc

    @lazy
    def edge_properties(self):
        """..."""
        def get_population(p):
            LOG.info("Look for extracted edge properties in population %s", p)
            try:
                return self._read_edge_properties(p)
            except (KeyError, FileNotFoundError):
                LOG.warning("Nothing found for extract-edge-populations %s", p+"/props")
                return None
        populations = list(self._config["parameters"]["extract-edge-populations"]["populations"])
        return {p: get_population(p) for p in populations}

    @lazy
    def randomizations(self):
        """Read randomizations."""
        with h5py.File(self._root, 'r') as hdf:
            if not (self._groups[ranconn.STEP] in hdf):
                return None

        def get(control):
            try:
                return self._read_matrix_toc(ranconn.STEP, control)
            except KeyError as err:
                LOG.warning("Could not find data for randomization %s", control)
                pass
            return None

        return {control:  get(control) for control in self._controls.keys()}

    @lazy
    def analyses(self):
        """A TOC for analyses results available in the HDF store.
        """
        def tabulate_contents(analysis, store):
            """..."""
            if not store:
                return None
            try:
                return store.toc
            except FileNotFoundError as error:
                LOG.error("Analysis %s NO TOC found: %s", analysis, error)
                return None
            return None

        p = (self._root, self._groups[anzconn.STEP])
        tocs = {an: tabulate_contents(an, anzconn.get_value_stores(analysis, at_path=p, in_mode='r'))
                for an, analysis in self._analyses.items()}
        return {tic: toc for tic, toc in tocs.items() if toc is not None}

    @lazy
    def circuits(self):
        """Available circuits for which subtargets have been computed."""
        return self.subtarget_gids.index.get_level_values("circuit").unique().to_list()

    def pour_subtarget(self, s, dataset):
        """..."""
        return self.read_dataset(dataset).loc[s]

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


    def locate_fmap_columns(self, circuit, subtargets=None, with_size=None):
        """..."""
        in_circuit = self.subtarget_gids.loc[circuit]
        nodes = in_circuit.apply(len).droplevel(["flat_x", "flat_y"]).rename("nodes")

        def get_edges(subtarget):
            return self.pour_adjacency(circuit, subtarget, "local").sum()

        idx_subtargets = in_circuit.index.get_level_values("subtarget")
        edges = pd.Series([get_edges(s) for s in idx_subtargets], name="edges", index=idx_subtargets)

        numbers = pd.concat([nodes, edges], axis=1)
        fmap_xy = in_circuit.index.to_frame().reset_index(drop=True).set_index("subtarget")
        return pd.concat([fmap_xy, numbers], axis=1, keys=["position", "number"])
