"""Interface to the HDFStore where the pipeline stores its data."""
from collections import OrderedDict, defaultdict
from lazy import lazy
from pathlib import Path
import h5py

import pandas as pd

from connsense import plugins
from connsense.define_subtargets.config import SubtargetsConfig
from connsense import analyze_connectivity as anzconn
from connsense.analyze_connectivity import matrices
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


CIRCUIT_ID = "circuit"
CONNECTOME_ID = "connectome"
SUBTARGET_ID = "subtarget_id"
MORPH_ID = "morphology_id"


class HDFStore:
    """Handle the pipeline's data.
    """
    # def load_analyses(self):
    #     """Load analyses from `anzconn` and cast them for the needs of the `HDFStore`.
    #     """
    #     configured = anzconn.get_analyses(self._config, as_dict=True)

    #     def load_configured(quantity, analyses):
    #         """..."""
    #         fullgraph = (analyses["fullgraph"].hdf_group, analyses["fullgraph"])
    #         subgraphs = [(a.hdf_group, a) for _,a in analyses["subgraphs"].items()]
    #         return [fullgraph] + subgraphs

    #     return {name: analysis for quantity, analyses in configured.items()
    #             for name, analysis in load_configured(quantity, analyses)}

    INDICES = pd.Series(["circuit", "connectome", "subtarget", "morphology"],
                        index=["circuit", "connectome", "subtarget_id", "morphology_id"])

    def load_analyses(self, group):
        """..."""
        try:
            group_analyses = self._config["parameters"][f"analyze-{group}"]
        except KeyError:
            LOG.warning("No analyses for group %s", group)
            return {}

        return group_analyses["analyses"]

    def __init__(self, config, in_connsense_h5=None):
        """..."""
        self._config = config
        self._root = locate_store(config, in_connsense_h5)
        self._groups = group_steps(config)
        self._circuits = {None: None}

    def get_path(self, step):
        """..."""
        return (self._root, self._groups[step])


    @lazy
    def datasets(self):
        """..."""
        return defaultdict(lambda: {})

    @classmethod
    def get_reader(cls, step):
        """..."""
        if step == "extract-edge-populations":
            return read_toc_plus_payload
        return read_dataset

    def read_dataset(self, d):
        """..."""
        from connsense.pipeline.parallelization.parallelization import describe
        step, dset = describe(d)
        h5, group = self.get_path(step)
        at_path = (h5, group+"/"+dset)

        if dset not in self.datasets[step]:
            if step == "extract-node-types":
                self.datasets[step][dset] = self.read_node_types(dset).sort_index()
            elif step.startswith("analyze-"):
                self.datasets[step][dset] = self.analyses['-'.join(step.split('-')[1:])].get(dset, None)
            else:
                read = self.get_reader(step)
                self.datasets[step][dset] = read(at_path, step).sort_index()
        return  self.datasets[step][dset]

    def subtargets(self, circuit=None, connectome=None):
        """..."""
        if connectome:
            assert circuit, f"Need circuit for connectome"

        def index_subtargets(df, value, key):
            return pd.concat([df], axis=0, keys=[value], names=[key])

        h5, group = self.get_path("define-subtargets")
        data = read_dataset((h5, group+"/index"), "define-subtargets")

        if not circuit:
            return data
        if connectome:
            data = index_subtargets(data, connectome, "connectome")
        return index_subtargets(data, circuit, "circuit")

    @lazy
    def index_circuits(self):
        """Placeholder."""
        return pd.Series(["Bio_M"], index=pd.Index(["Bio_M"], name="circuit_id"), name="circuit")

    @lazy
    def index_connectome(self):
        """Placeholder."""
        return pd.Series(["local"], index=pd.Index(["local"], name="connectome_id"), name="connectome")

    @lazy
    def index_subtargets(self):
        """..."""
        h5, group = self.get_path("define-subtargets")
        return read_dataset((h5, group+"/index"), "define-subtargets")

    @lazy
    def index_morphologies(self):
        """..."""
        h5, group = self.get_path("extract-node-types")
        return read_dataset((h5, group+"/biophysical/morphology/index"), "extract-node-types")

    @lazy
    def index_names(self):
        """..."""
        return {CIRCUIT_ID: self.index_circuits, CONNECTOME_ID: self.index_connectome,
                SUBTARGET_ID: self.index_subtargets,
                MORPH_ID: self.index_morphologies}

    def modeltypes(self, m, circuit=None):
        """..."""
        modeltype, component = m.split('/')
        if modeltype == "biophysical":
            if component == "morphology":
                morphologies = self.read_dataset(["extract-node-types", "biophysical/morphology"])
                return morphologies.loc[circuit] if circuit else morphologies
            raise KeyError(f"Unknown modeltype {modeltype} component {component}")
        raise KeyError(f"Unown modeltype {modeltype}")


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
            return nodes[populations[0]].sort_index()
        return nodes.sort_index()

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
                return self._read_matrix_toc("extract-edge-populations", p)
            except (KeyError, FileNotFoundError):
                LOG.warning("Nothing found for extract-edge-populations %s", p)
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
        from connsense import randomize_connectivity as ranconn
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

        return {control:  get(control) for control in ranconn.get_controls(self._config).keys()}


    def get_value_store(self, group, analysis):
        """..."""
        to_hdf_at_path, under_group = self.get_path(f"analyze-{group}")
        an,alysis = analysis
        return matrices.get_store(to_hdf_at_path, under_group + "/" + an, for_matrix_type=alysis["output"],
                                  in_mode="r")


    def get_circuit(self, labeled):
        """..."""
        if labeled not in self._circuits:
            sbtcfg = SubtargetsConfig(self._config)
            self._circuits[labeled] = sbtcfg.input_circuit[labeled]
        return self._circuits[labeled]

    def read_node_types(self, data):
        """..."""
        from connsense.pipeline.parallelization import parallelization
        LOG.info("READ node type models %s", data)
        h5, gp = self.get_path("extract-node-types")
        return read_dataset((h5, gp+"/biophysical/morphology/morphology_data"), "extract-node-types")

    def input_morphologies(self, circuit):
        """..."""
        return self.read_node_types("biophysica/morphologies")

    @lazy
    def analyses(self):
        """A TOC for analyses results available in the HDF store.
        """
        def tabulate(analysis, store):
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
        tocs = {"connectivity": {an: tabulate(an, self.get_value_store("connectivity", (an,alysis)))
                                 for an,alysis in self.load_analyses("connectivity").items()},
                "node-types": {an: tabulate(an, self.get_value_store("node-types", (an,alysis)))
                               for an,alysis in self.load_analyses("node-types").items()}}

        return {tic: toc for tic, toc in tocs.items() if toc is not None}

    @lazy
    def circuits(self):
        """Available circuits for which subtargets have been computed."""
        return self.subtarget_gids.index.get_level_values("circuit").unique().to_list()


    def index_values(self, described, subset=None):
        """..."""
        contents = tap.index_contents(described, subset)

    
    def index_contents(self, described, subset=None):
        """..."""
        circuit = described.get("circuit", None)
        connectome = described.get("connectome", None)
        circon = (circuit, connectome)

        def lookup(to_dataset):
            """..."""
            return tuple(v for i, v in enumerate(circon) if ["circuit", "connectome"][i] in to_dataset.index.names)

        dataset = self.pour_subtarget(described["dataset"], subset=lookup)

        if connectome:
            dataset = pd.concat([dataset], axis=0, keys=[connectome], names=["connectome"])
        if circuit:
            dataset = pd.concat([dataset], axis=0, keys=[circuit], names=["circuit"])

        return (dataset if subset is None else dataset.loc[subset]).index.to_frame().reset_index(drop=True)

    def pour_subtarget(self, dataset, subset=None):
        """..."""
        data = self.read_dataset(dataset)

        if subset is None:
            return data

        index = subset(data) if callable(subset) else subset

        if isinstance(index, (pd.Index, tuple, str)):
            return data.loc[index]

        if isinstance(subset, (pd.Series, pd.DataFrame)):
            return data.loc[index.index]

        raise TypeError(f"Not a valid subtarget reference type: {type(index)}")


    def pour_result(self, step, dataset, subset=None):
        """...For example tap.load_analysis('analyze-connectivity', 'pathway-strength)
        """
        dataset = self.pour_subtarget([step, dataset], subset)

        index_values = pd.DataFrame({self.INDICES[var]: self.index_names[var][vals].values
                                     for var, vals in dataset.index.to_frame().iteritems()},
                                    index=dataset.index)
        index = pd.MultiIndex.from_frame(index_values)

        if isinstance(dataset, pd.DataFrame):
            return dataset.set_index(index)
        return pd.Series(dataset.values, index=index, name=dataset.name)

    def pour_analysis_result(self, group, member):
        """..."""
        return self.pour_result(f"analyze-{group}", member)

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
