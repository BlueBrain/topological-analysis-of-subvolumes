"""Interface to the HDFStore where the pipeline stores its data."""
from collections.abc import Iterable, Mapping
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pprint import pformat
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
from connsense.pipeline import ConfigurationError, NotConfiguredError, COMPKEYS


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


    def decompose(self, computation_type, of_quantity):
        """..."""
        from connsense.pipeline.parallelization import parallelization as prl
        parameters = prl.parameterize(computation_type, of_quantity, self._config)
        return {var: val for var, val in parameters.items() if var not in COMPKEYS}

    def load_analyses(self, group):
        """..."""
        topic = f"analyze-{group}"
        try:
            group_analyses = self._config["parameters"][topic]
        except KeyError:
            LOG.warning("No analyses for group %s", group)
            return {}

        analyses = group_analyses["analyses"]
        components = {q: self.decompose(computation_type=topic, of_quantity=q) for q in analyses}

        def compose_analysis(a):
            """..."""
            analysis = {key: deepcopy(value) for key, value in analyses[a].items() if (key in COMPKEYS
                                                                                       and key != "description")}
            analysis["components"] = components[a]

            return analysis

        return {a: compose_analysis(a) for a in analyses}

    def __init__(self, config, in_connsense_h5=None):
        """..."""
        self._config = config
        self._root = locate_store(config, in_connsense_h5)
        self._groups = group_steps(config)
        self._circuits = {None: None}

    @lazy
    def index(self):
        """..."""
        return OrderedDict()

    def set_index(self, variable):
        """..."""
        value = self.create_index(variable)
        self.index[variable] = value
        return value

    def configure_index(self, variable=None):
        """..."""
        configured = self._config["parameters"]["create-index"]["variables"]
        return configured if not variable else configured[variable]

    def create_index(self, variable):
        """Create index for a variable and it's values.
        """
        configured = self._config["parameters"]["create-index"]["variables"]
        described = configured[variable]
        if isinstance(described, pd.Series):
            values = described.values
        elif isinstance(described, Mapping):
            try:
                dataset = described["dataset"]
            except KeyError as kerr:
                raise ConfigurationError(
                    "If not a index values, an index variable description should provide a value for `dataset`"
                ) from kerr
            else:
                transformations = {a: transform for a, transform in described.items() if a != "dataset"}
                values = self.pour_subtarget(dataset, **transformations).values
        elif isinstance(described, Iterable):
            values = list(described)
        else:
            raise ConfigurationError(f"CANNOT create index from description \n {pformat(described)}")

        index = pd.Series(values, index=pd.RangeIndex(0, len(values), 1, name=f"{variable}_id"), name=variable)
        return index

    def subset_index(self, variable, values):
        """..."""
        try:
            index = self.index[variable]
        except KeyError:
            index = self.set_index(variable)

        reverse = pd.Series(index.index.values, name=index.index.name, index=index)

        if isinstance(values, list):
            return pd.Index(reverse.loc[values])

        try:
            d = values["dataset"]
        except KeyError:
            pass
        else:
            computation_type, of_quantity = d

            with h5py.File(self._root, 'r') as hdf_store:
                _, group = self.get_path(computation_type)
                key = '/'.join([group, of_quantity])
                datakey = of_quantity + "/name" if "name" in hdf_store[key] else of_quantity

            LOG.info("subset index for %s datakey %s", d, datakey)
            dataset = self.read_dataset([computation_type, datakey])
            values = dataset.values

        return pd.Index(reverse.loc[values])

    def reindex(self, dataframe, variables):
        """..."""
        LOG.info("Reindex %s to \n%s", dataframe.index.names, pformat(variables))
        indices = {var: self.pour_subtarget(value["dataset"]) for var, value in variables.items()}

        def reverse(index):
            """..."""
            return pd.Series(index.index.values, index=index.values, name=index.name)

        idxframe = dataframe.index.to_frame().reset_index(drop=True)
        reverse = {var: reverse(idx) for var, idx in indices.items()}
        re_idxframe = pd.DataFrame({f"{var}_id": val[idxframe[var].values].values for var, val in reverse.items()})
        return dataframe.set_index(pd.MultiIndex.from_frame(re_idxframe))

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

        if dset not in self.datasets[step]:
            if step.startswith("analyze-"):
                self.datasets[step][dset] = self.analyses['-'.join(step.split('-')[1:])].get(dset, None)
            #elif step == "extract-node-types":
            #    self.datasets[step][dset] = self.read_node_types(dset).sort_index()
            elif step == "extract-node-populations":
                store = matrices.get_store(h5, group+"/"+dset, pd.DataFrame, in_mode="r")
                self.datasets[step][dset] = store.toc
            elif step == "sample-edge-populations":
                store = matrices.get_store(h5, group+"/"+dset, pd.DataFrame, in_mode="r")
                self.datasets[step][dset] = store.toc
            else:
                read = self.get_reader(step)
                self.datasets[step][dset] = read((h5, group+"/"+dset), step).sort_index()

        return self.datasets[step][dset]

    def subtarget(self, definition):
        """..."""
        h5, group = self.get_path("define-subtargets")
        return read_dataset((h5, '/'.join([group, definition, "data"])), "define-subtargets")

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
            circuit = sbtcfg.input_circuit[labeled]
            circuit.variant = labeled
            self._circuits[labeled] = circuit
        return self._circuits[labeled]

    def read_node_types(self, data):
        """..."""
        from connsense.pipeline.parallelization import parallelization
        LOG.info("READ node type models %s", data)
        h5, gp = self.get_path("extract-node-types")

        return read_dataset((h5, gp+"/biophysical/morphology/morphology_data"), "extract-node-types")

    def input_morphologies(self, circuit):
        """..."""
        return self.read_node_types("biophysical/morphology")

    @lazy
    def analyses(self):
        """A TOC for analyses results available in the HDF store.
        """
        # def tabulate(analysis, store):
        #     """..."""
        #     if not store:
        #         return None
        #     try:
        #         return store.toc
        #     except FileNotFoundError as error:
        #         LOG.error("Analysis %s NO TOC found: %s", analysis, error)
        #         return None
        #     return None

        # p = (self._root, self._groups[anzconn.STEP])
        # tocs = {"connectivity": {an: tabulate(an, self.get_value_store("connectivity", (an,alysis)))
        #                          for an,alysis in self.load_analyses("connectivity").items()},
        #         "node-types": {an: tabulate(an, self.get_value_store("node-types", (an,alysis)))
        #                        for an,alysis in self.load_analyses("node-types").items()},
        #         "physiology": {an: tabulate(an, self.get_value_store("physiology", (an,alysis)))
        #                        for an,alysis in self.load_analyses("physiology").items()}}

        # return {tic: toc for tic, toc in tocs.items() if toc is not None}


        def tabulate(group):
            """..."""
            def section(an, analysis):
                """..."""
                LOG.info("Tabulate stores for %s, %s", an, analysis)
                components = analysis.pop("components", {})

                if not components:
                    if not analysis:
                        return {}
                    store = self.get_value_store(group, (an, analysis))
                    if not store:
                        return None
                    try:
                        return store.toc
                    except FileNotFoundError as err:
                        LOG.error("Analysis %s NO TOC found %s", analysis, err)
                        return None
                    return None

                astore = self.get_value_store(group, (an, analysis)) if analysis else None
                stores = {an: astore} if astore else {}

                LOG.info("add components to analysis store: %s", stores)

                for c, component in components.items():
                    LOG.info("Load store component %s", c)
                    cstore = section(f"{an}/{c}", component)
                    if cstore is not None:
                        stores[f"{an}/{c}"] = cstore
                return stores

            tocs = {}
            for an, analysis in self.load_analyses(group).items():
                astore = section(an, analysis)
                if isinstance(astore, Mapping):
                    tocs.update(astore)
                if isinstance(astore, pd.Series):
                    tocs[an] = astore

            return tocs
            return {an: section(an, analysis) for an, analysis in self.load_analyses(group).items()}

        return {g: tabulate(group=g) for g in ["node-types", "composition", "connectivity", "physiology"]}

    @lazy
    def subtarget_circuits(self):
        """Available circuits for which subtargets have been computed."""
        return self.subtarget_gids.index.get_level_values("circuit").unique().to_list()

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

    @lazy
    def index_vars(self):
        """..."""
        return self._config["parameters"]["create-index"]["variables"]

    def pour_dataset(self, variable, values):
        """..."""
        from connsense.pipeline.parallelization import parallelization as prl
        computation_type, of_quantity = prl.describe(values)

        with h5py.File(self._root, 'r') as hdf_store:
            _, group = self.get_path(computation_type)
            key = '/'.join([group, of_quantity])
            datakey = of_quantity + "/data" if "data" in hdf_store[key] else of_quantity

        return self.pour_result(computation_type, datakey)

    def pour_result(self, step, dataset, subset=None):
        """...For example tap.load_analysis('analyze-connectivity', 'pathway-strength)
        """
        data = self.read_dataset([step, dataset])

        if subset is None:
            return data

        index = subset(data) if callable(subset) else subset

        if isinstance(index, (pd.Index, tuple, str)):
            return data.loc[index]

        if isinstance(subset, (pd.Series, pd.DataFrame)):
            return data.loc[index.index]

        raise TypeError(f"Not a valid subtarget reference type: {type(index)}")


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
