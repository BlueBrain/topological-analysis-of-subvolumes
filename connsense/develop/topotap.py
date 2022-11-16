

# Finally, let us collect the code in a module,


"""Interface to the HD5-store where the pipeline stores it's data.
"""
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
from connsense.io import read_config
from connsense.io.write_results import (read as read_dataset,
                                        read_subtargets,
                                        read_node_properties,
                                        read_toc_plus_payload)
from connsense.io import logging
from connsense.pipeline import ConfigurationError, NotConfiguredError, COMPKEYS
from .import parallelization as prl

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


SUBTARGET_ID = "subtarget_id"
CIRCUIT_ID = "circuit_id"
CONNECTOME_ID = "connectome_id"
MTYPE_ID = "mtype_id"
MORPHOLOGY_ID = "morphology_id"

from connsense.pipeline import COMPKEYS, PARAMKEY, ConfigurationError, NotConfiguredError



class TapDataset:
    """A dataset computed by connsense-TAP.
    """
    def __init__(self, tap, dataset, belazy=True):
        """..."""
        self._tap = tap
        self._dataset = dataset
        self._phenomenon, self._quantity = dataset
        self._belazy = belazy

    def index_ids(self, variable):
        """..."""
        try:
            series = self._tap.create_index(variable)
        except KeyError:
            LOG.warn("No values for %s in TAP at %s", variable, tap._root)
            return None

        return pd.Series(series.index.values, name=f"{series.name}_id",
                         index=pd.Index(series.values, name=series.name))

    @lazy
    def parameters(self):
        """Configure parameters for this TapDataset."""
        return self._tap.describe(self._phenomenon, self._quantity)

    @lazy
    def id_subtargets(self):
        """..."""
        return self.index_ids("subtarget")
    @lazy
    def id_circuits(self):
        """..."""
        return self.index_ids("circuit")
    @lazy
    def id_connectomes(self):
        """..."""
        return self.index_ids("connectome")

    @property
    def dataset(self):
        """..."""
        def load_component(c, ):
            """..."""


        def load_slicing(s):
            """..."""
            lazydset = self._tap.pour_dataset(self._phenomenon, self._quantity, slicing=s)
            if self._belazy:
                return lazydset

            dataset = lazydset.apply(lambda l: l.get_value())
            slices = prl.parse_slices(self.parameters["slicing"][s])
            slicing_args = list(prl.flatten_slicing(next(slices)).keys())
            return (pd.concat([g.droplevel(slicing_args) for _,g in dataset.groupby(slicing_args)], axis=1,
                              keys=[g for g,_ in dataset.groupby(slicing_args)], names=slicing_args)
                    .reorder_levels(dataset.columns.names + slicing_args, axis=1))

        if not "slicing" in self.parameters:
            lazydset = self._tap.pour(self._dataset).sort_index()
            return (lazydset if self._belazy
                    else (lazydset.apply(lambda l: l.get_value()) if isinstance(lazydset, pd.Series)
                          else {component: dset.apply(lambda l: l.get_value())}))

        slicings = self.parameters["slicing"]
        dataset = {s: load_slicing(s) for s in slicings}
        try:
            lazyfull = self._tap.pour_dataset(self._phenomenon, self._quantity, slicing="full")
        except KeyError as kerr:
            LOG.warning("No computation results for the full input of %s %s: \n%s", self._phenomenon, self._quantity, kerr)
            dataset["full"] = None
        else:
            dataset["full"] = (lazyfull if self._belazy else lazyfull.apply(lambda l: l.get_value()))
        return dataset

    def index(self, subtarget, circuit=None, connectome=None):
        """Get `connsense-TAP`index for the arguments.
        """
        subtarget_id = self.id_subtargets.loc[subtarget]

        if not circuit:
            assert not connectome, f"connectome must be of a circuit"
            return (subtarget_id,)

        circuit_id = self.id_circuits.loc[circuit]

        if not connectome:
            return (subtarget_id, circuit_id)

        connectome_id = self.id_connectomes.loc[connectome]
        return (subtarget_id, circuit_id, connectome_id)


    def __call__(self, subtarget, circuit=None, connectome=None):
        """Call to get data using the names for (subtarget, circuit, connectome).
        """
        result = self.dataset.loc[self.index(subtarget, circuit, connectome)]

        try:
            evaluate = result.get_value
        except AttributeError:
            pass
        else:
            return evaluate()

        if len(result) == 1:
            return result.iloc[0].get_value()
        return result



class HDFStore:
    """An interface to the H5 data extracted by connsense-TAP.
    """
    def __init__(self, config, in_connsense_h5=None):
        """Initialize an instance of connsense-TAP HDFStore.

        config: Path to a YAML / JSON file that configures the pipeline, or a Mapping resulting from reading
        ~       such a config file.
        in_consense_h5: Path to the connsense-TAP H5 store if different from the one configured
        ~               This can be used for testing the data produced in individual compute-nodes during
        ~               a pipeline run.
        """
        self._config = read_config.read(config) if not isinstance(config, Mapping) else config
        self._root = locate_store(self._config, in_connsense_h5)
        self._groups = group_steps(self._config)


    @lazy
    def parameters(tap):
        """Section `parameters` of the config, loaded without `create-index`.
        """
        return {param: config for param, config in tap._config["parameters"].items() if param != "create-index"}
    
    
    def read_parameters(tap, computation_type, quantity):
        """..."""
        pkey = tap.get_paramkey(computation_tap)
        if '/' not in quantity:
            return tap.parameters[computation_tap][pkey][quantity]

        group, quantity = quantity.split('/')
        return tap.parameters[computation_tap][pkey[group][quantity]

    def get_paramkey(tap, computation_type):
        """..."""
        return PARAMKEY[computation_type]
    

    def describe(tap, computation_type=None, of_quantity=None):
        """...Describe the dataset associated with a `(computation_type, of_quantity)`.
    
        computation_type: should be an entry in the configuration section parameters,
        ~                 if not provided, all computation-types
        of_quantity: should be an entry under argued `computation_type`
        ~            if not provided, all quantities under `computation_type`
        """
        if not computation_type:
            assert not of_quantity, "because a quantity without a computation-type does not make sense."
            return {c: tap.describe(computation_type=c) for c in tap.parameters}
    
        try:
            config = tap.parameters[computation_type]
        except KeyError as kerr:
            LOG.error("computation-type %s not configured! Update the config, or choose from \n%s",
                      computation_type, pformat(tap.parameters.keys()))
            raise NotConfiguredError(computation_type) from kerr
    
        paramkey = tap.get_paramkey(computation_type)
        try:
            config = config[paramkey]
        except KeyError as kerr:
            LOG.error("Missing %s entries in %s config.", paramkey, computation_type)
            raise ConfigurationError(f"{paramkey} entries for {computation_type}")
    
        def describe_quantity(q):
            if '/' not in q:
                config_q = {"description": config[q].get("description", "NotAvailable"),
                            "dataset": (computation_type, q)}
                for k, v in config[q].items():
                    if k != "description":
                        config_q[k] = v
                return config_q
    
            g, q = q.split('/')
            config_g = {"description": config[g].get("description", "NotAvailable")}
            config_g[q] = {"description": config[g].get("description", "NotAvailable"),
                           "dataset": (computation_type, f"{g}/{q}")}
            for k, v in config[g][q].items():
                if k != "description":
                    config_g[q][k] = v
            return config_g
    
    
        if not of_quantity:
            return [describe_quantity(q) for q in config]
    
        return describe_quantity(q=of_quantity)
    
    

    def get_path(tap, computation_type):
        """..."""
        return (tap._root, tap._groups[computation_type])
    
    def pour_dataset(tap, computation_type, of_quantity, slicing=None):
        """..."""
        connsense_h5, hdf_group = tap.get_path(computation_type)
        dataset = '/'.join([hdf_group, of_quantity] if not slicing else [hdf_group, of_quantity, slicing])
    
        with h5py.File(tap._root, 'r') as hdf:
            if "data" in hdf[dataset]:
                dataset = '/'.join([dataset, "data"])
    
        if computation_type == "extract-node-populations":
            return matrices.get_store(connsense_h5, dataset, pd.DataFrame).toc
    
        if computation_type == "extract-edge-populations":
            return read_toc_plus_payload((connsense_h5, dataset), "extract-edge-populations")
    
        if computation_type.startswith("analyze-"):
            return tap.pour_analyses(computation_type, of_quantity, slicing)
    
        return read_dataset((connsense_h5, dataset), computation_type)
    
    def pour(tap, dataset):
        """For convenience, allow queries with tuples (computation_type, of_quantity).
        """
        return tap.pour_dataset(*dataset)
    

    
    def decompose(self, computation_type, of_quantity):
        """Some computations may have components.
        We need to strip computation keys from the config, and return the resulting dict.
        """
        parameters = prl.parameterize(computation_type, of_quantity, self._config)
        return {var: val for var, val in parameters.items() if var not in COMPKEYS}
    
    
    def pour_analyses(tap, computation_type, quantity, slicing=None):
        """Pour the results of running an analysis computation.
        """
        LOG.info("Pour analyses for %s quantity %s", computation_type, quantity)
        connsense_h5, hdf_group = tap.get_path(computation_type)
        paramkey = tap.get_paramkey(computation_type)
    
        def pour_component(c, parameters):
            """..."""
            LOG.info("Pour %s %s component %s: \n%s\n from store %s", computation_type, quantity, c,
                     pformat(parameters), (connsense_h5, '/'.join([hdf_group, c])))
    
            dataset = '/'.join([hdf_group, quantity, c] if not slicing else [hdf_group, quantity, c, slicing])
            store = matrices.get_store(connsense_h5, dataset, parameters["output"], in_mode='r')
            return store.toc if store else None
    
        components = tap.decompose(computation_type, quantity)
        if not components:
            dataset = '/'.join([hdf_group, quantity] if not slicing else [hdf_group, quantity, slicing])
            parameters = tap.parameters[computation_type][paramkey][quantity]
            store = matrices.get_store(connsense_h5, dataset, parameters["output"], in_mode='r')
            return store.toc if store else None
    
        return {'/'.join([quantity, c]): pour_component(c, parameters) for c, parameters in components.items()}
    
    

    def create_index(tap, variable):
        """..."""
        described = tap._config["parameters"]["create-index"]["variables"][variable]
    
        if isinstance(described, pd.Series):
            values = descibed.values
        elif isinstance(described, Mapping):
            try:
                dataset = described["dataset"]
            except KeyError as kerr:
                LOG.error("Cannot create an index for %s of no dataset in config.", variable)
                raise ConfigurationError("No create-index %s dataset", variable)
            return tap.pour(dataset)
        elif isinstance(described, Iterable):
            values = list(described)
        else:
            raise ConfigurationError(f"create-index %s using config \n%s", pformat(described))
    
        return pd.Series(values, name=variable, index=pd.RangeIndex(0, len(values), 1, name=f"{variable}_id"))
    
    

    @lazy
    def subtargets(tap):
        """Subtargets in connsense-TAP
        """
        definitions = tap.describe("define-subtargets")
        pour_subtargets = lambda dataset: tap.pour(("define-subtargets", dataset))
    
        if len(definitions) == 0:
            LOG.warning("No subtargets configured!")
            return None
    
        def of_(definition):
            """..."""
            LOG.info("Load dataset %s: \n%s", definition["dataset"], pformat(definition["description"]))
            _, group = definition["dataset"]
            subtargets = pour_subtargets(f"{group}/name")
            try:
                info = pour_subtargets(f"{group}/info")
            except KeyError:
                return subtargets
            return pd.concat([subtargets, info], axis=1)
    
        if len(definitions) == 1:
            return of_(definitions[0])
        return {definition["dataset"][1]: of_(definition) for definition in definitions}
    
    

    @lazy
    def nodes(tap):
        """Nodes in connsense-TAP
        """
        populations = tap.describe("extract-node-populations")
    
        if len(populations) == 0:
            LOG.warning("No populations configured!")
            return None
    
        def of_(population):
            """..."""
            LOG.info("Load dataset %s: \n%s", population["dataset"], pformat(population["description"]))
            return TapDataset(tap, population["dataset"])
    
        if len(populations) == 1:
            return of_(populations[0])
        return {population["dataset"][1]: of_(population) for population in populations}
    
    

    @lazy
    def adjacency(tap):
        """Adjacency matrices of subtargets in connsense-TAP
        """
        populations = tap.describe("extract-edge-populations")
    
        if len(populations) == 0:
            LOG.warning("No populations configured!")
            return None
    
        def of_(population):
            """..."""
            LOG.info("Load dataset %s: \n%s", population["dataset"], pformat(population["description"]))
            return TapDataset(tap, population["dataset"])
    
        if len(populations) == 1:
            return of_(populations[0])
        return {population["dataset"][1]: of_(population) for population in populations}
    

    def get_phenomenon(tap, computation_type):
        """..."""
        analysis = computation_type.split('-')
        if analysis[0] != "analyze":
            LOG.warn("%s is not an analysis", computaiton_tyoe)
            return None
    
        return '-'.join(analysis[1:])
    
    def find_analyses(tap, phenomenon=None):
        """Find all analyses of phenomenon in the config.
        """
    
        if phenomenon:
            analyzed = tap.parameters[f"analyze-{phenomenon}"]
            return analyzed["analyses"]
    
        return {p: tap.find_analyses(phenomenon=p) for p in tap.phenomena}
    
    @property
    def phenomena(tap):
        """The analyze phenomena.
        """
        return [tap.get_phenomenon(computation_type=c) for c in tap.parameters if c.startswith("analyze-")]
    
    def describe_analyses(tap, phenomenon=None):
        """..."""
        analyze = "analyze-{}".format
        if phenomenon:
            return tap.describe(analyze(phenomenon))
        return {p: tap.describe(analyze(p)) for p in tap.phenomena}
    
    @lazy
    def analyses(tap):
        """..."""
        analyses = tap.describe_analyses()
        return {phenomenon: {q["dataset"][1]: TapDataset(tap, q["dataset"]) for q in quantities}
                for phenomenon, quantities in analyses.items()}
