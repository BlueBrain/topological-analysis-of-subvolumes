

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

import numpy as np
import pandas as pd

from bluepy import Direction, Synapse, Cell

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
    def __init__(self, tap, dataset, belazy=True, transform=None):
        """..."""
        self._tap = tap
        self._dataset = dataset
        self._phenomenon, self._quantity = dataset
        self._belazy = belazy
        self._transform = transform

    def load(self):
        """.."""
        return TapDataset(self._tap, self._dataset, belazy=False, transform=self._transform)

    def index_ids(self, variable):
        """..."""
        try:
            series = self._tap.create_index(variable)
        except KeyError:
            LOG.warn("No values for %s in TAP at %s", variable, self._tap._root)
            return None

        return pd.Series(series.index.values, name=f"{series.name}_id",
                         index=pd.Index(series.values, name=series.name))

    @lazy
    def parameters(self):
        """Configure parameters for this TapDataset."""
        description = self._tap.describe(self._phenomenon, self._quantity)
        return description[self._quantity] if '/' in self._quantity else description

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

    @lazy
    def id_mtypes(self):
        """..."""
        return self.index_ids("mtype")

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

    @lazy
    def toc(self):
        """TAP Table of Contents"""
        primary_ids = ["subtarget_id", "circuit_id", "connectome_id"]
        secondary_ids = [var_id for var_id in self.dataset.index.names if var_id not in primary_ids]



    @lazy
    def dataset(self):
        """..."""
        from connsense.develop.parallelization import DataCall

        def call(dataitem):
            """..."""
            return DataCall(dataitem, self._transform)

        def load_component(c):
            """..."""
            raise NotImplementedError("INPROGRESS")

        def load_slicing(s):
            """..."""
            slicing_cfg = self.parameters["slicing"][s]
            try:
                lazydset = self._tap.pour_dataset(self._phenomenon, self._quantity, slicing=s)
            except KeyError as kerr:
                LOG.warning("No computation results of %s %s for input slicing %s: \n%s",
                            self._phenomenon, self._quantity, s, kerr)
                return None

            if self._belazy:
                return lazydset

            dataset = lazydset.apply(lambda l: l.get_value())
            slices = prl.parse_slices(slicing_cfg)

            if slicing_cfg["compute_mode"] in ("execute", "EXECUTE"):
                return pd.concat(dataset.values, keys=dataset.index)

            slicing_args = list(prl.flatten_slicing(next(slices)).keys())
            sliced = pd.concat([g.droplevel(slicing_args) for _,g in dataset.groupby(slicing_args)], axis=1,
                                keys=[g for g,_ in dataset.groupby(slicing_args)], names=slicing_args)
            return (sliced if isinstance(dataset, pd.Series) else
                    sliced.reorder_levels(dataset.columns.names + slicing_args, axis=1))

        if not "slicing" in self.parameters:
            lazydset = self._tap.pour(self._dataset).sort_index()
            if not isinstance(lazydset, pd.Series):
                raise TypeError(f"Unexpected type of TAP-dataset {type(lazydset)}.\n"
                                "If you defined this TapDataset for measurement of a phenomenon/quantity,\n"
                                "we are still figuring out how to handle that. We may remove such a "
                                "possibility and define a TapDatasetGroup.\n "
                                "Thus we will keep TapDataset to contain only a single type of data"
                                "i.e. data that originates from a single TAP-computation")

            lazycalls = lazydset.apply(call)
            return lazycalls if self._belazy else lazycalls.apply(lambda l: l())

        slicings = self.parameters["slicing"]
        dataset = {s: load_slicing(s) for s in slicings if s not in ("description", "do-full")}
        try:
            lazyfull = self._tap.pour_dataset(self._phenomenon, self._quantity, slicing="full")
        except KeyError as kerr:
            LOG.warning("No computation results for the full input of %s %s: \n%s", self._phenomenon, self._quantity, kerr)
            dataset["full"] = None
        else:
            dataset["full"] = (lazyfull if self._belazy else lazyfull.apply(lambda l: l.get_value()))
        return dataset

    def summarize(self, method):
        """..."""
        if callable(method):
            return TapDataset(self._tap, self._dataset, belazy=self._belazy, transform=method)

        if isinstance(method, (list, str)):
            return TapDataset(self._tap, self._dataset, belazy=self._belazy,
                              transform=lambda measurement: measurement.agg(method))

        raise NotImplementedError(f"Method to handle of type {type(method)}")


    def check_reindexing(self):
        """Figure out other variables (than subtarget, circuit, connectome) in the index and name them."""
        from connsense.develop import parallelization as prl

        datasets = {variable: config["dataset"] for variable, config in self.parameters["input"].items()
                    if isinstance(config, Mapping) and "dataset" in config}
        try:
            return datasets["reindex"]
        except KeyError:
            pass

        for _, variable_dataset in datasets.items():
            variable_inputs = prl.parameterize(*variable_dataset, self._tap._config)["input"]
            for _, input_dataset in variable_inputs.items():
                try:
                    return input_dataset["reindex"]
                except KeyError:
                    pass
        return None

    def name_index(self, result, variable_id, index_only_variable=False):
        """..."""
        varindex = result.index.get_level_values(variable_id)

        if variable_id and variable_id.endswith("_id"):
            variable = variable_id.strip("_id")
            return pd.Series(self._tap.create_index(variable).loc[varindex].values, name=variable,
                             index=(varindex if index_only_variable else result.index))

        return pd.Series(varindex.values, name=variable_id, index=result.index)

    def name_reindex_variables(self, result):
        """..."""
        assert ("subtarget_id"  not in result.index.names
                and "circuit_id" not in result.index.names
                and "connectome_id" not in result.index.names)

        of_indices = pd.DataFrame({variable:(self._tap.create_index(variable)
                                             .loc[result.index.get_level_values(f"{variable}_id").values]
                                             .values)
                                   for variable in self.check_reindexing()})

        named_index = pd.MultiIndex.from_frame(of_indices)

        if isinstance(result, pd.Series):
            return pd.Series(result.values, index=named_index)

        if isinstance(result, pd.DataFrame):
            return result.set_index(named_index)

        raise TypeError("Unexpect type %s of result", type(result))

    def name_index_variables(self, result):
        """..."""
        assert ("subtarget_id"  not in result.index.names
                and "circuit_id" not in result.index.names
                and "connectome_id" not in result.index.names)

        varnames = pd.concat([self.name_index(result, var) for var in result.index.names],
                             axis=1)
        variables = pd.MultiIndex.from_frame(varnames)

        assert isinstance(result, pd.Series), f"Illegal type {type(result)}"
        return pd.Series(result.values, name=result.name, index=variables)

    def __call__(self, subtarget, circuit=None, connectome=None, *, control=None, slicing=None):
        """Call to get data using the names for (subtarget, circuit, connectome).
        """
        idx = self.index(subtarget, circuit, connectome)

        if "slicing" not in self.parameters:
            result = self.dataset.loc[idx]

            try:
                evaluate = result.get_value
            except AttributeError:
                pass
            else:
                return evaluate()

            if len(result) == 1:
                return result.iloc[0].get_value()

            return self.name_index_variables(result[~result.index.duplicated(keep="first")])

        slicings = {key for key in self.parameters["slicing"] if key not in ("do-full", "description")}

        if not slicing:
            if "full" not in self.dataset:
                LOG.info("TapDataset %s was configured with slicings, but not full."
                         "\n Please provide a `slicing=<value>`.", self._dataset)
                raise ValueError("TapDataset %s was configured with slicings, but not full."
                                 "\n Please provide a `slicing=<value>`."%(self._dataset,))
            return self.dataset["full"].loc[idx]

        if slicing not in slicings:
            LOG.warning("Slicing %s was not among those configured: \n%s", slcicing, slicings)
            raise ValueError("Slicing %s was not among those configured: \n%s"%(slcicing, slicings))

        return self.dataset[slicing].loc[idx]

    @lazy
    def variable_ids(self):
        """..."""
        if not isinstance(self.dataset, Mapping):
            return self.dataset.index.names
        return {component: (dset.index.names if dset is not None else None)
                for component, dset in self.dataset.items()}


    def frame_component(self, c=None, name_indices=True):
        """..."""
        LOG.info("Frame TapDataset (%s/%s) component %s",
                 self._phenomenon, self._quantity, c)

        if isinstance(c, str):
            assert isinstance(self.dataset, Mapping)
            component = self.dataset[c]
            variable_ids = self.variable_ids[c]
        else:
            assert c is None and not isinstance(self.dataset, Mapping), c
            component = self.dataset
            variable_ids = self.variable_ids

        if component is None:
            LOG.warning("No dataset found for component %s", c)
            return None

        def cleanup_index(r):
            try:
                return r.droplevel(self.variable_ids)
            except (IndexError, KeyError):
                return r

        if name_indices:
            index = pd.concat([self.name_index(component, varid)
                               for varid in variable_ids], axis=1)

            if isinstance(component, pd.Series):
                s = pd.Series(component.values, index=pd.MultiIndex.from_frame(index))
                s = s[~s.index.duplicated(keep="last")]
                return (pd.concat([cleanup_index(v) for v in s.values], keys=s.index)
                        if not self._belazy else s)

            assert isinstance(component, pd.DataFrame), f"Invalid type {type(component)}"
            return component.set_index(pd.MultiIndex.from_frame(index))

        if isinstance(component, pd.Series):
            component = component[~component.index.duplicated(keep="last")]
            return pd.concat(component.values, keys=component.index)

        assert isinstance(component, pd.DataFrame), f"Invalid type {type(component)}"
        return component

    @lazy
    def frame(self):
        """..."""
        if isinstance(self.dataset, Mapping):
            return {c: self.frame_component(c) for c in self.dataset}
        return self.frame_component()
        #index = pd.concat([self.name_index(self.dataset, varid) for varid in self.variable_ids], axis=1)
        #series = pd.Series(self.dataset.values, index=pd.MultiIndex.from_frame(index))
        #series = series[~series.index.duplicated(keep="first")]
        #return pd.concat(series.values, keys=series.index)

    def frame_fun(self, subtarget, circuit, connectome, summarize=None):
        """..."""
        data = self(subtarget, circuit, connectome)
        try:
            data = data.apply(lambda d: d())
        except TypeError:
            pass
        frame = pd.concat(data.values, keys=data.index)
        return frame.groupby(data.index.names).agg(summarize) if summarize else frame

    def name_indices(self, component=None):
        """..."""
        if isinstance(component, str):
            assert isinstance(self.dataset, Mapping)
            data = self.dataset[component]
            variable_ids = self.variable_ids[component]
            others = None
        elif component is None:
            assert component is None and not isinstance(self.dataset, Mapping), component
            data = self.dataset
            variable_ids = self.variable_ids
            others = None
        else:
            assert isinstance(component, (pd.Series, pd.DataFrame))
            data = component
            variable_ids = [varid for varid in component.index.names if varid.endswith("_id")]
            other_ids = [varid for varid in component.index.names if not varid.endswith("_id")]
            others = component.index.to_frame()[other_ids]

        named = pd.concat([self.name_index(data, varid) for varid in variable_ids], axis=1)
        return named.join(others) if others is not None else named

    @lazy
    def inputs(self):
        """..."""
        from connsense.develop import parallelization as devprl
        inputs = devprl.generate_inputs(self._dataset, self._tap._config)
        return inputs

    def input(self, subtarget, circuit=None, connectome=None, *, controls=None):
        """..."""
        from connsense.develop import parallelization as devprl
        toc_idx = self.index(subtarget, circuit, connectome)
        inputs = self.inputs.loc[toc_idx]
        if not isinstance(inputs.index, pd.MultiIndex):
            inputs.index = pd.MultiIndex.from_tuples([(v,) for v in inputs.index.values],
                                                     names=[inputs.index.name])

        if not controls:
            return inputs.apply(lambda l: l()) if self._belazy else inputs

        try:
            configured = self.parameters["controls"]
        except KeyError as kerr:
            Log.warning("No controls have been set for the TapDataset %s", self._dataset)
            raise kerr

        controls_configured = devprl.load_control(configured)

        argued = [c for c, _, _ in controls_configured if c.startswith(f"{controls}-")]

        controlled = pd.concat([inputs.xs(c, level="control") for c in argued], axis=0,
                               keys=[c.replace(controls, '')[1:] for c in argued],
                               names=[controls])
        return controlled.apply(lambda l: l()) if not self._belazy else controlled
    @lazy
    def controls(self):
        try:
            control_type = self.inputs.index.get_level_values("control")
        except KeyError as err:
            LOG.error("No control configured ?")
            raise err

        controls = self.inputs[control_type != "original"]
        idxframe = (self.name_indices(component="full") if isinstance(self.dataset, Mapping)
                    else self.name_indices())
        return pd.Series(controls.values,
                         index=pd.MultiIndex.from_frame(idxframe.loc[controls.index]))


    def apply_control(self, algorithm=None, seed=None, *, subtarget, circuit=None, connectome=None,
                      datacall=False):
        """...Get controlled input (datacalls).
        Note: This method makes sense only if this TapDataset is of an analysis.
        Returned datatype depends on the arguments.
        If a subtarget is provided, the randomization algorithm will be run by calling
        the datacall -- so you will get the randomized inputs to this analysis TapDataset.

        Parameters
        ------------
        algorithm: The name of the algorithm to load randomizations for
        seed: One, and only one, of the seeds in the connsense-pipeline
        subtarget: Name of the subtarget to control
        circuit: Name of the circuit whose inputs should be controled.
        ~        If None, and there is only one configured circuit, the circuit level will be removed.
        connectome: Name if the connectome whose inputs should be controlled
        ~        If None, and there is only one configured connectome, the connectome level will be removed.
        """
        from .import parallelization as prl
        controls = self.controls.loc[subtarget]

        if circuit:
            controls = controls.loc[circuit]
        else:
            if len(self._tap.circuits) == 1:
                controls = controls.droplevel("circuit")

        if connectome:
            controls = controls.loc[connectome]
        else:
            if len(self._tap.connectomes) == 1:
                controls = controls.droplevel("connectome")

        if not algorithm:
            return controls

        try:
            configured = self.parameters["controls"]
        except KeyError as kerr:
            Log.warning("No controls have been set for the TapDataset %s", self._dataset)
            raise kerr
        controls_configured = prl.load_control(configured)

        algorithm_index = [c for c, _, _ in controls_configured if c.startswith(f"{algorithm}-")]
        controls = controls.loc[algorithm_index]

        if seed is None:
            return controls

        _datacall = controls.loc[f"{algorithm}-{seed}"]

        return _datacall if datacall else _datacall()

    @lazy
    def slicings(self):
        """Slicings for this (analysis) TapDataset, read from connsense-pipeline.yaml
        """
        try:
            configured = self.parameters["slicing"]
        except KeyError:
            raise ConfigurationError(f"Slicing  not configured for {self._dataset}")

        return {key: value for key, value in configured.items()
                if key.lower() not in ("do-full", "do_full", "description")}

    @lazy
    def sliced_inputs(self):
        """..."""
        from .import parallelization as prl

        def slice(s):
            inputs = prl.generate_inputs(self._dataset, self._tap._config, slicing=s,
                                         datacalls_for_slices=True)
            idxframe = self.name_indices(inputs)
            return pd.Series(inputs.values,
                             index=pd.MultiIndex.from_frame(idxframe.loc[inputs.index]))

        return {s: slice(s) for s in self.slicings}

    def slice(self, s, value=None, *, subtarget,  circuit=None, connectome=None,
              datacall=False):
        """..."""
        from .import parallelization as prl
        slices = self.slices[s]

        if circuit:
            slices = slices.loc[circuit]
        else:
            if len(self._tap.circuits) == 1:
                slices = slices.droplevel("circuit")

        if connectome:
            slices = slices.loc[connectome]
        else:
            if len(self._tap.connectomes) == 1:
                slices = slices.droplevel("connectome")

        if not value:
            return slices

        return slices.loc[value]

    def load_adjacency_controls(self, subtargets, control_names, belazy=False):
        """...Load adjacency and control them by the provided name.
        Return pandas Series for the controls, each with an adjacency matrix.
        """
        raise NotImplementedError("INPROGRESS")

class HDFStore:
    """An interface to the H5 data extracted by connsense-TAP.
    """
    def __init__(self, config, in_connsense_h5=None):
        """Initialize an instance of connsense-TAP HDFStore.

        config: Path to a YAML / JSON file that configures the pipeline,
        or a Mapping resulting from reading such a config file.

        in_consense_h5: Path to the connsense-TAP H5 store if different

        from the one configured This can be used for testing the data produced in
        individual compute-nodes during a pipeline run.
        """
        self._config = read_config.read(config) if not isinstance(config, Mapping) else config
        self._root = locate_store(self._config, in_connsense_h5)
        self._groups = group_steps(self._config)
        self._circuits = {}

        self._external = locate_store(self._config, "external.h5")

    @lazy
    def parameters(tap):
        """Section `parameters` of the config, loaded without `create-index`.
        """
        return {param: config for param, config in tap._config["parameters"].items() if param != "create-index"}
    
    
    def read_parameters(tap, computation_type, quantity, slicing=None):
        """..."""
        from copy import deepcopy
        from connsense.analyze_connectivity import matrices
    
        pkey = tap.get_paramkey(computation_type)
        if not slicing or slicing == "full":
            if '/' not in quantity:
                configured = deepcopy(tap.parameters[computation_type][pkey][quantity])
            else:
                group, quantity = quantity.split('/')
                configured = deepcopy(tap.parameters[computation_type][pkey][group][quantity])
    
            if slicing == "full":
                configured.pop("slicing")
    
            return configured
    
        params = tap.read_parameters(computation_type, quantity)
    
        try:
            cfg_slicings = params.pop("slicing")
        except KeyError:
            raise ConfigurationError(f"No slicings were configured: \n{pformat(params)}")
    
        try:
            slicing_params = cfg_slicings[slicing]
        except KeyError:
            raise ConfigurationError(f"slicing {slicing} not among configure:\n{pformat(cfg_slicings)}")
    
        if slicing_params.get("compute_mode", ("EXECUTE")) in ("execute", "EXECUTE"):
            params["output"] = matrices.type_series_store(params["output"])
    
        return params

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
    
            try:
                g, qq = q.split('/')
            except ValueError:
                try:
                    g, qq, qqq = q.split('/')
                except Exception as err:
                    LOG.warning("Unable to parse computation %s, %s", computation_type, quantity)
                    raise err
                #qq = '/'.join([qq, qqq])
    
    
            config_g = {"description": config[g].get("description", "NotAvailable")}
            config_g[q] = {"description": config[g].get("description", "NotAvailable"),
                           "dataset": (computation_type, f"{g}/{qq}")}
            for k, v in config[g][qq].items():
                if k != "description":
                    config_g[q][k] = v
            return config_g
    
    
        if not of_quantity:
            return [describe_quantity(q) for q in config]
    
        return describe_quantity(q=of_quantity)
    
    

    def get_circuit(self, labeled):
        """..."""
        if labeled not in self._circuits:
            sbtcfg = SubtargetsConfig(self._config)
            circuit = sbtcfg.input_circuit[labeled]
            circuit.variant = labeled
            self._circuits[labeled] = circuit
        return self._circuits[labeled]
    
    @lazy
    def circuits(self):
        return self.index_variable("circuit")
    
    @lazy
    def connectomes(self):
        return self.index_variable("connectome")

    def get_path(tap, computation_type):
        """..."""
        return (tap._root, tap._groups[computation_type])
    
    def pour_dataset(tap, computation_type, of_quantity, slicing=None,
                     *, subset=None):
        """..."""
        def slice_subset(dataset):
            """..."""
            return dataset.loc[subset] if subset is not None else dataset
    
        if (circuit_dataset:=tap.check_circuit_dataset(computation_type, of_quantity)):
            LOG.info("Circuit dataset %s: %s",
                     (computation_type, of_quantity), circuit_dataset[0])
            assert slicing is None,\
                "We do not apply slicing to circuit datasets, only analyses"
            return slice_subset(circuit_dataset[0])
    
        cnsh5, hdf_group = tap.get_path(computation_type)
        dataset = '/'.join([hdf_group, of_quantity] if not slicing
                           else [hdf_group, of_quantity, slicing])
    
        with h5py.File(tap._root, 'r') as hdf:
            if "data" in hdf[dataset]:
                dataset = '/'.join([dataset, "data"])
    
        if computation_type == "extract-node-populations":
            assert slicing is None,\
                "We do not apply slicing to extracted node populations, only analyses"
            return slice_subset(matrices
                                .get_store(cnsh5, dataset, pd.DataFrame, in_mode='r').toc)
    
        if computation_type == "extract-edge-populations":
            assert slicing is None,\
                "We do not apply slicing to extracted edge populations, only analyses"
            return slice_subset(read_toc_plus_payload((cnsh5, dataset),
                                                      "extract-edge-populations"))
    
        if computation_type.startswith("analyze-"):
            return slice_subset(tap.pour_analyses(computation_type, of_quantity, slicing))
    
        return slice_subset(read_dataset((cnsh5, dataset), computation_type))
    
    
    def pour(tap, dataset):
        """For convenience, allow queries with tuples (computation_type, of_quantity).
        """
        return tap.pour_dataset(*dataset)
    
    
    def assign_subtargets(tap, to_report_ids=False):
        """..."""
        def assign_subtargets_for_circuit(_id):
            def series(of_gids):
                return pd.Series(of_gids, name="gid",
                                 index=pd.RangeIndex(0, len(of_gids), 1, name="node_id"))
    
            circuit_subtarget_gids = tap.subtarget_gids.xs(_id, level="circuit_id")
            assignment = (pd.concat([series(gids) for gids in circuit_subtarget_gids],
                                    keys=circuit_subtarget_gids.index)
                          .reset_index().set_index("gid"))
            if to_report_ids: return assignment
    
            subtargets = tap.subtargets.subtarget.loc[assignment.subtarget_id.values].values
            return pd.Series(subtargets, name="subtarget", index=assignment.index)
    
        circuit_ids = tap.subtarget_gids.index.get_level_values("circuit_id").unique()
    
        return pd.Series([assign_subtargets_for_circuit(c) for c in circuit_ids],
                         name="subtargets", index=circuit_ids)
    
    
    def check_circuit_dataset(tap, computation_type, of_quantity):
        """..."""
        if not computation_type.startswith("circuit-"):
            return False
    
        if computation_type == "circuit-node-populations":
            from connsense.develop import parallelization
            circuits = (tap.create_index("circuit")
                        .apply(lambda c: parallelization.input_circuit(c, tap._config)))
            try:
                extraction = tap.describe("extract-node-populations", of_quantity)
            except KeyError as kerr:
                raise KeyError("No extraction for node population %s was defined."
                               "Circuit datasets may be defined for corresponding "
                               "extractions"%(of_quantity,))
    
            of_cells = extraction.get("kwargs", {}).get("properties", None)
            def get_cells(circuit):
                cells = circuit.cells.get(properties=of_cells)
                cells.index.rename(Cell.ID, inplace=True)
                return cells
            circuit_nodes = circuits.apply(get_cells).rename("cell_properties")
            subtargets = tap.assign_subtargets()
            dataset = (pd.merge(circuit_nodes, subtargets,
                                left_index=True, right_index=True)
                       .apply(lambda c: c.cell_properties.join(c.subtargets), axis=1))
            return (dataset,)
    
        raise NotImplementedError(f"Circuit data {computation_type}")

    
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
        cnsh5, hdf_group = tap.get_path(computation_type)
    
    #    if quantity == "psp/traces":
    #        return pd.read_hdf(cnsh5, '/'.join([hdf_group, quantitye))
    
    
        def pour_component(c, parameters):
            """..."""
            LOG.info("Pour %s %s component %s: \n%s\n from store %s",
                     computation_type, quantity, c, pformat(parameters),
                     (cnsh5, '/'.join([hdf_group, c])))
    
            dataset = '/'.join([hdf_group, quantity, c] if not slicing
                               else [hdf_group, quantity, c, slicing])
            store = matrices.get_store(cnsh5, dataset, parameters["output"], in_mode='r')
            return store.toc if store else None
    
        components = tap.decompose(computation_type, quantity)
        if not components:
            dataset = '/'.join([hdf_group, quantity] if not slicing
                               else [hdf_group, quantity, slicing])
            parameters = tap.read_parameters(computation_type, quantity, slicing)
            store = matrices.get_store(cnsh5, dataset, parameters["output"], in_mode='r')
            return store.toc if store else None
    
        return {'/'.join([quantity, c]): pour_component(c, parameters)
                for c, parameters in components.items()}

    def create_index(tap, variable):
        """..."""
        described = tap._config["parameters"]["create-index"]["variables"][variable]
    
        if isinstance(described, pd.Series):
            values = described.values
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
    
    
    
    def index_variable(tap, name, value=None):
        """..."""
        import numpy as np
    
        index = tap.create_index(variable=name)
    
        if value is not None and not isinstance(value, (list, np.ndarray)):
            idx = index.index.values[index == value]
            return idx[0] if len(idx) == 1 else idx
    
        reverse = pd.Series(index.index.values, name=index.name, index=pd.Index(index.values, name=index.name))
        return reverse.reindex(value) if value is not None else reverse
    

    def pour_subtargets(tap, dataset):
        """..."""
        return tap.pour(("define-subtargets", dataset))
    @lazy
    def subtargets(tap):
        """Subtargets in connsense-TAP."""
        definitions = tap.describe("define-subtargets")
    
        if len(definitions) == 0:
            LOG.warning("No subtargets configured!")
            return None
    
        def of_(definition):
            """..."""
            LOG.info("Load dataset %s: \n%s", definition["dataset"], pformat(definition["description"]))
            _, group = definition["dataset"]
            subtargets = tap.pour_subtargets(f"{group}/name")
            try:
                info = tap.pour_subtargets(f"{group}/info")
            except KeyError:
                return subtargets
            return pd.concat([subtargets, info], axis=1)
    
        if len(definitions) == 1:
            return of_(definitions[0])
        return {definition["dataset"][1]: of_(definition) for definition in definitions}
    
    
    @lazy
    def subtarget_gids(tap):
        """..."""
        definitions = tap.describe("define-subtargets")
    
        if len(definitions) == 0:
            LOG.warning("No subtargets configured!")
            return None
    
        def of_(definition):
            """..."""
            LOG.info("Load dataset %s: \n%s", definition["dataset"], pformat(definition["description"]))
            _, group = definition["dataset"]
            return tap.pour_subtargets(f"{group}/data")
    
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
            LOG.warn("%s is not an analysis", computation_type)
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
    
    def get_analyses(tap, phenomenon, quantity, control=None, slicing=None):
        """..."""
        dataset = tap.analyses[phenomenon][quantity].load().dataset
        print("get analyses dataset", dataset.keys())
        return dataset[slicing] if slicing else dataset
    
    def load_controls(tap, phenomenon, quantity, label=None, subtargets=None):
        """..."""
        pass
    
    def load_adjacency_controls(tap, analysis, subtargets, control_name):
        """..."""
        pass
    
    

    def get_controls(tap, analysis, controls=None):
        """..."""
        from .import parallelization as prl
        ctrls = prl.control_inputs(of_computation=analysis, in_config=tap._config, using_tap=tap)
        return ctrls.reorder_levels([l for l in ctrls.index.names if l!="control"]+["control"])

    def append_external(tap, dataset, data):
        """Append an external dataset to a TAP instance.
        dataset: [computation_type, quantity] that can be used to determine the h5/group to save.
        data: the actual dataframe to save.
        """
        external_h5 = Path(tap._root).parent/"external.h5"
    
        computation_type, of_quantity = dataset
        _, group = tap.get_path(computation_type)
        dset = f"{group}/{of_quantity}"
    
        data.to_hdf(external_h5, key=dset)
        return (external_h5, dset)
    
    def read_external(tap, dataset):
        """Read the external dataset that has been previously registered."""
        external_h5 = Path(tap._root).parent/"external.h5"
    
        computation_type, of_quantity = dataset
        _, group = tap.get_path(computation_type)
        dset = f"{group}/{of_quantity}"
    
        return pd.read_hdf(external_h5, key=dset)
