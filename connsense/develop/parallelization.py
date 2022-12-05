

# #+RESULTS:

# For development we can use a develop version,


from collections.abc import Mapping
from copy import deepcopy
import shutil
from pathlib import Path
from lazy import lazy
from pprint import pformat

import json
import yaml

from multiprocessing import Process, Manager

import numpy as np
import pandas as pd

from connsense import extract_nodes,  plugins
from connsense.extract_connectivity import read_results
from connsense.extract_connectivity import extract as extract_connectivity
from connsense.pipeline import workspace
from connsense.pipeline import PARAMKEY, COMPKEYS
from connsense.io import logging, time, read_config as read_pipeline
from connsense.io.slurm import SlurmConfig
from connsense.io.write_results import read_toc_plus_payload, write_toc_plus_payload
from connsense.pipeline.workspace import find_base
from connsense.pipeline import ConfigurationError, NotConfiguredError
#from connsense.pipeline.store.store import HDFStore
#from connsense.develop.topotap import HDFStore
from connsense.define_subtargets.config import SubtargetsConfig
from connsense.analyze_connectivity import check_paths, matrices
from connsense.analyze_connectivity.analysis import SingleMethodAnalysisFromSource

# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, too-many-locals, comparison-with-callable, too-many-arguments, invalid-name, unspecified-encoding, unnecessary-lambda-assignment

LOG = logging.get_logger("connsense pipeline")


def _remove_link(path):
    try:
        return path.unlink()
    except FileNotFoundError:
        pass
    return None


EXECUTABLE = {"define": "loader", "extract": "extractor", "sample": "generator", "analyze": "computation"}

BATCH_SUBTARGETS = ("subtargets.h5", "batch")
COMPUTE_NODE_SUBTARGETS = ("inputs.h5", "subtargets")
INPUTS = ("inputs.h5", "subtargets")
COMPUTE_NODE_ASSIGNMENT = ("subtargets.h5", "compute_node")

INPROGRESS = "INPROGRESS"
DONE = "DONE"


class IllegalParallelComputationError(ValueError):
    """..."""


def describe(computation):
    """...Describe a `connsense-TAP computation`
    as the `connsense-TAP computation-type` to run, and the `quantity` that it computes.

    The parsed values will be used to look up parameters in the `connsense-TAP config.`
    """
    if isinstance(computation, str):
        description = computation.split('/')
        computation_type = description[0]
        quantity = '/'.join(description[1:])
    elif isinstance(computation, (tuple, list)):
        computation_type, quantity = computation
    else:
        raise TypeError(f"copmutation of illegal type {computation}")

    return (computation_type, quantity)


def parameterize(computation_type, of_quantity, in_config):
    """..."""
    """..."""
    paramkey = PARAMKEY[computation_type]

    if not computation_type in in_config["parameters"]:
        raise RuntimeError(f"Unknown {computation_type}")

    configured = in_config["parameters"][computation_type][paramkey]

    if of_quantity not in configured:
        try:
            multicomp, component = of_quantity.split('/')
        except ValueError:
            raise ConfigurationError(f"Unknown {paramkey} {of_quantity} for {computation_type}")
        configured_quantity =  configured[multicomp][component]

    else:
        configured_quantity = configured[of_quantity]

    return deepcopy(configured_quantity)

    if computation_type != "define-subtargets":
        if of_quantity not in in_config["parameters"][computation_type][paramkey]:
            raise RuntimeError(f"Unknown {paramkey[:-1]} {of_quantity} for {computation_type}")
        return deepcopy(in_config["parameters"][computation_type][paramkey][of_quantity])

    return deepcopy(in_config["parameters"]["define-subtargets"])



def batch_multinode(computation, of_inputs, in_config, at_dirpath, using_parallelization, single_submission=500,
                    with_weights=True, unit_weight=None):
    """...Just read the method definition above,
        and code below
    """
    from tqdm import tqdm; tqdm.pandas()

    n_compute_nodes, n_parallel_jobs = using_parallelization
    n_parallel_jobs_per_node_biggest_subtarget = int(n_parallel_jobs / n_compute_nodes)

    LOG.info("Assign compute-nodes to %s inputs", len(of_inputs))
    toc_index = of_inputs.index
    weights = of_inputs.progress_apply(lambda l: estimate_load(to_compute=None)(l())).dropna()
    weights = (weights[~np.isclose(weights, 0.)].groupby(toc_index.names).max()
               .sort_values(ascending=True)).rename("weight")

    unit_weight = max(unit_weight or 0.,  weights.max())

    compute_nodes = ((n_compute_nodes * (np.cumsum(weights) / weights.sum() - 1.e-6))
                     .astype(int).rename("compute_node"))
    LOG.info("Assign batches to compute node subtargets")

    def weigh_one(compute_node):
        return weights.loc[compute_node.index]

    def batch(compute_node):
        """..."""
        cn_weights = weigh_one(compute_node)
        n_parallel = min(int(n_parallel_jobs_per_node_biggest_subtarget * unit_weight / cn_weights.max()),
                         len(cn_weights))
        return pd.Series(np.linspace(0, n_parallel - 1.e-6, len(cn_weights), dtype=int), name="batch",
                         index=cn_weights.index)

    batches = ((pd.concat([batch(compute_nodes)], keys=[compute_nodes.unique()[0]], names=["compute_node"])
                if compute_nodes.nunique() == 1
                else pd.DataFrame(compute_nodes).groupby("compute_node").apply(batch))
               .reorder_levels(compute_nodes.index.names + ["compute_node"]))

    if not with_weights:
        return batches

    if not isinstance(weights.index, pd.MultiIndex):
        weights.index = pd.MultiIndex.from_arrays([weights.index.values], names=[weights.index.name])

    cn_weights = ((pd.concat([weigh_one(compute_nodes)], keys=[compute_nodes.unique()[0]], names=["compute_node"])
                   if compute_nodes.nunique() == 1
                   else pd.DataFrame(compute_nodes).groupby("compute_node").apply(weigh_one))
                  .reorder_levels(compute_nodes.index.names + ["compute_node"]))

    assignment =  pd.concat([batches, cn_weights], axis=1) if with_weights else batches

    if at_dirpath:
        assignment_h5, dataset = COMPUTE_NODE_ASSIGNMENT
        assignment.to_hdf(at_dirpath/assignment_h5, key=dataset)
    return assignment


def batch_parallel_groups(of_inputs, upto_number, to_compute=None, return_load=False):
    """..."""
    from tqdm import tqdm; tqdm.pandas()


    if isinstance(of_inputs, pd.Series):
        weights = (of_inputs.progress_apply(lambda l: estimate_load(to_compute)(l())).rename("load")
                   .sort_values(ascending=True))

    elif isinstance(of_inputs, pd.DataFrame):
        weights = (of_inputs.progress_apply(lambda l: estimate_load(to_compute)(l()), axis=1).rename("load")
                   .sort_values(ascending=True))

    else:
        raise TypeError(f"Unhandled type of input: {of_inputs}")

    nan_weights = weights[weights.isna()]
    if len(nan_weights) > 0:
        LOG.warning("No input data for %s / %s of_inputs:\n%s", len(nan_weights), len(weights),
                    pformat(nan_weights))
        weights = weights.dropna()

    computational_load = (np.cumsum(weights) / weights.sum()).rename("load")
    n = np.minimum(upto_number, len(weights))
    batches = (n * (computational_load - computational_load.min())).apply(int).rename("batch")

    LOG.info("Load balanced batches for %s of_inputs: \n %s", len(of_inputs), batches.value_counts())
    return batches if not return_load else pd.concat([batches, weights/weights.sum()], axis=1)


def estimate_load(to_compute):
    def of_input_data(d):
        """What would it take to compute input data d?
        """
        if d is None: return None

        try:
            shape = d.shape
        except AttributeError:
            pass
        else:
            return np.prod(shape)

        if callable(d): return of_input_data(d())

        if isinstance(d, Mapping):
            if not d: return 1.
            first = next(v for v in d.values())
            return of_input_data(first)

        if isinstance(d, pd.Series): return d.apply(of_input_data).sum()

        try:
            return len(d)
        except TypeError:
            pass

        return 1.

    return of_input_data


def distribute_compute_nodes(parallel_batches, upto_number):
    """..."""
    LOG.info("Assign compute nodes to batches \n%s", parallel_batches)
    _, dset = COMPUTE_NODE_ASSIGNMENT

    n_parallel_batches = parallel_batches.max() + 1
    compute_nodes = np.linspace(0, upto_number - 1.e-6, n_parallel_batches, dtype=int)
    assignment = pd.Series(compute_nodes[parallel_batches.values], name=dset, index=parallel_batches.index)
    return assignment


def read_compute_nodes_assignment(at_dirpath):
    """..."""
    assignment_h5, dataset = COMPUTE_NODE_ASSIGNMENT

    if not (at_dirpath/assignment_h5).exists():
        raise RuntimeError(f"No compute node assignment saved at {at_dirpath}")

    return pd.read_hdf(at_dirpath / assignment_h5, key=dataset)



def group_launchscripts(compute_nodes, max_entries):
    """..."""
    submit = lambda compute_node: int(compute_node / max_entries)
    return compute_nodes.apply(submit).rename("submission")


def read_index(of_computation, in_config):
    """..."""
    LOG.info("READ index of computation %s", of_computation)
    parameters = parameterize(*describe(of_computation), in_config)

    try:
        return parameters["index"]
    except KeyError as missing_index:
        LOG.info("No index configured for computation %s: \n%s", of_computation, missing_index)
        try:
            LOG.info("read index from the configured input.")
            return parameters["input"]
        except KeyError as missing_input:
            LOG.info("Neither an index, nor inputs were configured for computation %s", of_computation)
            raise NotConfiguredError("%s `input` was not configured %s") from missing_input
    raise RuntimeError("Python executtion must not reach here.")


def index_inputs(of_computation, in_tap):
    """..."""
    index_vars = read_index(of_computation, in_tap._config)

    if len(index_vars) > 1:
        return pd.MultiIndex.from_product([to_tap.subset_index(var, values) for var, values in index_vars.items()])

    var, values = next(iter(index_vars.items()))
    return pd.Index(to_tap.subset_index(var, values))


def slice_units(of_computation, in_tap):
    """..."""
    unit_computations = input_units(of_computation, in_tap)
    return [unit_computations[s:s+1] for s in range(0, len(unit_computations))]


def filter_datasets(described):
    """..."""
    return {var: val for var, val in described.items()
            if (var not in ("circuit", "connectome") and isinstance(val, Mapping)
                and ("dataset" in val or "datacall" in val))}


def lazily(to_evaluate):
    """..."""
    LOG.info("Evaluate %s lazily", to_evaluate.__name__)
    def evaluate_subtarget(s):
        return lambda: to_evaluate(s)
    return evaluate_subtarget


def load_dataset(tap, variable, values):
    """...Load a configured `computation-variable` from `connsense-TAP`
       values: as configured
    """
    def unpack_value(v):
        """..."""
        try:
            return v()
        except TypeError:
            pass

        try:
            get = v.get_value
        except AttributeError:
            return v
        return get()


    try:
        dset = values["dataset"]
    except KeyError:
        try:
            to_call = values["datacall"]
        except KeyError:
            raise ValueError("values need to define either dataset or datacall")
        dataset = brew_dataset(tap, to_call)
    else:
        LOG.info("Pour %s dataset: \n%s", variable, dset)
        dataset = tap.pour_dataset(*dset).apply(lazily(to_evaluate=unpack_value))

    if isinstance(dataset, pd.DataFrame):
        return pd.Series([r for _, r in dataset.iterrows()], index=dataset.index).apply(DataCall)

    return dataset if "reindex" not in values else reindex(tap, dataset, values["reindex"])


def bind_belazy(call):
    """..."""
    def get_value(belazy):
        """..."""
        def f(): raise TypeError("not a callable, i am")
        f.get_value = lambda : call(belazy.get_value())
        return f
    return get_value


def bind_lazy(call, **kwargs):
    """..."""
    def datacall(in_store):
        return DataCall(in_store, transform=(call, kwargs))
    return datacall


def brew_dataset(tap, call):
    """..."""
    in_store = pour(tap, call["input"]).apply(lazy_keyword).apply(DataCall)
    _, recipe = plugins.import_module(call["recipe"])
    return in_store.apply(bind_lazy(call=recipe, **call["kwargs"]))


def lazy_keyword(input_datasets):
    """...Repack a Mapping[String->CallData[D]] to CallData[Mapping[String->Data]]
    """
    def unpack(value):
        """..."""
        if callable(value):
            return value()

        try:
            get_value = value.get_value()
        except AttributeError:
            pass
        else:
            return get_value()

        return value

    return lambda: {var: unpack(value) for var, value in input_datasets.items()}



def pour(tap, datasets):
    """..."""
    LOG.info("Pour tap \n%s\n to get values for variables:\n%s", tap._root, pformat(datasets))
    dsets = sorted([(variable, load_dataset(tap, variable, values)) for variable, values in datasets.items()],
                   key=lambda x: len(x[1].index.names), reverse=True)

    common_index = dsets[0][1].index

    return (pd.concat([dset.reindex(common_index) for _, dset in dsets],
                      axis=1, keys=[name for name, _ in dsets])
            .apply(lambda row: row.to_dict(), axis=1))


def get_workspace(for_computation, in_config, in_mode=None):
    """..."""
    m = {'r': "test", 'w': "prod", 'a': "develop"}.get(in_mode, "test")
    computation_type, of_quantity = describe(for_computation)
    rundir = workspace.get_rundir(in_config, step=computation_type, substep=of_quantity, in_mode=m)
    basedir = workspace.find_base(rundir)
    return (basedir, rundir)

def configure_multinode(process, of_computation, in_config, at_dirpath):
    """..."""
    if process == setup_compute_node:
        return write_configs(of_computation, in_config, at_dirpath)
    if process == collect_results:
        return read_configs(of_computation, in_config, at_dirpath)
    raise ValueError(f"Unknown multinode {process}")


def write_configs(of_computation, in_config, at_dirpath):
    """..."""
    LOG.info("Write configs of %s at %s", of_computation, at_dirpath)
    return {"base": write_pipeline_base_configs(in_config, at_dirpath),
            "description": write_description(of_computation, in_config, at_dirpath)}


def read_configs(of_computation, in_config, at_dirpath):
    """..."""
    LOG.info("Read configs of %s at %s", of_computation, at_dirpath)
    return {"base": read_pipeline_base_configs(in_config, at_dirpath)}


def write_pipeline_base_configs(in_config, at_dirpath):
    """..."""
    basedir = find_base(rundir=at_dirpath)
    LOG.info("Check base configs at %s", basedir)

    def write_config(c):
        """..."""
        def write_format(f):
            filename = f"{c}.{f}"
            base_config = basedir / filename
            if base_config.exists():
                run_config = at_dirpath / filename
                _remove_link(run_config)
                run_config.symlink_to(base_config)
                return run_config
            LOG.info("Not found config %s", base_config)
            return None
        return {f: write_format(f) for f in ["json", "yaml"] if f}
    return {c: write_config(c) for c in ["pipeline", "runtime", "config", "parallel"]}


def read_pipeline_base_configs(in_config, at_dirpath):
    """..."""
    basedir = find_base(rundir=at_dirpath)

    def read_config(c):
        """..."""
        def read_format(f):
            """..."""
            filename = f"{c}.{f}"
            path_config = at_dirpath / filename
            if path_config.exists():
                LOG.warning("Pipeline config %s found at %s", filename, at_dirpath)

                if c in ("pipeline", "config"):
                    return read_pipeline.read(path_config)

                if c in ("runtime", "parallel"):
                    return read_runtime_config(path_config, of_pipeline=in_config)

                raise ValueError(f"NOT a connsense config: {filename}")

            LOG.warning("No pipeline config %s found at %s", filename, at_dirpath)
            return None

        return {f: read_format(f) for f in ["json", "yaml"] if f}

    return {c: read_config(c) for c in ["pipeline", "runtime", "config", "parallel"]}


def read_runtime_config(for_parallelization, *, of_pipeline=None, return_path=False):
    """..."""
    assert not of_pipeline or isinstance(of_pipeline, Mapping), of_pipeline

    if not for_parallelization:
        return (None, None) if return_path else None

    try:
        path = Path(for_parallelization)
    except TypeError:
        assert isinstance(for_parallelization, Mapping)
        path = None
        config = for_parallelization
    else:
        if path.suffix.lower() in (".yaml", ".yml"):
            with open(path, 'r') as fid:
                config = yaml.load(fid, Loader=yaml.FullLoader)
        elif path.suffix.lower() == ".json":
            with open(path, 'r') as fid:
                config = json.load(fid)
        else:
            raise ValueError(f"Unknown config type {for_parallelization}")

    if not of_pipeline:
        return (path, config) if return_path else config

    from_runtime = config["pipeline"]
    default_sbatch = lambda : deepcopy(config["slurm"]["sbatch"])

    def configure_slurm_for(computation_type):
        """..."""
        LOG.info("Configure slurm for %s", computation_type)
        try:
            cfg_computation_type = of_pipeline["parameters"][computation_type]
        except KeyError:
            return None

        paramkey = PARAMKEY[computation_type]
        try:
            quantities_to_configure = cfg_computation_type[paramkey]
        except KeyError:
            LOG.warning("No quantities to configure for %s". computation)
            return None

        try:
            runtime = from_runtime[computation_type]
        except KeyError:
            LOG.warning("No runtime configured for computation type %s", computation_type)
            return None

        configured = runtime[paramkey]

        def decompose_quantity(q):
            """..."""
            return [var for var in quantities_to_configure[q].keys() if var not in COMPKEYS]

        def configure_quantity(q):
            """..."""
            LOG.info("configure quantity %s", q)

            q_cfg = deepcopy(configured.get(q) or {})
            if "sbatch" not in q_cfg:
                q_cfg["sbatch"] = default_sbatch()
            if "number-compute-nodes" not in q_cfg:
                q_cfg["number-compute-nodes"] = 1
            if "number-tasks-per-node" not in q_cfg:
                q_cfg["number-tasks-per-node"] = 1

            def configure_component(c):
                """..."""
                cfg = deepcopy(configured.get(q, {}).get(c, {}))
                if "sbatch" not in cfg:
                    cfg["sbatch"] = q_cfg["sbatch"]
                if "number-compute-nodes" not in cfg:
                    cfg["number-compute-nodes"] = q_cfg["number-compute-nodes"]
                if "number-tasks-per-node" not in cfg:
                    cfg["number-tasks-per-node"] = q_cfg['number-tasks-per-node']

                return cfg

            LOG.info("decomposed quantity: \n%s", decompose_quantity(q))
            for c in decompose_quantity(q):
                q_cfg[c] = configure_component(c)

            return q_cfg

        return {q: configure_quantity(q) for q in quantities_to_configure if q != "description"}

    runtime_pipeline = {c: configure_slurm_for(computation_type=c) for c in of_pipeline["parameters"]}
    config = {"version": config["version"], "date": config["date"], "pipeline": runtime_pipeline}
    return (path, config) if return_path else config



def prepare_parallelization(of_computation, in_config, using_runtime):
    """..."""
    computation_type, quantity = describe(of_computation)
    from_runtime = (read_runtime_config(for_parallelization=using_runtime, of_pipeline=in_config)
                    if not isinstance(using_runtime, Mapping) else using_runtime)
    LOG.info("Prepare parallelization %s using runtime \n%s", of_computation, pformat(from_runtime))
    configured = from_runtime["pipeline"].get(computation_type, {})
    LOG.info("\t Configure \n%s", pformat(configured))
    return read_njobs(to_parallelize=configured, computation_of=quantity)


def read_njobs(to_parallelize, computation_of):
    """..."""
    if not to_parallelize:
        return (1, 1)

    try:
        q = computation_of.name
    except AttributeError:
        q = computation_of

    try:
        p = to_parallelize[q]
    except KeyError:
        if '/' in q:
            try:
                q0, q1 = q.split('/')
            except ValueError: #TODO: log something
                return (1, 1)
            else:
                try:
                    p0 = to_parallelize[q0]
                except KeyError:
                    return (1, 1)
                else:
                    try:
                        p = p0[q1]
                    except KeyError:
                        return (1, 1)
                    else:
                        pass
        else:
            return (1, 1)

    compute_nodes = p["number-compute-nodes"]
    tasks = p["number-tasks-per-node"]
    return (compute_nodes, compute_nodes * tasks)


def write_description(of_computation, in_config, at_dirpath):
    """..."""
    computation_type, of_quantity = describe(of_computation)
    configured = parameterize(computation_type, of_quantity, in_config)
    configured["name"] = of_quantity
    return read_pipeline.write(configured, to_json=at_dirpath/"description.json")


def reindex(tap, inputs, variables):
    """..."""
    connsense_ids = {v: tap.index_variable(value) for v, value in variables.items()}
    def index_ids(subtarget):
        return pd.MultiIndex.from_frame(pd.DataFrame({f"{var}_id": connsense_ids[var].loc[values.values].values
                                                      for var, values in subtarget.index.to_frame().items()}))
    reinputs = inputs.apply(lambda s: pd.DataFrame(s()).set_index(index_ids(subtarget=s())))
    frame = pd.concat(reinputs.values, keys=reinputs.index)
    groups_of_inner_frames = list(frame.groupby(frame.index.names))
    return (pd.Series([d.reset_index(drop=True) for _, d in groups_of_inner_frames],
                      index=pd.MultiIndex.from_tuples([i for i,_ in groups_of_inner_frames],
                                                      names=frame.index.names))
            .apply(DataCall))




class DataCall:
    """Call data..."""
    def __init__(self, dataitem, transform=None, cache=False):
        self._dataitem = dataitem
        self._transform = transform
        self._cache = None if not cache else {}

    @lazy
    def dataset(self):
        """..."""
        return self()

    @lazy
    def shape(self):
        """..."""
        original = self._dataitem() if callable(self._dataitem) else self._dataitem
        if self._cache is not None:
            self._cache["original"] = original

        if isinstance(original, Mapping):
            return next(v for v in original.values()).shape
        try:
            return original.shape
        except AttributeError:
            pass

        try:
            return len(original)
        except TypeError:
            pass

        return 1.

    def __call__(self):
        """Call Me."""
        if self._cache is not None and "original" in self._cache:
            original = self._cache["original"]
        else:
            try:
                original = self._dataitem()
            except TypeError:
                try:
                    original = self._dataitem.get_value()
                except AttributeError:
                    original = self._dataitem

            if self._cache is not None:
                self._cache["original"] = original

        if not self._transform:
            return original

        if callable(self._transform):
            return self._transform(original)

        transform, kwargs = self._transform
        return transform(**original, **kwargs)


def generate_inputs(of_computation, in_config, slicing=None, circuit_args=None):
    """..."""
    from connsense.develop.topotap import HDFStore
    LOG.info("Generate inputs for %s.", of_computation)

    computation_type, of_quantity = describe(of_computation)
    params = parameterize(computation_type, of_quantity, in_config)
    tap = HDFStore(in_config)

    datasets = pour(tap=HDFStore(in_config), datasets=filter_datasets(params["input"]))
    original = datasets.apply(lazy_keyword).apply(DataCall)

    controlled = control_inputs(of_computation, in_config, using_tap=tap)
    if controlled is not None:
        original = pd.concat([original], axis=0, keys=["original"], names=["control"])
        full = pd.concat([original, controlled])
        full = full.reorder_levels([l for l in full.index.names if l != "control"] + ["control"])
    else:
        full = original

    def index_circuit_args(inputs):
        """..."""
        if not circuit_args:
            return inputs
        missing = {f"{var}_id": tap.index_variable(var, value) for var, value in circuit_args.items()
                   if f"{var}_id" not in inputs.index.names}
        for variable_id, value in missing.items():
            inputs = (pd.concat([inputs], axis=0, keys=[value], names=[variable_id])
                      .reorder_levels(inputs.index.names + [variable_id]))
        return inputs

    if not slicing or slicing == "full":
        return index_circuit_args(full)

    assert slicing in params["slicing"]
    cfg = {slicing: params["slicing"][slicing]}
    return index_circuit_args(generate_slices(tap, inputs=full,
                                              using_knives=load_slicing(cfg, tap, lazily=False)[slicing]))

def generate_slices(of_tap, inputs, using_knives):
    """Generate slices of inputs accoring to configured knives."""
    from tqdm import tqdm; tqdm.pandas()
    slices = pd.concat([inputs.apply(datacall(cut)) for cut in using_knives], axis=0, keys=using_knives.index)
    return slices.reorder_levels(inputs.index.names +
                                 slices.index.names[0:(len(slices.index.names) - len(inputs.index.names))])



def pour_datasets(from_tap, for_inputs, and_additionally=None):
    """..."""
    LOG.info("Get input data from tap: \n%s", for_inputs)
    references = deepcopy(for_inputs)
    if and_additionally:
        LOG.info("And additionally: \n%s", and_additionally)
        references.update({key: {"dataset": ref} for key, ref in and_additionally.items()} or {})
    datasets = pour(from_tap, datasets=references)
    return datasets.apply(lazy_keyword).apply(DataCall)


def datacall(transformation):
    """Apply a transformation, lazily."""
    def transform(dataitem):
        """..."""
        return DataCall(dataitem, transformation)
    return transform


def control_inputs(of_computation, in_config, using_tap):
    """..."""
    params = parameterize(*describe(of_computation), in_config)
    try:
        randomizations = params["controls"]
    except KeyError:
        LOG.warning("It seems no controls have been configured for the input of %s: \n%s", of_computation, params)
        return None

    controls = load_control(randomizations, lazily=False)
    assert controls, "Cannot be empty. Check your config."

    for_input = filter_datasets(params["input"])
    return pd.concat([pour_datasets(using_tap, for_input, and_additionally=to_tap).apply(datacall(shuffle))
                      for _, shuffle, to_tap in controls], axis=0,
                     keys=[control_label for control_label, _, _ in controls], names=["control"])


def slice_inputs(of_computation, in_config, datasets=None, using_tap=None):
    """..."""
    params = parameterize(*describe(of_computation), in_config)
    try:
        configured = params["slicing"]
    except KeyError:
        LOG.warning("It seems no slicing have been configured for the input of %s:\n%s", of_computation, params)
        return (None, None)

    knives = {k: v for k, v in configured.items() if k not in ("description", "do-full")}

    slicing = load_slicing(knives, lazily=False, using_tap=using_tap)
    assert slicing, "Cannot be empty. Check your config."

    for_input = filter_datasets(params["input"])
    if datasets is None:
        assert using_tap
        datasets = pour_datasets(using_tap, for_input)
    return pd.concat([datasets.apply(datacall(cut)) for _, cut in slicing],
                     axis=0, keys=[knife_label for knife_label, _ in slicing], names=["slice"]),


def configure_slurm(computation, in_config, using_runtime):
    """..."""
    computation_type, quantity = describe(computation)
    pipeline_config = in_config if isinstance(in_config, Mapping) else read_pipeline.read(in_config)
    from_runtime = (read_runtime_config(for_parallelization=using_runtime, of_pipeline=pipeline_config)
                    if not isinstance(using_runtime, Mapping) else using_runtime)

    params = from_runtime["pipeline"].get(computation_type, {})
    try:
        configured = params[quantity]
    except KeyError:
        quantity, component = quantity.split('/')
        configured = params[quantity][component]
    return configured


def setup_compute_node(c, of_computation, with_inputs, using_configs, at_dirpath, in_mode=None,
                       slicing=None):
    """..."""
    assert not in_mode or in_mode in ("prod", "develop")

    from connsense.apps import APPS
    LOG.info("Configure chunk %s with %s inputs to compute %s, using configs \n%s",
             c, len(with_inputs), of_computation, using_configs)

    computation_type, of_quantity = describe(of_computation)
    for_compute_node = at_dirpath / f"compute-node-{c}"
    for_compute_node.mkdir(parents=False, exist_ok=True)
    configs = symlink_pipeline(using_configs, at_dirpath=for_compute_node)

    inputs_to_read = write_compute(batches=with_inputs, to_hdf=INPUTS, at_dirpath=for_compute_node)
    output_h5 = f"{for_compute_node}/connsense.h5"

    try:
        slurm_params = using_configs["slurm_params"]
    except KeyError as kerr:
            raise RuntimeError("Missing slurm params") from kerr
    of_executable = cmd_sbatch(APPS["main"], of_computation, config=slurm_params, at_dirpath=for_compute_node)

    if "submission" not in with_inputs:
        launchscript = at_dirpath / "launchscript.sh"
    else:
        submission = with_inputs.submission.unique()
        assert len(submission) == 1
        launchscript = at_dirpath / f"launchscript-{submission[0]}.sh"


    run_mode = in_mode or "prod"
    command_lines = ["#!/bin/bash",
                     (f"########################## LAUNCH {computation_type} for chunk {c}"
                      f" of {len(with_inputs)} _inputs. #######################################"),
                     f"pushd {for_compute_node}",
                     f"sbatch {of_executable.name} run {computation_type} {of_quantity} \\",
                     "--configure=pipeline.yaml --parallelize=runtime.yaml \\",
                     None if not slicing else f"--slicing={slicing} \\",
                     f"--mode={run_mode} \\",
                     f"--input={inputs_to_read} \\",
                     f"--output={output_h5}",
                     "popd"]

    with open(launchscript, 'a') as to_launch:
        to_launch.write('\n'.join(l for l in command_lines if l) + "\n")

    setup = {"dirpath": for_compute_node, "sbatch": of_executable, "input": inputs_to_read, "output": output_h5}

    return read_pipeline.write(setup, to_json=for_compute_node/"setup.json")


def cmd_sbatch(executable, of_computation, config, at_dirpath):
    """..."""
    computation_type, _ = describe(of_computation)
    slurm_params = deepcopy(config)
    slurm_params.update({"name": computation_type, "executable": executable})
    slurm_config = SlurmConfig(slurm_params)
    return slurm_config.save(to_filepath=at_dirpath/f"{computation_type}.sbatch")


def write_compute(batches, to_hdf, at_dirpath):
    """..."""
    batches_h5, and_hdf_group = to_hdf
    batches.to_hdf(at_dirpath / batches_h5, key=and_hdf_group, format="fixed", mode='w')
    return at_dirpath / batches_h5


def write_multinode_setup(compute_nodes, inputs, at_dirpath):
    """..."""
    inputs_h5, dataset = INPUTS
    return read_pipeline.write({"compute_nodes": compute_nodes, "inputs": at_dirpath / inputs_h5},
                               to_json=at_dirpath/"setup.json")


def read_setup_compute_node(c, for_quantity):
    """..."""
    for_compute_node = for_quantity / f"compute-node-{c}"

    if not for_compute_node.exists():
        raise RuntimeError(f"Expected compute node directory {for_compute_node} created by the TAP run to collect")

    return read_setup(at_dirpath=for_quantity, compute_node=c)


def read_setup(at_dirpath, compute_node):
    """..."""
    setup_json = at_dirpath / f"compute-node-{compute_node}" / "setup.json"

    if not setup_json.exists():
        raise RuntimeError(f"No setup json found at {setup_json}")

    with open(setup_json, 'r') as f:
        return json.load(f)

    raise RuntimeError("Python execution must not have reached here.")


def symlink_pipeline(configs, at_dirpath):
    """..."""
    to_base = symlink_pipeline_base(configs["base"], at_dirpath)
    return {"base": to_base}


def create_symlink(at_dirpath):
    """..."""
    def _to(config_at_path):
        """..."""
        it_is_a = at_dirpath / config_at_path.name
        _remove_link(it_is_a)
        it_is_a.symlink_to(config_at_path)
        return it_is_a

    return _to


def symlink_pipeline_base(configs, at_dirpath):
    """..."""
    symlink_to = create_symlink(at_dirpath)
    return {"pipeline": {fmt: symlink_to(config_at_path=p) for fmt, p in configs["pipeline"].items() if p},
            "runtime": {fmt: symlink_to(config_at_path=p) for fmt, p in configs["runtime"].items() if p}}



def setup_multinode(process, of_computation, in_config, using_runtime, in_mode=None):
    """Setup a multinode process.
    """
    from tqdm import tqdm; tqdm.pandas()
    from connsense.develop.topotap import HDFStore

    n_compute_nodes, n_parallel_jobs = prepare_parallelization(of_computation, in_config, using_runtime)

    def prepare_compute_nodes(inputs, at_dirpath, slicing, unit_weight=None):
        """..."""
        at_dirpath.mkdir(exist_ok=True, parents=False)
        using_configs = configure_multinode(process, of_computation, in_config, at_dirpath)

        if process == setup_compute_node:
            batched = batch_multinode(of_computation, inputs, in_config, at_dirpath, unit_weight=unit_weight,
                                      using_parallelization=(n_compute_nodes, n_parallel_jobs))
            using_configs["slurm_params"] = (configure_slurm(of_computation, in_config, using_runtime)
                                             .get("sbatch", None))
            compute_nodes = {c: setup_compute_node(c, of_computation, inputs, using_configs, at_dirpath,
                                                   in_mode=in_mode, slicing=slicing)
                             for c, inputs in batched.groupby("compute_node")}
            return {"configs": using_configs,
                    "number_compute_nodes": n_compute_nodes, "number_total_jobs": n_parallel_jobs,
                    "setup": write_multinode_setup(compute_nodes, inputs, at_dirpath)}

        if process == collect_results:
            batched = read_compute_nodes_assignment(at_dirpath)
            _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
            h5_group = output_paths["steps"][computation_type]

            setup = {c: read_setup_compute_node(c, for_quantity=at_dirpath)
                     for c, _ in batched.groupby("compute_node")}
            return collect_results(computation_type, setup, at_dirpath, in_connsense_store=h5_group,
                                   slicing=slicing)

        return ValueError(f"Unknown multinode {process}")

    _, to_stage = get_workspace(of_computation, in_config)

    using_configs = configure_multinode(process, of_computation, in_config, at_dirpath=to_stage)

    computation_type, of_quantity = describe(of_computation)
    params = parameterize(*describe(of_computation), in_config)

    full = generate_inputs(of_computation, in_config,
                           circuit_args=input_circuit_args(of_computation, in_config, load_circuit=True))
    if process == setup_compute_node:
        full_weights = full.progress_apply(lambda l: estimate_load(to_compute=None)(l())).dropna()

        have_zero_weight = np.isclose(full_weights.values, 0.)
        LOG.info("Inputs with zero weight %s: \n%s", have_zero_weight.sum(), full_weights[have_zero_weight])

        full = full[~have_zero_weight]
        full_weights = full_weights[~have_zero_weight]
        max_weight = full_weights.max()
    else:
        max_weight = None

    compute_nodes = {"full": prepare_compute_nodes(full, at_dirpath=(to_stage/"full" if "slicing" in params
                                                                     else to_stage),
                                                   slicing=("full" if "slicing" in params else None),
                                                   unit_weight=max_weight)}

    if "slicing" not in params:
        return compute_nodes

    slicings = load_slicing({k:v for k,v in params["slicing"].items() if k not in ("description", "do-full")},
                            lazily=False)
    of_tap = HDFStore(in_config)
    for slicing, to_slice in slicings.items():
        sliced_inputs = generate_slices(of_tap, inputs=full, using_knives=to_slice)
        compute_nodes[slicing] = prepare_compute_nodes(sliced_inputs, at_dirpath=to_stage/slicing, slicing=slicing,
                                                       unit_weight=None)

    return compute_nodes


def collect_results(computation_type, setup, from_dirpath, in_connsense_store, slicing):
    """..."""
    if computation_type == "extract-node-populations":
        assert not slicing, "Does not apply"
        return collect_node_population(setup, from_dirpath, in_connsense_store)

    if computation_type == "extract-edge-populations":
        assert not slicing, "Does not apply"
        return collect_edge_population(setup, from_dirpath, in_connsense_store)

    if computation_type in ("analyze-connectivity", "analyze-composition",
                            "analyze-node-types", "analyze-physiology"):
        return collect_analyze_step(setup, from_dirpath, in_connsense_store, slicing)

    raise NotImplementedError(f"INPROGRESS: {computation_type}")


def collect_node_population(setup, from_dirpath, in_connsense_store):
    """..."""
    try:
        with open(from_dirpath/"description.json", 'r') as f:
            population = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the population extracted: {from_dirpath}")

    connsense_h5, group = in_connsense_store
    hdf_population = group + '/' + population["name"]

    def describe_output(of_compute_node):
        """..."""
        try:
            with open(Path(of_compute_node["dirpath"]) / "output.json", 'r') as f:
                output = json.load(f)
        except FileNotFoundError as ferr:
            raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
        return output

    outputs = {c: describe_output(of_compute_node) for c, of_compute_node in setup.items()}
    LOG.info("Extract node populations %s reported outputs: \n%s", population["name"], pformat(outputs))

    def in_store(at_path, hdf_group=None):
        """..."""
        return matrices.get_store(at_path, hdf_group or hdf_population, pd.DataFrame)

    def move(compute_node, output):
        """..."""
        LOG.info("Get node population store for compute-node %s output %s", compute_node, output)
        h5, g = output
        return in_store(at_path=h5, hdf_group=g)

    return in_store(connsense_h5).collect({c: move(compute_node=c, output=o) for c, o in outputs.items()})


def collect_edge_population(setup, from_dirpath, in_connsense_store):
    """..."""
    LOG.info("Collect edge population at %s using setup \n%s", from_dirpath, setup)

    try:
        with open(from_dirpath/"description.json", 'r') as f:
            population = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the population extracted: {at_basedir}") from ferr

    #p = population["name"]
    #hdf_edge_population = f"edges/populations/{p}"
    connsense_h5, group = in_connsense_store
    hdf_edge_population = group + '/' + population["name"]

    LOG.info("Collect edges with description \n%s", pformat(population))

    def describe_output(of_compute_node):
        """..."""
        try:
            with open(Path(of_compute_node["dirpath"]) / "output.json", 'r') as f:
                output = json.load(f)
        except FileNotFoundError as ferr:
            raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
        return output

    outputs = {c: describe_output(of_compute_node) for c, of_compute_node in setup.items()}
    LOG.info("Edge extraction reported outputs: \n%s", pformat(outputs))

    def collect_adjacencies(of_compute_node, output):
        """..."""
        LOG.info("Collect adjacencies compute-node %s output %s", of_compute_node, output)
        adj = read_toc_plus_payload(output, for_step="extract-edge-populations")
        return write_toc_plus_payload(adj, (connsense_h5, hdf_edge_population), append=True, format="table",
                                      min_itemsize={"values": 100})
        #return write_toc_plus_payload(adj, (in_connsense_store, hdf_edge_population), append=True, format="table")

    LOG.info("Collect adjacencies")
    for of_compute_node, output in outputs.items():
        collect_adjacencies(of_compute_node, output)

    LOG.info("Adjacencies collected: \n%s", len(outputs))

    return (in_connsense_store, hdf_edge_population)


def collect_analyze_step(setup, from_dirpath, in_connsense_store, slicing):
    """..."""
    try:
        with open(from_dirpath/"description.json", 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the analysis extracted: {from_dirpath}") from ferr

    #a = analysis["name"]
    #hdf_analysis = f"analyses/connectivity/{a}"
    connsense_h5, group = in_connsense_store
    hdf_group = group + '/' + analysis["name"]
    if slicing:
        hdf_group = hdf_group + "/" + slicing
    output_type = analysis["output"]

    def describe_output(of_compute_node):
        """..."""
        try:
            with open(Path(of_compute_node["dirpath"]) / "output.json", 'r') as f:
                output = json.load(f)
        except FileNotFoundError as ferr:
            raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
        return output

    outputs = {c: describe_output(of_compute_node) for c, of_compute_node in setup.items()}
    LOG.info("Analysis %s reported outputs: \n%s", analysis["name"], pformat(outputs))

    def in_store(at_path, hdf_group):
        """..."""
        return matrices.get_store(at_path, hdf_group or hdf_analysis, output_type)

    def move(compute_node, output):
        """..."""
        LOG.info("Get analysis store for compute-node %s output %s", compute_node, output)
        h5, g = output
        return in_store(at_path=h5, hdf_group=g)

    return (in_store(connsense_h5, hdf_group)
            .collect({c: move(compute_node=c, output=o) for c, o in outputs.items()}))



SERIAL_BATCHES = "serial-batches"
PARALLEL_BATCHES = "parallel-batches"

def run_multiprocess(of_computation, in_config, using_runtime, on_compute_node,
                     batching=SERIAL_BATCHES, slicing=None):
    """..."""
    from connsense.develop.topotap import HDFStore
    on_compute_node = run_cleanup(on_compute_node)

    run_in_progress = on_compute_node.joinpath(INPROGRESS)
    run_in_progress.touch(exist_ok=False)

    execute, to_store_batch, to_store_one = configure_execution(of_computation, in_config, on_compute_node,
                                                                slicing=slicing)

    assert to_store_batch or to_store_one
    assert not (to_store_batch and to_store_one)

    computation_type, of_quantity = describe(of_computation)
    parameters = parameterize(computation_type, of_quantity, in_config)

    in_hdf = "connsense-{}.h5"

    circuit_kwargs = input_circuit_args(of_computation, in_config, load_circuit=True)
    circuit_args = tuple(k for k in ["circuit", "connectome"] if circuit_kwargs[k])
    circuit_args_values = tuple(v for v in (circuit_kwargs["circuit"], circuit_kwargs["connectome"]) if v)
    circuit_args_names = tuple(v for v in ((lambda c: c.variant if c else None)(circuit_kwargs["circuit"]),
                                           circuit_kwargs["connectome"]) if v)

    kwargs = load_kwargs(parameters, HDFStore(in_config), on_compute_node)

    inputs = generate_inputs(of_computation, in_config, slicing, circuit_kwargs)

    collector = plugins.import_module(parameters["collector"]) if "collector" in parameters else None

    def collect_batch(results):
        """..."""
        if not collector:
            return results

        _, collect = collector
        return collect(results)

    def execute_one(lazy_subtarget, bowl=None, index=None):
        """..."""
        result = execute(*circuit_args_values, **lazy_subtarget(), **kwargs)
        if bowl:
            assert index
            bowl[index] = result
        return result

    def lazy_dataset(s):
        """..."""
        if callable(s): return s

        if isinstance(s, Mapping): return lambda: {var: value() for var, value in s.items()}

        raise ValueError(f"Cannot resolve lazy dataset of type {type(s)}")

    def serial_batch(of_input, *, index, in_bowl=None):
        """..."""
        LOG.info("Run %s batch %s of %s inputs args, and circuit %s, \n with kwargs %s slicing %s", of_computation,
                 index, len(of_input), circuit_args_values, pformat(kwargs), slicing)

        def to_subtarget(s):
            """..."""
            r = execute_one(lazy_dataset(s))
            LOG.info("store one lazy subtarget %s result \n%s", s, r)
            LOG.info("Result data types %s", r.describe())
            return to_store_one(in_hdf.format(index), result=r)

        if to_store_batch:
            results = of_input.apply(execute_one)
            try:
                results = results.droplevel("compute_node")
            except KeyError:
                pass
            result = to_store_batch(in_hdf.format(index), results=collect_batch(results))
        else:
            toc = of_input.apply(to_subtarget)
            try:
                toc = toc.droplevel("compute_node")
            except KeyError:
                pass
            result = to_store_one(in_hdf.format(index), update=toc)

        if in_bowl is not None:
            in_bowl[index] = result
        return result

    n_compute_nodes,  n_total_jobs = prepare_parallelization(of_computation, in_config, using_runtime)

    batches = load_input_batches(on_compute_node)
    n_batches = batches.batch.max() - batches.batch.min() + 1

    if n_compute_nodes == n_total_jobs:
        results = {}
        for batch, subtargets in batches.groupby("batch"):
            LOG.info("Run Single Node %s process %s / %s batches", on_compute_node, batch, n_batches)
            results[batch] = serial_batch(inputs.loc[subtargets.index], index=batch)
        LOG.info("DONE Single Node connsense run.")
    else:
        if batching == SERIAL_BATCHES:
            manager = Manager()
            bowl = manager.dict()
            processes = []

            for batch, subtargets in batches.groupby("batch"):
                LOG.info("Spawn Compute Node %s process %s / %s batches", on_compute_node, batch, n_batches)
                p = Process(target=serial_batch,
                            args=(inputs.loc[subtargets.index],), kwargs={"index": batch, "in_bowl": bowl})
                p.start()
                processes.append(p)

            LOG.info("LAUNCHED %s processes", n_batches)

            for p in processes:
                p.join()

            LOG.info("Parallel computation %s results %s", of_computation, len(bowl))

            results = {key: value for key, value in bowl.items()}
            LOG.info("Computation %s results %s", of_computation, len(results))

        else:
            assert batching == PARALLEL_BATCHES, "No other is known."
            for batch, subtargets in batches.groupby("batch"):
                LOG.info("Run %s subtargets in parallel batch %s / %s batches.", len(subtargets), batch, len(batches))

                manager = Manager()
                bowl = manager.dict()
                processes = []

                for i, s in enumerate(subtargets):
                    p = Process(target=execute_one, args=(s,), kwargs={"index": i, "bowl": bowl})
                    p.start()
                    process.append(p)
                LOG.info("LAUNCHED %s process", i)

                for p in process:
                    p.join()
                LOG.info("Parallel computation for batch %s: %s", batch, len(bowl))
                values = pd.Series([v for v in bowl.values()], index=subtargets.index)
                hdf = in_hdf.format(batch)
                result = (to_store_batch(hdf, results=values) if to_store_batch else
                          to_store_one(hdf, update=values.apply(lambda v: to_store_one(hdf, result=v))))

    read_pipeline.write(results, to_json=on_compute_node/"batched_output.json")

    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, hdf_group = output_paths["steps"][computation_type]
    of_output_type = parameters["output"]

    collected = collect_batches(of_computation, results, on_compute_node, hdf_group, slicing, of_output_type)
    read_pipeline.write(collected, to_json=on_compute_node/"output.json")

    run_in_progress.unlink()
    on_compute_node.joinpath(DONE).touch(exist_ok=False)

    return collected


def input_circuit(labeled, in_config):
    """..."""
    if not labeled:
        return None
    sbtcfg = SubtargetsConfig(in_config)
    circuit = sbtcfg.attribute_depths(circuit=labeled)

    return circuit


def input_connectome(labeled, in_circuit):
    """..."""
    if not labeled:
        return None

    from bluepy import Circuit
    assert isinstance(in_circuit, Circuit)

    if labeled == "local":
        return in_circuit.connectome
    return in_circuit.projection[labeled]


def input_circuit_args(computation, in_config, load_circuit=True, load_connectome=False):
    """..."""
    computation_type, of_quantity = describe(computation)
    parameters = parameterize(computation_type, of_quantity, in_config)

    try:
        computation_inputs = parameters["input"]
    except KeyError as kerr:
        raise ValueError(f"No inputs configured for {computation}") from kerr

    input_circuits = computation_inputs.get("circuit", None)
    if input_circuits:
        assert len(input_circuits) == 1, f"NotImplemented processing more than one circuit"
        c = input_circuits[0]
    else:
        c = None
    circuit = input_circuit(c, in_config) if load_circuit else c

    input_connectomes = computation_inputs.get("connectome", None)
    if input_connectomes:
        assert len(input_connectomes) == 1, f"NotImplemented processing more than one connectome"
        x = input_connectomes[0]
    else:
        x = None
    connectome = input_connectome(x, in_circuit) if load_connectome else x
    return {"circuit": circuit, "connectome": connectome}


def subtarget_circuit_args(computation, in_config, load_circuit=False, load_connectome=False):
    """..."""
    computation_type, of_quantity = describe(computation)
    parameters = parameterize(computation_type, of_quantity, in_config)

    try:
        subtarget = parameters["subtarget"]
    except KeyError as kerr:
        LOG.warning("No subtargets specified for %s", computation)
        return input_circuit_args(computation, in_config, load_circuit, load_connectome)

    c = subtarget.get("circuit", None)
    circuit = input_circuit(c, in_config) if load_circuit else c

    x = subtarget.get("connectome", None)
    return {"circuit": circuit, "connectome": input_connectome(x, circuit) if load_connectome else x}


def load_input_batches(on_compute_node, inputs=None, n_parallel_tasks=None):
    """..."""
    store_h5, dataset = COMPUTE_NODE_SUBTARGETS

    assert inputs is None or inputs == on_compute_node / store_h5, (
        "inputs dont seem to be what was configured\n"
        f"Expected {inputs} to be {on_compute_node / store_h5} if setup by run_multinode(...)")

    inputs_read = pd.read_hdf(on_compute_node/store_h5, key=dataset)
    if not n_parallel_tasks:
        return inputs_read
    return inputs_read.assign(batch=pd.Series(np.arange(0, len(inputs_read))%n_parallel_tasks).to_numpy(int))




def load_kwargs(parameters, to_tap, on_compute_node, consider_input=False):
    """..."""
    def load_dataset(value):
        """..."""
        return (to_tap.read_dataset(value["dataset"]) if isinstance(value, Mapping) and "dataset" in value
                else value)

    kwargs = parameters.get("kwargs", {})
    kwargs.update({var: load_dataset(value) for var, value in kwargs.items() if var not in COMPKEYS})

    if consider_input:
        kwargs.update({var: value for var, value in parameters.get("input", {}).items()
                       if var not in ("circuit", "connectome") and (
                               not isinstance(value, Mapping) or "dataset" not in value)})

    try:
        workdir = kwargs["workdir"]
    except KeyError:
        return kwargs

    if isinstance(workdir, Path):
        return kwargs

    if isinstance(workdir, str):
        path = Path(workdir)/on_compute_node.relative_to(to_tap._root.parent)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            LOG.warning("Compute node has been run before, leaving a workdir:\n%s", path)
            archive = path.parent / "history"
            archive.mkdir(parents=False, exist_ok=True)

            history = archive / path.name
            if history.exists():
                LOG.warning("Previous runs exist in %s history at \n%s", path.name, history)
            history.mkdir(parents=False, exist_ok=True)

            to_archive = history/ time.stamp(now=True)
            if to_archive.exists():
                LOG.warning("There is a previous run with the same time stamp as now!!!\n%s"
                            "\n It will be removed", to_archive)
                shutil.copytree(path, to_archive,
                                symlinks=True, ignore_dangling_symlinks=True, dirs_exist_ok=True)
                for filepath in path.glob('*'):
                    filepath.unlink()
            else:
                path.rename(to_archive)

        path.mkdir(parents=False, exist_ok=True)

        try:
            on_compute_node.joinpath("workdir").symlink_to(path)
        except FileExistsError as ferr:
            LOG.warn("Symlink to workdir compute-node %s already exists, most probably from a previous run"
                        " Please cleanup before re-run", str(on_compute_node))

        kwargs["workdir"] = path
        return kwargs

    if workdir is True:
        workdir = on_compute_node / "workdir"
        workdir.mkdir(parents=False, exist_ok=True)
        kwargs["workdir"] = workdir
        return kwargs

    raise NotImplementedError(f"What to do with workdir type {type(workdir)}")


def get_executable(computation_type, parameters):
    """..."""
    executable_type = EXECUTABLE[computation_type.split('-')[0]]

    try:
        executable = parameters[executable_type]
    except KeyError as err:
        raise RuntimeError(f"No {executable_type} defined for {computation_type}") from err

    _, execute = plugins.import_module(executable["source"], executable["method"])

    return execute


def store_node_properties_batch(of_population, on_compute_node, in_hdf_group):
    """...This will extract node properties for all subtargets as a single datasframe.
    NOT-IDEAL and needs hacks to gather differemt resuts into the same input dataframe.
    REPLACE by single subtarget store using matrices
    """
    def write_batch(connsense_h5, results):
        """..."""
        in_hdf = (on_compute_node/connsense_h5, in_hdf_group)
        LOG.info("Write %s  results %s ", in_hdf, len(results))
        return extract_nodes.write(results, of_population, in_hdf)

    return write_batch


def store_node_properties(of_population, on_compute_node, in_hdf_group):
    """..."""
    LOG.info("Store node properties of population %s on compute node %s in hdf %s, one subtarget at a time",
             of_population, on_compute_node, in_hdf_group)

    def write_hdf(at_path, *, result=None, update=None):
        """..."""
        assert not(result is None and update is None)
        assert result is not None or update is not None

        hdf_population = in_hdf_group+'/'+of_population
        store = matrices.get_store(on_compute_node/at_path, hdf_population, "pandas.DataFrame")

        if result is not None:
            return store.write(result)

        store.append_toc(store.prepare_toc(of_paths=update))
        return (at_path, hdf_population)

    return write_hdf


def store_edge_extraction(of_population, on_compute_node, in_hdf_group):
    """..."""
    def write_batch(connsense_h5, results):
        """..."""
        in_hdf = (on_compute_node/connsense_h5, f"{in_hdf_group}/{of_population}")
        LOG.info("Write %s batch results to %s", len(results), in_hdf)
        return extract_connectivity.write_adj(results, to_output=in_hdf,  append=True, format="table",
                                              return_config=True)

    return write_batch


def store_matrix_data(of_quantity, parameters, on_compute_node, in_hdf_group, slicing):
    """..."""
    LOG.info("Store matrix data for %s", parameters)
    of_output = parameters["output"]
    hdf_group = f"{in_hdf_group}/{of_quantity}"
    if slicing:
        hdf_group = f"{hdf_group}/{slicing}"

    cached_stores = {}

    def write_hdf(at_path, *, result=None, update=None):
        """..."""
        assert at_path
        assert not(result is None and update is None)
        assert result is not None or update is not None

        p = on_compute_node/at_path
        if p not in cached_stores:
            cached_stores[p] = matrices.get_store(p, hdf_group, for_matrix_type=of_output)

        if result is not None:
            return cached_stores[p].write(result)

        cached_stores[p].append_toc(cached_stores[p].prepare_toc(of_paths=update))
        return (at_path, hdf_group)

    return write_hdf


def get_executable(computation_type, parameters):
    """..."""
    executable_type = EXECUTABLE[computation_type.split('-')[0]]

    try:
        executable = parameters[executable_type]
    except KeyError as err:
        raise RuntimeError(f"No {executable_type} defined for {computation_type}") from err

    _, execute = plugins.import_module(executable["source"], executable["method"])

    return execute


def store_node_properties_batch(of_population, on_compute_node, in_hdf_group):
    """...This will extract node properties for all subtargets as a single datasframe.
    NOT-IDEAL and needs hacks to gather differemt resuts into the same input dataframe.
    REPLACE by single subtarget store using matrices
    """
    def write_batch(connsense_h5, results):
        """..."""
        in_hdf = (on_compute_node/connsense_h5, in_hdf_group)
        LOG.info("Write %s  results %s ", in_hdf, len(results))
        return extract_nodes.write(results, of_population, in_hdf)

    return write_batch


def store_node_properties(of_population, on_compute_node, in_hdf_group):
    """..."""
    LOG.info("Store node properties of population %s on compute node %s in hdf %s, one subtarget at a time",
             of_population, on_compute_node, in_hdf_group)

    def write_hdf(at_path, *, result=None, update=None):
        """..."""
        assert not(result is None and update is None)
        assert result is not None or update is not None

        hdf_population = in_hdf_group+'/'+of_population
        store = matrices.get_store(on_compute_node/at_path, hdf_population, "pandas.DataFrame")

        if result is not None:
            return store.write(result)

        store.append_toc(store.prepare_toc(of_paths=update))
        return (at_path, hdf_population)

    return write_hdf


def store_edge_extraction(of_population, on_compute_node, in_hdf_group):
    """..."""
    def write_batch(connsense_h5, results):
        """..."""
        in_hdf = (on_compute_node/connsense_h5, f"{in_hdf_group}/{of_population}")
        LOG.info("Write %s batch results to %s", len(results), in_hdf)
        return extract_connectivity.write_adj(results, to_output=in_hdf,  append=True, format="table",
                                              return_config=True)

    return write_batch


def store_matrix_data(of_quantity, parameters, on_compute_node, in_hdf_group, slicing):
    """..."""
    LOG.info("Store matrix data for %s", parameters)
    of_output = parameters["output"]
    hdf_group = f"{in_hdf_group}/{of_quantity}"
    if slicing:
        hdf_group = f"{hdf_group}/{slicing}"

    cached_stores = {}

    def write_hdf(at_path, *, result=None, update=None):
        """..."""
        assert at_path
        assert not(result is None and update is None)
        assert result is not None or update is not None

        p = on_compute_node/at_path
        if p not in cached_stores:
            cached_stores[p] = matrices.get_store(p, hdf_group, for_matrix_type=of_output)

        if result is not None:
            return cached_stores[p].write(result)

        cached_stores[p].append_toc(cached_stores[p].prepare_toc(of_paths=update))
        return (at_path, hdf_group)

    return write_hdf


def configure_execution(computation, in_config, on_compute_node, slicing=None):
    """..."""
    computation_type, of_quantity = describe(computation)
    parameters = parameterize(computation_type, of_quantity, in_config)
    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, in_hdf_group = output_paths["steps"][computation_type]

    execute = get_executable(computation_type, parameters)

    if computation_type == "extract-node-populations":
        assert not slicing, "Does not apply"
        return (execute, None,  store_node_properties(of_quantity, on_compute_node, in_hdf_group))
        #return (execute, store_node_properties(of_quantity, on_compute_node, in_hdf_group), None)

    if computation_type == "extract-edge-populations":
        assert not slicing, "Does not apply"
        return (execute, store_edge_extraction(of_quantity, on_compute_node, in_hdf_group), None)

    return (execute, None, store_matrix_data(of_quantity, parameters, on_compute_node, in_hdf_group, slicing))


def collect_batches(of_computation, results, on_compute_node, hdf_group, slicing, of_output_type):
    """..."""
    LOG.info("Collect bactched %s results of %s on compute node %s in group %s output type %s",
             of_computation, len(results), on_compute_node, hdf_group, of_output_type)
    computation_type, of_quantity = describe(of_computation)

    if computation_type == "extract-edge-populations":
        assert not slicing, "Does not apply"
        return collect_batched_edge_population(of_quantity, results, on_compute_node, hdf_group)

    if computation_type == "extract-node-populations":
        assert not slicing, "Does not apply"

    hdf_group = hdf_group+"/"+of_quantity
    if slicing:
        hdf_group = hdf_group+"/"+slicing
    in_connsense_h5 = on_compute_node / "connsense.h5"
    in_store = matrices.get_store(in_connsense_h5, hdf_group, for_matrix_type=of_output_type)

    batched = results.items()
    in_store.collect({batch: matrices.get_store(on_compute_node / batch_connsense_h5, hdf_group,
                                                for_matrix_type=of_output_type)
                      for batch, (batch_connsense_h5, group) in batched})
    return (in_connsense_h5, hdf_group)


def collect_batched_node_population(p, results, on_compute_node, hdf_group):
    """..."""
    from connsense.io.write_results import read as read_batch, write as write_batch

    LOG.info("Collect batched node populations of %s %s results on compute-node %s to %s", p,
             len(results), on_compute_node, hdf_group)

    in_connsense_h5 = on_compute_node / "connsense.h5"

    hdf_node_population = (in_connsense_h5, hdf_group+"/"+p)

    def move(batch, output):
        """..."""
        LOG.info("Write batch %s read from %s", batch, output)
        result = read_batch(output, "extract-node-populations")
        return write_batch(result, to_path=hdf_node_population, append=True, format="table")

    LOG.info("collect batched extraction of nodes at compute node %s", on_compute_node)
    for batch, output in results.items():
        move(batch, output)

    LOG.info("DONE collecting %s", results)
    return hdf_node_population


def collect_batched_edge_population(p, results, on_compute_node, hdf_group):
    """..."""

    in_connsense_h5 = on_compute_node / "connsense.h5"

    hdf_edge_population = (in_connsense_h5, hdf_group+'/'+p)

    def move(batch, output):
        """.."""
        LOG.info("collect batch %s of adjacencies at %s output %s ", batch, on_compute_node, output)
        adjmats = read_toc_plus_payload(output, for_step="extract-edge-populations")
        return write_toc_plus_payload(adjmats, hdf_edge_population, append=True, format="table",
                                      min_itemsize={"values": 100})

    LOG.info("collect batched extraction of edges at compute node %s", on_compute_node)
    for batch, output in results.items():
        move(batch, output)

    LOG.info("DONE collecting %s", results)
    return hdf_edge_population



def run_cleanup(on_compute_node):
    """..."""
    if on_compute_node.joinpath(INPROGRESS).exists() or on_compute_node.joinpath(DONE).exists():
        LOG.warning("Compute node has been run before: %s", on_compute_node)

        archive = on_compute_node.parent / "history"
        archive.mkdir(parents=False, exist_ok=True)

        history_compute_node = archive/on_compute_node.name
        if history_compute_node.exists():
            LOG.warning("Other than the existing run, there were previous ones too: \n%s",
                        list(history_compute_node.glob('*')))

        to_archive = history_compute_node/time.stamp(now=True)
        if to_archive.exists():
            LOG.warning("The last run archived at \n %s \n"
                        "must have been within the last minute of now (%s) and may be overwritten",
                        to_archive, time.stamp(now=True))
        shutil.copytree(on_compute_node, to_archive,
                        symlinks=False, ignore_dangling_symlinks=True, dirs_exist_ok=True)

    files_to_remove = ([on_compute_node / path for path in ("batched_output.json", "output.json",
                                                            INPROGRESS, DONE)]
                       + list(on_compute_node.glob("connsense*.h5")))
    LOG.info("On compute node %s, cleanup by removing files \n%s", on_compute_node.name, files_to_remove)
    for to_remove in files_to_remove:
        to_remove.unlink(missing_ok=True)

    return on_compute_node



def load_control(transformations, lazily=True):
    """..."""
    def load_config(control, description):
        """..."""
        LOG.info("Load configured control %s: \n%s", control, pformat(description))

        _, algorithm = plugins.import_module(description["algorithm"])

        seeds = description.get("seeds", [0])

        try:
            to_tap = description["tap_datasets"]
        except KeyError:
            to_tap = None

        kwargs = description.get("kwargs", {})

        def seed_shuffler(s):
            """..."""
            def shuffle(inputs):
                """..."""
                if lazily:
                    return lambda: algorithm(**inputs(), seed=s, **kwargs)
                return algorithm(**inputs, seed=s, **kwargs)
            return (f"{control}-{s}", shuffle, to_tap)
        return [seed_shuffler(s) for s in seeds]

    controls = {k: v for k, v in transformations.items() if k != "description"}
    return [shuffled for ctrl, cfg in controls.items() for shuffled in load_config(ctrl, cfg)]



def parse_slices(slicing):
    """..."""
    from itertools import product
    def prepare_singleton(slicespec):
        """..."""
        assert len(slicespec) == 2, "must be a tuple from a dict items"
        key, values = slicespec
        if isinstance(values, list):
            return ((key, value) for value in values)
        if isinstance(values, Mapping):
            if len(values) == 1:
                innerkey, innervalues = next(iter(values.items()))
                if not isinstance(innervalues, list):
                    innervalues = [innervalues]
                return ((key, {innerkey: val}) for val in innervalues)
            innerdicts = product(*(s for s in (prepare_singleton(slicespec=s)  for s in values.items())))
            return ((key, dict(dvalue)) for dvalue in innerdicts)

        return ((key, value) for value in [values])

    slicing = slicing["slices"].items()
    if len(slicing) == 1:
        return (dict([s]) for s in prepare_singleton(next(iter(slicing))))

    slices = product(*(singleton for singleton in (prepare_singleton(slicespec=s) for s in slicing)))
    return (dict(s) for s in slices)


def flatten_slicing(_slice):
    """..."""
    def denest(key, value):
        if not isinstance(value, dict):
            return value
        return {f"{key}_{innerkey}": denest(innerkey, innervalue)
                for innerkey, innervalue in value.items()}
    if len(_slice) == 1:
        key, value = next(iter(_slice.items()))
        return {key: denest(key, value)}
    flat = {}
    for var, values in _slice.items():
        denested = denest(var, values)
        flat.update(denested)
    return flat


def load_slicing(transformations, using_tap=None, lazily=True):
    """..."""
    from copy import deepcopy

    def load_dataset(slicing):
        """..."""
        slicing = deepcopy(slicing)
        slices = slicing["slices"]
        def load_dataset(values):
            """..."""
            if isinstance(values, dict):
                if "dataset" in values:
                    return using_tap.pour_dataset(*values["dataset"]).tolist()
                return {var: load_dataset(vals) for var, vals in values.items()}
            return values
        slicing["slices"] = {variable: load_dataset(values) for variable, values in slices.items()}
        return slicing

    def load(slicing):
        """..."""
        _, algorithm = plugins.import_module(slicing["algorithm"])
        kwargs = slicing.get("kwargs", {})
        slices = list(parse_slices(load_dataset(slicing)))
        def specify(aslice):
            """..."""
            def slice_input(datasets):
                if lazily:
                    assert callable(datasets)
                    return lambda: algorithm(**datasets(), **aslice, **kwargs)
                assert not callable(datasets)
                return algorithm(**datasets, **aslice, **kwargs)
            return slice_input
        return pd.Series([specify(aslice) for aslice in slices],
                         index=pd.MultiIndex.from_frame(pd.DataFrame([flatten_slicing(s) for s in slices])))
    return {slicing: load(slicing=s) for slicing, s in transformations.items()}



def input_units(computation, to_tap):
    """..."""
    described = parameterize(*describe(computation), to_tap._config)
    datasets = {variable: apply_transformations(in_values, to_tap, variable,
                                                load_dataset(to_tap, variable, in_values), of_analysis=computation)
                for variable, in_values in filter_datasets(described["input"]).items()}
    return datasets
