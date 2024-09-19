

# For development we can use a develop version,


from collections.abc import Mapping
from copy import deepcopy
import shutil
from pathlib import Path
from lazy import lazy
from pprint import pformat

import json
import yaml

import multiprocessing
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



def batch_multinode(computation, of_inputs, in_config, at_dirpath, using_parallelization,
                    single_submission=250,  with_weights=True, unit_weight=None,
                    njobs_to_estimate_load=None, max_parallel_jobs=None):
    """...Just read the method definition above,
        and code below
    """
    from tqdm import tqdm; tqdm.pandas()

    n_compute_nodes, n_parallel_jobs, order_complexity = using_parallelization
    n_parallel_biggest = int(n_parallel_jobs / n_compute_nodes)

    LOG.info("Assign compute-nodes to %s inputs", len(of_inputs))
    toc_index = of_inputs.index

    weights = (
        #of_inputs.progress_apply(lambda l: estimate_load(to_compute=None)(l())).dropna()
        of_inputs.progress_apply(estimate_load(order_complexity)).dropna()
        if not njobs_to_estimate_load
        else multiprocess_load_estimate(order_complexity, of_inputs, njobs_to_estimate_load))
    weights = (
        weights[~np.isclose(weights, 0.)].groupby(toc_index.names).max()
        .sort_values(ascending=True)).rename("weight")

    unit_weight = max(unit_weight or 0.,  weights.max())

    compute_nodes = ((n_compute_nodes * (np.cumsum(weights) / weights.sum() - 1.e-6))
                     .astype(int).rename("compute_node"))
    LOG.info(
        "Assign batches to subtargets on %s compute_nodes: \n%s",
        len(compute_nodes), pformat(compute_nodes))


    def weigh_one(compute_node):
        return weights.loc[compute_node.index]

    def batch(compute_node):
        """..."""
        cn_weights = weigh_one(compute_node)
        n_parallel = int(min(max_parallel_jobs or multiprocessing.cpu_count(), #/2
                             min(int(n_parallel_biggest * unit_weight / cn_weights.max()),
                                 len(cn_weights))))
        #batches = np.random.choice(np.arange(n_parallel), size=len(cn_weights), replace=True)
        n_comp = len(cn_weights)
        batch_values = list(range(n_parallel))
        batches = ((int(n_comp / n_parallel) + 1) * batch_values)[0:n_comp]
        return pd.Series(batches, name="batch", index=cn_weights.index)

    batches = (
        pd.concat([batch(compute_nodes)], axis=0,
                   keys=[compute_nodes.unique()[0]], names=["compute_node"])
        if compute_nodes.nunique() == 1 else
        pd.DataFrame(compute_nodes).groupby("compute_node").apply(batch)
    ).reorder_levels(compute_nodes.index.names + ["compute_node"])

    if not with_weights:
        return batches

    if not isinstance(weights.index, pd.MultiIndex):
        weights.index = (pd.MultiIndex
                         .from_arrays([weights.index.values], names=[weights.index.name]))

    cn_weights = (
        pd.concat([weigh_one(compute_nodes)], keys=[compute_nodes.unique()[0]],
                  names=["compute_node"])
        if compute_nodes.nunique() == 1
        else pd.DataFrame(compute_nodes).groupby("compute_node").apply(weigh_one)
    ).reorder_levels(compute_nodes.index.names + ["compute_node"])

    assignment = pd.concat([batches, cn_weights], axis=1) if with_weights else batches
    if len(compute_nodes) > single_submission:
        LOG.info(
            "Too many compute nodes (%s) for a single submission (upper limit %s)\n"
            "Create multiple submissions.", len(compute_nodes), single_submission)
        assignment = assignment.assign(
            submission=(compute_nodes // single_submission).rename("submission"))
        LOG.info(
            "Submit %s compute nodes in %s submissions",
            len(compute_nodes), assignment.submission.max())
    if at_dirpath:
        assignment_h5, dataset = COMPUTE_NODE_ASSIGNMENT
        assignment.to_hdf(at_dirpath/assignment_h5, key=dataset)
    return assignment


def batch_parallel_groups(of_inputs, upto_number, to_compute=None, return_load=False):
    """..."""
    from tqdm import tqdm; tqdm.pandas()

    if isinstance(of_inputs, pd.Series):
        weights = (of_inputs
                   .progress_apply(estimate_load(to_compute)).rename("load")
                   .sort_values(ascending=True))
    elif isinstance(of_inputs, pd.DataFrame):
        weights = (of_inputs
                   .progress_apply(estimate_load(to_compute), axis=1).rename("load")
                   .sort_values(ascending=True))
    else:
        raise TypeError(f"Unhandled type of input: {of_inputs}")

    nan_weights = weights[weights.isna()]
    if len(nan_weights) > 0:
        LOG.warning("No input data for %s / %s of_inputs:\n%s",
                    len(nan_weights), len(weights), pformat(nan_weights))
        weights = weights.dropna()

    computational_load = (np.cumsum(weights) / weights.sum()).rename("load")
    n = np.minimum(upto_number, len(weights))
    batches = ((n * (computational_load - computational_load.min()))
               .apply(int).rename("batch"))

    LOG.info("Load balanced batches for %s of_inputs: \n %s",
             len(of_inputs), batches.value_counts())
    return (batches if not return_load
            else pd.concat([batches, weights/weights.sum()], axis=1))

def estimate_load(order_complexity, to_compute=None):
    def of_input_data(d):
        """What would it take to compute input data d?
        """
        LOG.info("Estimate load to compute a %s", type(d))
        if d is None:
            return None

        try:
            S = d.shape
        except AttributeError:
            pass
        else:
            if order_complexity == -1:
                return np.prod(S)

            assert order_complexity >= 0

            return (np.nan if np.isnan(S[0]) else
                    0 if np.isclose(S[0], 0.0) else S[0] ** order_complexity)

        if callable(d):
            try:
                dload = d(to_get="shape")
            except TypeError:
                dload = d()
            return of_input_data(dload)

        if isinstance(d, Mapping):
            if not d: return 1
            first = next(v for v in d.values())
            return of_input_data(first)

        if isinstance(d, pd.Series):
            return d.apply(of_input_data).sum()

        try:
            N = len(d)
        except TypeError as terror:
            try:
                S = d.shape
            except AttributeError as aerror:
                LOG.error("Neither length, nor shape for input \n%s\n%s\n%s",
                          d, terror, aerror)
                return 1
            else:
                N = S[0]

        if N == 0:
            return 0

        if order_complexity == -1:
            try:
                S = d.shape
            except AttributeError:
                return N
            return np.prod(S)

        return N ** order_complexity

    return of_input_data


def multiprocess_load_estimate(order_complexity, inputs, njobs):
    """..."""
    from tqdm import tqdm; tqdm.pandas()
    assert njobs > 1, f"njobs={njobs} does not need multi-process."

    def weigh(batch_inputs, *, in_bowl, index):
        """..."""
        weight = batch_inputs.progress_apply(estimate_load(order_complexity))
        in_bowl[index] = weight
        return weight

    manager = Manager()
    bowl = manager.dict()
    processes = []

    batched_inputs = pd.DataFrame({"input": inputs,
                                   "batch": np.linspace(0, njobs - 1.e-6, len(inputs), dtype=int)})
    for b, batch in batched_inputs.groupby("batch"):
        LOG.info("Estimate load of %s inputs in batch %s / %s", len(batch), b, njobs)
        p = Process(target=weigh, args=(batch.input,), kwargs={"index": b, "in_bowl": bowl})
        p.start()
        processes.append(p)
    LOG.info("LAUNCHED %s processes", len(processes))

    for p in processes:
        LOG.info("Join process %s", p)
        p.join()
    LOG.info("Parallel load estimation results %s", len(bowl))

    return pd.concat([weights for weights in bowl.values()])


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
        LOG.info("No index configured for computation %s: \n%s",
                 of_computation, missing_index)
        try:
            LOG.info("read index from the configured input.")
            return parameters["input"]
        except KeyError as missing_input:
            LOG.info("Neither an index, nor inputs were configured for computation %s",
                     of_computation)
            raise NotConfiguredError("%s `input` was not configured %s") from missing_input
    raise RuntimeError("Python executtion must not reach here.")


def index_inputs(of_computation, in_tap):
    """..."""
    index_vars = read_index(of_computation, in_tap._config)

    if len(index_vars) > 1:
        return pd.MultiIndex.from_product([to_tap.subset_index(var, values)
                                           for var, values in index_vars.items()])

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
                and any(dataterm in val
                        for dataterm in ("dataset", "datacall", "datajoin", "dataprod")))}

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
    properties = values.get("properties", None)

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

        data = get()
        return data if properties is None else data[properties]

    try:
        dset = values["dataset"]
    except KeyError:
        try:
            to_call = values["datacall"]
        except KeyError:
            try:
                to_join = values["datajoin"]
            except KeyError:
                try:
                    to_prod = values["dataprod"]
                except KeyError:
                    raise ValueError("values need to define either"
                                     " dataset, datacall, datajoin, or dataprod")
                else:
                    dataset = cross_datasets(tap, variable, to_prod)
            else:
                dataset = mix_datasets(tap, variable, recipe=to_join)
        else:
            dataset = brew_dataset(tap, variable, to_call)
    else:
        LOG.info("Pour %s dataset: \n%s", variable, dset)
        #columns = values.get("columns", None)
        dataset = tap.pour_dataset(*dset).apply(lazily(to_evaluate=unpack_value))

    if isinstance(dataset, pd.DataFrame):
        return (pd.Series([r for _, r in dataset.iterrows()], index=dataset.index)
                .apply(DataCall))

    try:
        values_reindex = values["reindex"]
    except KeyError:
        pass
    else:
        dataset = reindex(tap, dataset, values_reindex)

    try:
        subset = values["subset"]
    except KeyError:
        return dataset
    return dataset.loc[subset]


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

def brew_dataset(tap, variable, call):
    """..."""
    in_store = pour(tap, call["input"]).apply(lazy_keyword).apply(DataCall)
    _, recipe = plugins.import_module(call["recipe"])
    return in_store.apply(bind_lazy(call=recipe, **call["kwargs"]))

def mix_datasets(tap, variable, recipe):
    """..."""
    how = recipe.get("how", "cross")
    dsets = [(var, load_dataset(tap, var, values)) for var, values in recipe.items()
             if var!= "how"]

    assert len(dsets) > 0
    assert isinstance(dsets[0][1], pd.Series)
    if len(dsets) == 1:
        to_include_varname = {varid: dsets[0][0] + '_' + varid
                              for varid in dsets[0][1].index.names
                              if varid not in ("circuit_id", "connectome_id")}
        dataset = dsets[0][1].copy()
        dataset.index.rename(to_include_varname, inplace=True)
        return dataset

    assert len(dsets) == 2

    assert isinstance(dsets[1][1], pd.Series)
    assert dsets[0][1].name == dsets[1][1].name
    quantity = dsets[0][1].name

    assert dsets[1][1].index.names == dsets[0][1].index.names

    def merge(d0, d1):
        assert "circuit_id" not in d0.index.names
        assert "connectome_id" not in d0.index.names
        idxvars = d0.index.names

        dd = (pd.merge(d0.reset_index(), d1.reset_index(),
                       suffixes=("_"+n for n,_ in dsets), how=how)
              .rename(columns={c+'_'+n:n+'_'+c for c in idxvars for n,_ in dsets}))
        return (dd.set_index([n+'_'+c for c in idxvars for n,_ in dsets])
                .rename(columns={quantity+'_'+n: n for n,_ in dsets}))
                #.rename(columns={quantity+'_'+n: n+'_'+quantity for n,_ in dsets}))

    try:
        circuits_0 = dsets[0][1].index.get_level_values("circuit_id").unique()
    except KeyError:
        circuits = None
    else:
        circuits_1 = dsets[0][1].index.get_level_values("circuit_id").unique()
        circuits = circuits_0.intersection(circuits_1)

    if circuits is None:
        return merge(dsets[0][1], dsets[1][1])

    def merge_circuit(c):
        d0 = dsets[0][1].xs(c, level="circuit_id")
        d1 = dsets[1][1].xs(c, level="circuit_id")
        try:
            connectomes_0 = d0.index.get_level_values("connectome_id").unique()
        except KeyError:
            return merge(d0, d1)
        connectomes_1 = d1.index.get_level_values("connectome_id").unique()
        connectomes = connectomes_0.intersection(connectomes_1)

        def merge_connectome(x):
            d0_no_conn = d0.xs(x, level="connectome_id")
            d1_no_conn = d1.xs(x, level="connectome_id")
            return merge(d0_no_conn, d1_no_conn)

        merged = pd.concat([merge_connectome(x) for x in connectomes], axis=0,
                           keys=connectomes, names=["connectome_id"])

    merged = pd.concat([merge_circuit(c) for c in circuits], axis=0,
                       keys=circuits, names=["circuit_id"])

    indices = (merged.index.names[1:] + ["circuit_id"]
               if "connectome_id" not in merged.index.names else
               merged.index.names[2:] + ["circuit_id", "connectome_id"])

    return merged.reorder_levels(indices).apply(lambda r: r.to_dict(), axis=1).rename(variable)


def cross_datasets(tap, variable, recipe):
    """..."""
    assert isinstance(recipe, list)
    assert len(recipe) == 2

    dsets = [load_dataset(tap, variable, values) for values in recipe]

    assert len(dsets) == 2

    assert isinstance(dsets[0], pd.Series)
    assert isinstance(dsets[1], pd.Series)
    assert dsets[0].name == dsets[1].name
    quantity = dsets[0].name

    assert dsets[1].index.names == dsets[0].index.names

    def merge(d0, d1):
        assert "circuit_id" not in d0.index.names
        assert "connectome_id" not in d0.index.names
        idxvars = d0.index.names

        dd = pd.merge(d0.reset_index(), d1.reset_index(), how="cross")
        mdd = dd.set_index([f"{c}_x" for c in idxvars] + [f"{c}_y" for c in idxvars])

        if len(idxvars) == 1:
            mdd.index = pd.Index(mdd.index.values, name=idxvars[0])
        else:
            idxframe = pd.DataFrame({var: (mdd.index.to_frame()[[f"{var}_x", f"{var}_y"]]
                                           .apply(tuple, axis=1))
                                     for var in idxvars})
            mdd.index = pd.MultiIndex.from_frame(idxframe)

    try:
        circuits_0 = dsets[0].index.get_level_values("circuit_id").unique()
    except KeyError:
        circuits = None
    else:
        circuits_1 = dsets[0].index.get_level_values("circuit_id").unique()
        circuits = circuits_0.intersection(circuits_1)

    if circuits is None:
        return merge(dsets[0], dsets[1])

    def merge_circuit(c):
        d0 = dsets[0].xs(c, level="circuit_id")
        d1 = dsets[1].xs(c, level="circuit_id")
        try:
            connectomes_0 = d0.index.get_level_values("connectome_id").unique()
        except KeyError:
            return merge(d0, d1)
        connectomes_1 = d1.index.get_level_values("connectome_id").unique()
        connectomes = connectomes_0.intersection(connectomes_1)

        def merge_connectome(x):
            d0_no_conn = d0.xs(x, level="connectome_id")
            d1_no_conn = d1.xs(x, level="connectome_id")
            return merge(d0_no_conn, d1_no_conn)

        merged = pd.concat([merge_connectome(x) for x in connectomes], axis=0,
                           keys=connectomes, names=["connectome_id"])

    merged = pd.concat([merge_circuit(c) for c in circuits], axis=0,
                       keys=circuits, names=["circuit_id"])

    indices = (merged.index.names[1:] + ["circuit_id"]
               if "connectome_id" not in merged.index.names else
               merged.index.names[2:] + ["circuit_id", "connectome_id"])

    return merged.reorder_levels(indices).apply(lambda r: (r["x"], r["y"]), axis=1).rename(variable)


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



def pour_cross(tap, datasets):
    """...develop a version of pour that can cross datasets.
    The levels of circuit_id and connectome_id in the right dataset will be dropped,
    assuming that the dataset has already been filtered to contain circuit_id and connectome_id
    that are present in the left dataset.
    """
    LOG.info("Pour cross product of tap \n%s\n values for variables:\n%s",
             tap._root, pformat(datasets))
    dsets = sorted([(variable, load_dataset(tap, variable, values))
                    for variable, values in datasets.items()],
                   key=lambda x: len(x[1].index.names), reverse=True)

    assert len(dsets) == 2, f"Cross can only be between two datasets. Provided: {len(dsets)}"

    assert dsets[0][1].index.names == dsets[1][1].index.names, "Index variables must be the same."
    circargs = [level for level in dsets[0][1].index.names if level in ("circuit_id", "connectome_id")]

    def prefix_dset(name, to_index):
        return {varid: f"{name}_{varid}" for varid in to_index.names if varid not in circargs}

    left_name = dsets[0][0]
    left = dsets[0][1].rename(left_name)
    left_index = prefix_dset(left_name, left.index)
    LOG.info("rename left index to:\n%s", pformat(left_index))
    left.index.rename(left_index, inplace=True)

    right_name = dsets[1][0]
    right = dsets[1][1].rename(right_name)
    right_index = prefix_dset(right_name, right.index)
    LOG.info("rename right index to:\n%s", pformat(right_index))
    right.index.rename(right_index, inplace=True)

    return (pd.merge(left, right, left_index=True, right_index=True)
            .reorder_levels(list(left_index.values()) + list(right_index.values()) + circargs)
            .apply(lambda row: row.to_dict(), axis=1))


def pour(tap, datasets):
    """..."""
    LOG.info("Pour tap \n%s\n to get values for variables:\n%s",
             tap._root, pformat(datasets))

    dsets = sorted([(variable, load_dataset(tap, variable, values))
                    for variable, values in datasets.items()],
                   key=lambda x: len(x[1].index.names), reverse=True)

    def rename_series(dset, data):
        try:
            rename = data.rename
        except AttributeError:
            LOG.warning("Dataset reference %s does not point to a pandas.Series", dset)
            return data
        return rename(dset)

    primary = rename_series(*dsets[0])

    if len(dsets) == 1:
        return (primary.apply(lambda value: {dsets[0][0]: value})
                if isinstance(primary, pd.Series) else
                primary.apply(lambda row: row.to_dict(), axis=1))

    common_index = dsets[0][1].index.names

    def rename_index(dset, data):
        if not (join_index := datasets[dset].get("join_index", None)):
            return rename_series(dset, data)

        data = data.droplevel([varid for varid, revarid in join_index.items()
                               if revarid.lower() == "drop"])
        data.index.rename({varid: revarid for varid, revarid in join_index.items()
                           if revarid and revarid.lower() != "drop"}, inplace=True)
        return rename_series(dset, data)

    def merge_with(leading, dset, data):
        reindexed = rename_index(dset, data)
        return pd.merge(leading, reindexed, left_index=True, right_index=True)

    leading = primary
    for dset, data in dsets[1:]:
        leading = merge_with(leading, dset, data)
    return leading.apply(lambda row: row.to_dict(), axis=1).reorder_levels(common_index)

    # def reindex (dset):
    #     in_dset = dset[1].index.names
    #     not_in_dset = [n for n in primary.index.names if n not in in_dset]
    #     return (rename_index(*dset)
    #             .reindex(primary.reorder_levels(in_dset + not_in_dset).index)
    #             .reorder_levels(primary.index.names))

    # return (pd.concat([reindex(dset) for dset in dsets], axis=1,
    #                   keys=[name for name, _ in dsets])
    #         .apply(lambda row: row.to_dict(), axis=1))


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
        else:
            LOG.info("Pipeline for %s: \n%s", computation_type, pformat(cfg_computation_type))

        paramkey = PARAMKEY[computation_type]
        try:
            quantities_to_configure = cfg_computation_type[paramkey]
        except KeyError:
            LOG.warning("No quantities %s: \n%s", computation_type, cfg_computation_type)
            return None
        else:
            LOG.info("Configure runtime for %s %s", computation_type, quantities_to_configure)

        try:
            runtime = from_runtime[computation_type]
        except KeyError:
            LOG.warning("No runtime configured for computation type %s", computation_type)
            return None
        else:
            LOG.info("Use configuration: \n%s", pformat(runtime))

        configured = runtime[paramkey]

        if not configured:
            return None

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

    runtime_pipeline = {c: configure_slurm_for(computation_type=c) for c in of_pipeline["parameters"]
                        if c != "description"}
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
    n_compute_nodes, n_parallel_batches =  read_njobs(to_parallelize=configured,
                                                      computation_of=quantity)
    #order_complexity = configured[quantity].get("order_complexity", -1)
    order_complexity = _read_runtime("order_complexity",
                                     to_parallelize=configured, computation_of=quantity,
                                     default=-1)
    return (n_compute_nodes, n_parallel_batches, order_complexity)


def _read_runtime(key, to_parallelize, computation_of, default=None):
    """..."""
    if not to_parallelize:
        return default
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
            except ValueError:
                return default
            else:
                try:
                    p0 = to_parallelize[q0]
                except KeyError:
                    return default
                else:
                    try:
                        p = p0[q1]
                    except KeyError:
                        return default
                    else:
                        pass
        else:
            return default

    return p.get(key, default)

def read_njobs(to_parallelize, computation_of):
    """..."""
    compute_nodes = _read_runtime("number-compute-nodes", to_parallelize, computation_of,
                                  default=1)
    tasks = _read_runtime("number-tasks-per-node", to_parallelize, computation_of,
                          default=1)
    return (compute_nodes, compute_nodes * tasks)

def write_description(of_computation, in_config, at_dirpath):
    """..."""
    computation_type, of_quantity = describe(of_computation)
    configured = deepcopy(parameterize(computation_type, of_quantity, in_config))
    configured["name"] = of_quantity

    LOG.info("Write setup description of computation %s in config %s\n \n\t at path %s",
             of_computation, pformat(configured), at_dirpath)

    return read_pipeline.write(configured, to_json=at_dirpath/"description.json")


def reindex(tap, inputs, variables):
    """..."""
    connsense_ids = {
        v: tap.index_variable(value) for v, value in variables.items()}

    def index_ids(subtarget):
        frame = pd.DataFrame({
            f"{var}_id": connsense_ids[var].loc[values.values].values
            for var, values in subtarget.index.to_frame().items()})
        return pd.MultiIndex.from_frame(frame)

    reinputs = inputs.apply(lambda s: pd.DataFrame(s()).set_index(index_ids(subtarget=s())))
    frame = pd.concat(reinputs.values, keys=reinputs.index)
    groups_of_inner_frames = list(frame.groupby(frame.index.names))

    return(
        pd.Series(
            data=[
                d.reset_index(drop=True) for _, d in groups_of_inner_frames],
            index=pd.MultiIndex.from_tuples(
                [i for i,_ in groups_of_inner_frames], names=frame.index.names))
        .apply(DataCall))


class DataCall:
    """Call data..."""
    def __init__(self, dataitem, transform=None, preserves_shape=True, cache=False):
        self._dataitem = dataitem
        self._transform = transform
        self._preserves_shape = preserves_shape
        self._cache = None if not cache else {}

    @lazy
    def dataset(self):
        """..."""
        LOG.warning("This will hold the result of this DataCall. DO NOT USE with a series.apply")
        return self()

    @lazy
    def shape(self):
        """..."""
        LOG.debug("Get shape for DataCall %s, \n\t with dataitem %s", self, self._dataitem)
        if self._transform and self._preserves_shape:
            return self._dataitem.shape

        value = self(to_get="shape")

        if isinstance(value, Mapping):
            value = next(v for v in value.values())

        try:
            return value.shape
        except AttributeError:
            pass

        try:
            length = len(value)
        except TypeError:
            length = 1

        return (length, )

    def transform(self, original=None):
        """..."""
        assert self._transform is not None

        if original is None:
            original = self._cache["original"]

        try:
            transform, kwargs = self._transform
        except TypeError as not_tuple_error:
            try:
                return self._transform(original)
            except TypeError as not_callable_error:
                LOG.error("self._transform should be a callable or tuple of (callable, kwargs)"
                          "not %s\nErrors:\n%s\n%s",
                          type(self._transform), not_tuple_error, not_callable_error)
                raise TypeError("self._transform type %s inadmissible."
                                " Plesase use a (callable, kwargs) or callable"%
                                (type(self._transform),))
            return transform(original, **kwargs)

    def __call__(self, to_get=None):
        """Call Me."""
        LOG.debug("Data called %s. | Has Transform? %s | Will it Cache? %s|", self,
                  self._transform is not None, self._cache is not None)

        if self._cache is not None and "original" in self._cache:
            original = self._cache["original"]
        else:
            try:
                get_value = self._dataitem.get_value
            except AttributeError:
                try:
                    original = self._dataitem(to_get)
                except TypeError:
                    try:
                        original = self._dataitem()
                    except TypeError:
                        original = self._dataitem
            else:
                original = get_value()

            if self._cache is not None:
                self._cache["original"] = original

        if not self._transform:
            return original

        if to_get and to_get.lower() == "shape":
            return original if self._preserves_shape else self.transform(original)

        return self.transform(original)

def generate_inputs(of_computation, in_config, slicing=None, circuit_args=None,
                     datacalls_for_slices=False, **circuit_kwargs):
    """..."""
    from connsense.develop.topotap import HDFStore
    LOG.info("Generate inputs for %s.", of_computation)

    computation_type, of_quantity = describe(of_computation)
    params = parameterize(computation_type, of_quantity, in_config)
    tap = HDFStore(in_config)

    join_input = params["input"].get("join", None)

    if not join_input:
        datasets = pour(tap, filter_datasets(params["input"]))
    elif join_input.lower() == "cross":
        datasets = pour_cross(tap, filter_datasets(params["input"]))
    else:
        raise ValueError(f"Unhandled join of inputs: {join_input}")

    original = datasets.apply(lazy_keyword).apply(DataCall)

    controlled = control_inputs(of_computation, in_config, using_tap=tap)
    if controlled is not None:
        original = pd.concat([original], axis=0, keys=["original"], names=["control"])
        full = pd.concat([original, controlled])
        full = full.reorder_levels([l for l in full.index.names if l != "control"]
                                   + ["control"])
    else:
        full = original

    def index_circuit_args(inputs):
        """..."""
        if circuit_args is None:
            return inputs

        missing = {}
        for var, value in circuit_args.items():
            varid = f"{var}_id"
            idval = tap.index_variable(var, value)

            if varid not in inputs.index.names:
                missing[varid] = idval
            else:
                inputs = inputs.xs(idval, level=varid, drop_level=False)

        #missing = {f"{var}_id": tap.index_variable(var, value)
                   #for var, value in circuit_args.items()
                   #if f"{var}_id" not in inputs.index.names}
        for variable_id, value in missing.items():
            inputs = (pd.concat([inputs], axis=0, keys=[value], names=[variable_id])
                      .reorder_levels(inputs.index.names + [variable_id]))
        return inputs

    if not slicing or slicing == "full":
        LOG.info("generate inputs for slicing None or full %s", slicing)
        return index_circuit_args(full)

    assert slicing in params["slicing"]
    cfg = {slicing: params["slicing"][slicing]}

    if (not datacalls_for_slices
        and (cfg[slicing].get("compute_mode", "EXECUTE") in ("execute", "EXECUTE"))):
        LOG.info("generate inputs for slicing %s with compute mode execute", slicing)
        return index_circuit_args(full)

    LOG.info("generate inputs for slicing %s with compute mode not execute", slicing)
    to_cut = load_slicing(cfg, tap, lazily=False, **circuit_kwargs)[slicing]
    slices = generate_slices(tap, inputs=full, using_knives=to_cut)
    return index_circuit_args(slices)


def generate_slices(of_tap, inputs, using_knives):
    """Generate slices of inputs accoring to configured knives."""
    from tqdm import tqdm; tqdm.pandas()
    def datacut(c):
        return datacall(c, preserves_shape=False)
    slices = pd.concat([inputs.apply(datacut(c)) for c in using_knives], axis=0,
                       keys=using_knives.index)
    n_slice_index = len(slices.index.names) - len(inputs.index.names)
    return slices.reorder_levels(inputs.index.names
                                 + slices.index.names[0:n_slice_index])

def pour_datasets(from_tap, for_inputs, and_additionally=None):
    """..."""
    LOG.info("Get input data from tap: \n%s", for_inputs)
    references = deepcopy(for_inputs)
    if and_additionally:
        LOG.info("And additionally: \n%s", and_additionally)
        references.update({key: {"dataset": ref} for key, ref in and_additionally.items()} or {})
    datasets = pour(from_tap, datasets=references)
    return datasets.apply(lazy_keyword).apply(DataCall)


def datacall(transformation, preserves_shape=True):
    """Apply a transformation, lazily."""
    def transform(dataitem):
        """..."""
        return DataCall(dataitem, transformation, preserves_shape=preserves_shape, cache=False)
    return transform


def control_inputs(of_computation, in_config, using_tap):
    """..."""
    params = parameterize(*describe(of_computation), in_config)
    try:
        randomizations = params["controls"]
    except KeyError:
        LOG.warning("No controls have been configured for the input of %s: \n%s",
                    of_computation, params)
        return None

    controls = load_control(randomizations, lazily=False)
    assert controls, "Cannot be empty. Check your config."

    for_input = filter_datasets(params["input"])
    return pd.concat(
        [pour_datasets(using_tap, for_input, and_additionally=to_tap).apply(datacall(shuffle))
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
    return pd.concat([datasets.apply(datacall(cut, preserves_shape=False)) for _, cut in slicing],
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


def setup_compute_node(c, of_computation, with_inputs, using_configs, at_dirpath,
                       in_mode=None, slicing=None):
    """..."""
    assert not in_mode or in_mode in ("prod", "develop")

    from connsense.apps import APPS
    LOG.info("Configure compute-node %s (%s inputs) to %s slicing %s, with configs \n%s",
             c, len(with_inputs), of_computation, slicing or "none", using_configs)

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
    of_executable = cmd_sbatch(APPS["main"], of_computation, config=slurm_params,
                               at_dirpath=for_compute_node)

    if "submission" not in with_inputs:
        launchscript = at_dirpath / "launchscript.sh"
    else:
        submission = with_inputs.submission.unique()
        assert len(submission) == 1,\
            f"A single compute node's inputs must be submitted together"
        launchscript = at_dirpath / f"launchscript-{submission[0]}.sh"

    run_mode = in_mode or "prod"
    command_lines = ["#!/bin/bash",
                     (f"########################## LAUNCH {computation_type} for chunk {c}"
                      f" of {len(with_inputs)} _inputs."
                     "#######################################"),
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

    setup = {"dirpath": for_compute_node, "sbatch": of_executable,
             "input": inputs_to_read, "output": output_h5}

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


def setup_multinode(process, of_computation, in_config, using_runtime, *,
                    in_mode=None, njobs_to_estimate_load=None):
    """Setup a multinode process.
    """
    from tqdm import tqdm; tqdm.pandas()
    from connsense.develop.topotap import HDFStore

    n_compute_nodes, n_parallel_jobs, o_complexity =\
        prepare_parallelization(of_computation, in_config, using_runtime)

    def prepare_compute_nodes(inputs, at_dirpath, slicing, output_type, unit_weight=None):
        """..."""
        at_dirpath.mkdir(exist_ok=True, parents=False)
        using_configs = configure_multinode(process, of_computation, in_config, at_dirpath)

        if process == setup_compute_node:
            batched = batch_multinode(
                of_computation, inputs, in_config, at_dirpath,
                unit_weight=unit_weight,
                using_parallelization=(n_compute_nodes, n_parallel_jobs, o_complexity),
                njobs_to_estimate_load=njobs_to_estimate_load)

            using_configs["slurm_params"] = (
                configure_slurm(of_computation, in_config, using_runtime)
                .get("sbatch", None))

            LOG.warn("Check batched %s", batched.columns)
            compute_nodes = {
                c: setup_compute_node(c, of_computation, inputs, using_configs, at_dirpath,
                                      in_mode=in_mode, slicing=slicing)
                for c, inputs in batched.groupby("compute_node")}

            return {
                "configs": using_configs,
                "number_compute_nodes": n_compute_nodes,
                "number_total_jobs": n_parallel_jobs,
                "setup": write_multinode_setup(compute_nodes, inputs, at_dirpath)}

        if process == collect_results:
            batched = read_compute_nodes_assignment(at_dirpath)
            _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
            h5_group = output_paths["steps"][computation_type]

            setup = {c: read_setup_compute_node(c, for_quantity=at_dirpath)
                     for c, _ in batched.groupby("compute_node")}
            return collect_results(computation_type, setup, at_dirpath, h5_group, slicing,
                                    output_type=output_type)

        return ValueError(f"Unknown multinode {process}")

    _, to_stage = get_workspace(of_computation, in_config)

    using_configs = configure_multinode(process, of_computation, in_config,
                                        at_dirpath=to_stage)

    computation_type, of_quantity = describe(of_computation)
    params = parameterize(*describe(of_computation), in_config)
    circuit_args = input_circuit_args(of_computation, in_config, load_circuit=False)
    circuit_kwargs = input_circuit_args(of_computation, in_config, load_circuit=True)

    full = generate_inputs(of_computation, in_config, circuit_args=circuit_args, **circuit_kwargs)


    if process == setup_compute_node:
        full_weights = (
            #full.progress_apply(lambda l: estimate_load(to_compute=None)(l())).dropna()
            full.progress_apply(estimate_load(o_complexity)).dropna()
            if not njobs_to_estimate_load
            else multiprocess_load_estimate(o_complexity, full, njobs_to_estimate_load))

        have_zero_weight = np.isclose(full_weights.values, 0.)
        LOG.info("Inputs with zero weight %s: \n%s",
                 have_zero_weight.sum(), full_weights[have_zero_weight])

        full = full[~have_zero_weight]
        full_weights = full_weights[~have_zero_weight]
        max_weight = full_weights.max()
    else:
        max_weight = None

    full_path, full_slicing = ((to_stage, None) if "slicing" not in params else
                               (to_stage/"full", "full"))
    compute_nodes = {"full": prepare_compute_nodes(full, full_path, full_slicing,
                                                   params["output"], max_weight)}

    if "slicing" not in params:
        return compute_nodes

    exclude = ("description", "do-full")
    cfg_slicings = {k: v for k, v in params["slicing"].items() if k not in exclude}
    of_tap = HDFStore(in_config)
    slicings = load_slicing(cfg_slicings, of_tap, lazily=False, **circuit_kwargs)
    for slicing, to_slice in slicings.items():
        slicing_mode = cfg_slicings[slicing].get("compute_mode", "EXECUTE")
        if slicing_mode in ("datacall", "DATACALL"):
            sliced_inputs = generate_slices(of_tap, inputs=full, using_knives=to_slice)
            output_type = params["output"]
        elif slicing_mode in ("execute", "EXECUTE"):
            sliced_inputs = full
            output_type = matrices.type_series_store(params["output"])
        else:
            raise ValueError(f"Unhangled compute mode for slicing: {slicing_mode}")

        compute_nodes[slicing] = prepare_compute_nodes(sliced_inputs, to_stage/slicing,
                                                       slicing, output_type, None)
    return compute_nodes

def collect_results(computation_type, setup, from_dirpath, in_connsense_store, slicing,
                    output_type=None):
    """..."""
    if computation_type == "extract-node-populations":
        assert not slicing, "Does not apply"
        return collect_node_population(setup, from_dirpath, in_connsense_store)

    if computation_type == "extract-edge-populations":
        assert not slicing, "Does not apply"

        return collect_edge_population(setup, from_dirpath, in_connsense_store)

    if computation_type in ("analyze-connectivity", "analyze-composition",
                            "analyze-node-types", "analyze-physiology"):
        return collect_analyze_step(setup, from_dirpath, in_connsense_store, slicing,
                                    output_type)

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
            LOG.info("No output configured for compute node %s\n%s", {pformat(of_compute_node)}, ferr)
            return None
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
            LOG.info("No output configured for compute node %s\n%s", {pformat(of_compute_node)}, ferr)
            return None
            #raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
        return output

    outputs = {c: describe_output(of_compute_node) for c, of_compute_node in setup.items()}
    LOG.info("Edge extraction reported outputs: \n%s", pformat(outputs))

    def collect_adjacencies(of_compute_node, output):
        """..."""
        LOG.info("Collect adjacencies compute-node %s output %s", of_compute_node, output)
        adj = read_toc_plus_payload(output, for_step="extract-edge-populations")
        return write_toc_plus_payload(adj, (connsense_h5, hdf_edge_population), append=True, format="table",
                                      min_itemsize={"values": 100})

    LOG.info("Collect adjacencies")
    for of_compute_node, output in outputs.items():
        collect_adjacencies(of_compute_node, output)

    LOG.info("Adjacencies collected: \n%s", len(outputs))

    return (in_connsense_store, hdf_edge_population)


def collect_analyze_step(setup, from_dirpath, in_connsense_store, slicing,
                         output_type=None):
    """..."""
    try:
        with open(from_dirpath/"description.json", 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the analysis extracted: {from_dirpath}") from ferr

    connsense_h5, group = in_connsense_store
    hdf_group = group + '/' + analysis["name"]
    if slicing:
        hdf_group = hdf_group + "/" + slicing
    output_type = output_type if output_type else analysis["output"]

    def describe_output(of_compute_node):
        """..."""
        try:
            with open(Path(of_compute_node["dirpath"]) / "output.json", 'r') as f:
                output = json.load(f)
        except FileNotFoundError as ferr:
            LOG.info("No output configured fo compute node %s: \n%s", pformat(of_compute_node), ferr)
            return None
            #raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
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
            .collect({c: move(compute_node=c, output=o) for c, o in outputs.items() if o}))


SERIAL_BATCHES = "serial-batches"
PARALLEL_BATCHES = "parallel-batches"

def run_multiprocess(of_computation, in_config, using_runtime, on_compute_node,
                     batching=SERIAL_BATCHES, slicing=None):
    """..."""
    import time
    from connsense.develop.topotap import HDFStore
    on_compute_node = run_cleanup(on_compute_node)

    run_in_progress = on_compute_node.joinpath(INPROGRESS)
    run_in_progress.touch(exist_ok=False)


    computation_type, of_quantity = describe(of_computation)
    parameters, execute, to_store_batch, to_store_one = (
        configure_execution(of_computation, in_config, on_compute_node, slicing=slicing)
    )
    assert to_store_batch or to_store_one
    assert not (to_store_batch and to_store_one)

    in_hdf = "connsense-{}.h5"

    circuit_args = input_circuit_args(of_computation, in_config,
                                      load_circuit=False, load_connectome=False, drop_nulls=False)
    circuit_kwargs = input_circuit_args(of_computation, in_config,
                                        load_circuit=True, load_connectome=False, drop_nulls=False)
    circuit_args_values = tuple(v for v in (circuit_kwargs.get("circuit"),
                                            circuit_kwargs.get("connectome")) if v)

    kwargs = load_kwargs(parameters, HDFStore(in_config), on_compute_node)

    timeout = kwargs.pop("timeout", None)

    inputs = generate_inputs(of_computation, in_config, slicing, circuit_args,
                             **circuit_kwargs)

    if not isinstance(inputs.index, pd.MultiIndex):
        inputs.index = pd.MultiIndex.from_arrays([inputs.index.values,], names=[inputs.index.name])

    collector = (plugins.import_module(parameters["collector"]) if "collector" in parameters
                 else None)

    def collect_batch(results):
        """..."""
        if not collector:
            return results

        _, collect = collector
        return collect(results)

    def execute_one(lazy_subtarget, bowl=None, index=None):
        """..."""
        subtarget_inputs = lazy_subtarget()
        LOG.info("Execute circuit args %s lazy subtarget %s, kwargs %s",
                 circuit_args_values, subtarget_inputs.keys(), kwargs.keys())
        #result = execute(*circuit_args_values, **subtarget_inputs)
        result = execute(circuit_kwargs, subtarget_inputs)
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
        LOG.info(
            "Run %s batch %s of %s inputs args, and circuit %s, \n with kwargs %s slicing %s",
            of_computation,  index, len(of_input), circuit_args_values, pformat(kwargs), slicing)

        def to_subtarget(s):
            """..."""
            r = execute_one(lazy_dataset(s))
            LOG.info("store one lazy subtarget %s result \n%s", s, r)

            if r is None:
                LOG.warning("THERE is no RESULT for subtarget index  %s", index)
                return None

            if r.empty:
                LOG.warning("Result is empty.")
            LOG.info("Result data types\n%s", r.describe())
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

            tocna = toc[toc.isna()]
            LOG.info("TOC elements that were NA for subtarget index %s: \n%s", index, tocna)
            result = to_store_one(in_hdf.format(index), update=toc.dropna())

        if in_bowl is not None:
            in_bowl[index] = result
        return result

    n_compute_nodes,  n_total_jobs, _  = prepare_parallelization(of_computation,
                                                                 in_config, using_runtime)

    batches = load_input_batches(on_compute_node)
    n_batches = batches.batch.max() - batches.batch.min() + 1

    if n_compute_nodes == n_total_jobs:
        results = {}
        for batch, subtargets in batches.groupby("batch"):
            LOG.info("Run Single Node %s process %s / %s batches",
                     on_compute_node, batch, n_batches)
            results[batch] = serial_batch(inputs.loc[subtargets.index], index=batch)
        LOG.info("DONE Single Node connsense run.")
    else:
        if batching == SERIAL_BATCHES:
            manager = Manager()
            bowl = manager.dict()
            processes = []

            for b, subtargets in batches.groupby("batch"):
                LOG.info("Spawn Compute Node %s process %s / %s batches",
                         on_compute_node, b, n_batches)
                p = Process(target=serial_batch,
                            args=(inputs.loc[subtargets.index],),
                            kwargs={"index": b, "in_bowl": bowl})
                p.start()
                processes.append((b, p))

            LOG.info("LAUNCHED %s processes", n_batches)

            if timeout:
                start = time.time()
                in_run = pd.Series([p.is_alive() for _, p in processes], name="running",
                                   index=pd.Index([b for b, _ in processes], name="batch"))

                while (in_run.sum() > 0) and (time.time() - start < timeout):
                    in_run = pd.Series([p.is_alive() for _, p in processes], name="running",
                                       index=pd.Index([b for b, _ in processes],
                                                      name="batch"))
                    LOG.info("RUNNING PROCESSES (timeout at %s)", timeout)
                    LOG.info("at time %s processing still running: %s/%s\n%s",
                             time.time() - start, in_run.sum(), len(processes),
                             batches[in_run.reindex(batches.batch.values)
                                     .fillna(False).values]
                             .droplevel("compute_node").reset_index())

                    if in_run.sum() > 0:
                        time.sleep(60)
                    else:
                        break
                else:
                    LOG.info(("TIMEOUT: terminating %s alive of %s processes:"
                              " computation time %s exceeded."),
                             in_run.sum(), len(processes), timeout)
                    for _,p in processes:
                        p.terminate()

            for _,p in processes:
                p.join()

            LOG.info("Parallel computation %s results %s", of_computation, len(bowl))

            results = {key: value for key, value in bowl.items()}
            LOG.info("Computation %s results %s", of_computation, len(results))

        else:
            assert batching == PARALLEL_BATCHES, "No other is known."
            for batch, subtargets in batches.groupby("batch"):
                LOG.info("Run %s subtargets in parallel batch %s / %s batches.",
                         len(subtargets), batch, len(batches))

                manager = Manager()
                bowl = manager.dict()
                processes = []

                for i, s in enumerate(subtargets):
                    p = Process(target=execute_one,
                                args=(s,), kwargs={"index": i, "bowl": bowl})
                    p.start()
                    process.append(p)
                    LOG.info("LAUNCHED %s process", i)

                for p in process:
                    p.join()
                LOG.info("Parallel computation for batch %s: %s", batch, len(bowl))
                values = pd.Series([v for v in bowl.values()], index=subtargets.index)
                hdf = in_hdf.format(batch)
                of_each_value = lambda: values.apply(lambda v: to_store_one(hdf, result=v))
                results = (to_store_batch(hdf, results=values) if to_store_batch else
                           to_store_one(hdf, update=of_each_value()))

    read_pipeline.write(results, to_json=on_compute_node/"batched_output.json")

    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, hdf_group = output_paths["steps"][computation_type]
    collected = collect_batches(of_computation, results, on_compute_node, hdf_group, slicing,
                                of_output_type=parameters["output"])
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


def input_circuit_args(computation, in_config,
                       load_circuit=True, load_connectome=False, *,
                       drop_nulls=True):
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
    connectome = input_connectome(x, in_circuit=c) if load_connectome else x
    circonn = {"circuit": circuit, "connectome": connectome}
    return {key: value for key, value in circonn.items() if value}


def subtarget_circuit_args(computation, in_config,
                           load_circuit=False, load_connectome=False):
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



def load_kwargs(parameters, to_tap, on_compute_node=None, consider_input=False):
    """..."""
    def load_if_dataset(variable, value):
        """..."""
        if not isinstance(value, Mapping):
            return value

        if "dataset" in value:
            return to_tap.pour_dataset(*value["dataset"], subset=value.get("subset", None))
            #return load_dataset(to_tap, variable, value)
        return value

    kwargs = parameters.get("kwargs", {})
    kwargs.update({var: load_if_dataset(var, value) for var, value in kwargs.items()
                   if var not in COMPKEYS})

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
        path = (Path(workdir)/on_compute_node.relative_to(to_tap._root.parent)
                if on_compute_node else Path(workdir))
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

        if on_compute_node:
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


def get_executable(computation, in_config, on_compute_node=None, slicing=None):
    """..."""
    from connsense.develop.topotap import HDFStore

    computation_type, of_quantity = describe(computation)
    parameters = parameterize(computation_type, of_quantity, in_config)

    executable_type = EXECUTABLE[computation_type.split('-')[0]]
    try:
        executable = parameters[executable_type]
    except KeyError as err:
        raise RuntimeError(f"No {executable_type} defined for {computation_type}") from err

    _, ex = plugins.import_module(executable["source"], executable["method"])

    of_tap = HDFStore(in_config)

    analysis_kwargs = load_kwargs(parameters, of_tap, on_compute_node)
    def execute_one(circuit_kwargs, subtarget_kwargs):
        LOG.info("Execute %s for a single subtarget", ex.__name__)
        return ex(**circuit_kwargs, **subtarget_kwargs, **analysis_kwargs)
#    def execute_one(*circuit_args, **subtarget_kwargs):
#        return ex(*circuit_args, **subtarget_kwargs, **analysis_kwargs)

    if not slicing or slicing == "full":
        return (execute_one, parameters)

    exclude = ("description", "do-full")
    cfg_slicings = {k: v for k, v in parameters["slicing"].items() if k not in exclude}
    if cfg_slicings[slicing].get("compute_mode", "EXECUTE") in ("datacall", "DATACALL"):
        return (execute_one, parameters)

    circuit_kwargs = input_circuit_args(computation, in_config, load_circuit=True)
    knives = load_slicing(cfg_slicings, of_tap, lazily=False, **circuit_kwargs)[slicing]

    def apply_sliced(circuit_kwargs, subtarget_kwargs):
        """..."""
        return (pd.Series([cut(subtarget_kwargs) for cut in knives],
                          index=knives.index)
                .apply(lambda xs: execute_one(circuit_kwargs, xs)))

    parameters["output"] = matrices.type_series_store(parameters["output"])
    return (apply_sliced, parameters)

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
        return extract_connectivity.write_adj(results, to_output=in_hdf, append=True,
                                              format="table", return_config=True)

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
            assert update is None, "No update allowed once result has been presented."
            LOG.info("Write a result to store %s with size %s", p, len(result))
            path = cached_stores[p].write(result)
            LOG.info("Wrote a result to store %s with size %s to path %s", p, len(result), path)
            return path

        cached_stores[p].append_toc(cached_stores[p].prepare_toc(of_paths=update))
        return (at_path, hdf_group)

    return write_hdf


def configure_execution(computation, in_config, on_compute_node, slicing=None):
    """..."""
    computation_type, of_quantity = describe(computation)
    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, in_hdf_group = output_paths["steps"][computation_type]

    execute, parameters = get_executable(computation, in_config, on_compute_node, slicing)

    if computation_type == "extract-node-populations":
        assert not slicing, "Does not apply"
        return (parameters, execute, None,  store_node_properties(of_quantity, on_compute_node,
                                                                  in_hdf_group))

    if computation_type == "extract-edge-populations":
        assert not slicing, "Does not apply"
        return (parameters, execute, store_edge_extraction(of_quantity, on_compute_node,
                                                           in_hdf_group), None)

    return (parameters, execute,
            None, store_matrix_data(of_quantity, parameters, on_compute_node,
                                    in_hdf_group, slicing))


def collect_batches(of_computation, results, on_compute_node, hdf_group, slicing,
                    of_output_type):
    """..."""
    LOG.info("Collect %s results of %s on compute node %s in group %s output type %s",
             len(results), of_computation, on_compute_node, hdf_group, of_output_type)
    computation_type, of_quantity = describe(of_computation)

    if computation_type == "extract-edge-populations":
        assert not slicing, "Does not apply"
        return collect_batched_edge_population(of_quantity, results,
                                               on_compute_node, hdf_group)

    if computation_type == "extract-node-populations":
        assert not slicing, "Does not apply"

    hdf_group = hdf_group+"/"+of_quantity
    if slicing:
        hdf_group = hdf_group+"/"+slicing
    in_connsense_h5 = on_compute_node / "connsense.h5"

    in_store = (matrices
                .get_store(in_connsense_h5, hdf_group, for_matrix_type=of_output_type))
    in_store.collect({batch: matrices.get_store(on_compute_node / batch_connsense_h5,
                                                hdf_group,
                                                for_matrix_type=of_output_type)
                      for batch, (batch_connsense_h5, group) in results.items()})
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
            LOG.info("No tap datasets for control: \n%s", description)
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
    return [shfld for ctrl, cfg in controls.items() for shfld in load_config(ctrl, cfg)]



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


def load_slicing(transformations, using_tap=None, lazily=True, **circuit_args):
    """..."""
    from copy import deepcopy

    LOG.info("Load slicing %s transformations for circuit args: \n%s",
             len(transformations), circuit_args)

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
                    return lambda: algorithm(**circuit_args, **datasets(), **aslice,
                                             **kwargs)
                assert not callable(datasets)
                return algorithm(**datasets, **aslice, **circuit_args, **kwargs)
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
