

# and ~.parallelize_multiprocess~ to multiprocess a sinlge compute-node.


from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
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
from connsense.pipeline.pipeline import PARAMKEY
from connsense.io import logging, read_config as read_pipeline
from connsense.io.slurm import SlurmConfig
from connsense.io.write_results import read_toc_plus_payload, write_toc_plus_payload
from connsense.pipeline.workspace import find_base
from connsense.define_subtargets.config import SubtargetsConfig
from connsense.analyze_connectivity import check_paths, matrices
from connsense.analyze_connectivity.analysis import SingleMethodAnalysisFromSource
from connsense.apps import APPS

# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, too-many-locals, comparison-with-callable, too-many-arguments, invalid-name, unspecified-encoding, unnecessary-lambda-assignment

LOG = logging.get_logger("connsense pipeline")


def _remove_link(path):
    try:
        return path.unlink()
    except FileNotFoundError:
        pass
    return None


BATCH_SUBTARGETS = ("subtargets.h5", "batch")
COMPUTE_NODE_SUBTARGETS = ("inputs.h5", "subtargets")
INPUTS = ("inputs.h5", "subtargets")
COMPUTE_NODE_ASSIGNMENT = ("subtargets.h5", "compute_node")



def run_multiprocess(of_computation, in_config, using_runtime, on_compute_node, inputs):
    """..."""
    execute, to_store_batch, to_store_one = configure_execution(of_computation, in_config, on_compute_node)

    assert to_store_batch or to_store_one
    assert not (to_store_batch and to_store_one)

    computation_type, of_quantity = describe(of_computation)

    parameters = parameterize(computation_type, of_quantity, in_config)
    computation_inputs = parameters["input"]

    in_hdf = "connsense-{}.h5"

    circuit_args = get_circuit_args_for(computation_inputs, in_config)
    kwargs = {key: value for key, value in parameters.items() if key not in ("description", "circuit", "connectome",
                                                                             "input", "extractor", "computation",
                                                                             "output")}

    subset_input = generate_inputs_of(of_computation, in_config, on_compute_node, by_subtarget=True)

    def run_batch(of_input, *, index, in_bowl):
        """..."""
        LOG.info("Compute batch %s of %s inputs args, and circuit %s, \n with kwargs %s ",
                 index, len(of_input), circuit_args, pformat(kwargs))

        if to_store_batch:
            result = execute(*circuit_args, of_input, **kwargs)
            in_bowl[index] = to_store_batch(in_hdf.format(index), result)
            return result

        def to_subtarget(s):
            return to_store_one(in_hdf.format(index), result=execute(*circuit_args, of_input, **kwargs))

        in_bowl[index] = to_store_one(update=of_input.apply(to_subtarget))
        return result

    manager = Manager()
    bowl = manager.dict()
    processes = []

    batches = load_inputs_batches(on_compute_node, inputs)
    n_batches = batches.batch.max() - batches.batch.min() + 1

    for batch, subtargets in batches.groupby("batch"):
        LOG.info("Spawn compute node %s process %s / %s batches", on_compute_node, batch, n_batches)
        p = Process(target=run_batch, args=(subset_input(subtargets),), kwargs={"index": batch, "in_bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED %s processes", n_batches)

    for p in processes:
        p.join()

    results = {key: value for key, value in bowl.items()}
    LOG.info("Parallel computation %s results %s", of_computation, len(results))

    read_pipeline.write(results, to_json=on_compute_node/"batched_output.h5")

    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, hdf_group = output_paths["steps"][computation_type]
    of_output_type = parameters["output"]

    collected = collect_batches(of_computation, results, on_compute_node, hdf_group, of_output_type)
    read_pipeline.write(collected, to_json=on_compute_node/"output.json")
    return collected


def input_circuit(labeled, in_config):
    """..."""
    if not labeled:
        return None
    sbtcfg = SubtargetsConfig(in_config)
    return sbtcfg.input_circuit[labeled]

def get_circuit_args_for(computation_inputs, in_config):
    """..."""
    c = computation_inputs.get("circuit", None)
    circuit = input_circuit(c, in_config)
    connectome = computation_inputs.get("connectome", None)
    return tuple(x for x in (circuit, connectome) if x)


def load_input_batches(on_compute_node, inputs=None):
    """..."""
    store_h5, dataset = COMPUTE_NODE_SUBTARGETS

    assert inputs is None or inputs == on_compute_node / store_h5, (
        "inputs dont seem to be what was configured\n"
        f"Expected {inputs} to be {on_compute_node / store_h5} if setup by run_multinode(...)")

    return pd.read_hdf(on_compute_node/store_h5, key=dataset)


def configure_execution(computation, in_config, on_compute_node):
    """..."""
    computation_type, of_quantity = describe(computation)

    parameters = parameterize(computation_type, of_quantity, in_config)

    executable_type = EXECUTABLE[computation_type.split('-')[0]]

    try:
        executable = parameters[executable_type]
    except KeyError as err:
        raise RuntimeError(f"No {executable_type} defined for {computation}") from err

    _, execute = plugins.import_module(executable["source"], executable["method"])

    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, at_path = output_paths["steps"][computation_type]

    if computation_type == "extract-node-populations":
        return (execute, store_node_properties(of_quantity, on_compute_node, at_path), None)

    if computation_type == "extract-edge-populations":
        return (execute, store_edge_extraction(of_quantity, on_compute_node, at_path), None)

    return (execute, None, store_matrix_data(computation_type, of_quantity, parameters, on_compute_node, at_path))


def store_node_properties(of_population, on_compute_node, in_hdf_group):
    """..."""
    def write_batch(connsense_h5, results):
        """..."""
        in_hdf = (on_compute_node/connsense_h5, in_hdf_group)
        LOG.info("Write %s batch results to %s ", len(results), in_hdf)
        return extract_nodes.write(results, of_population, in_hdf)

    return write_batch


def store_edge_extraction(of_population, on_compute_node, in_hdf_group):
    """..."""
    def write_batch(connsense_h5, results):
        """..."""
        in_hdf = (on_compute_node/connsense_h5, f"{in_hdf_group}/{of_population}")
        LOG.info("Write %s batch results to %s", len(results), in_hdf)
        return extract_connectivity.write(results, to_output=in_hdf,  append=True, format="table",
                                          return_config=True)

    return write_batch


def store_matrix_data(computation_type, of_quantity, parameters, on_compute_node, in_hdf_group):
    """..."""

    def write_hdf(at_path, *, result=None, update=None):
        """..."""
        assert at_path
        assert not(result is None and update is None)
        assert result is not None or update is not None
        of_output = parameters["computation"]["output"]
        store = matrices.get_store(at_path, in_hdf_group/of_quantity, for_matrix_type=of_output)

        if result:
            return store.write(result)

        store.append(store.prepare_toc(of_paths=update))
        return (at_path, in_hdf_group)

    return write_hdf

def collect_batches(of_computation, results, on_compute_node, hdf_group, of_output_type):
    """..."""
    computation_type, of_quantity = describe(of_computation)


    if computation_type == "extract-node-populations":
        return collect_batched_node_population(of_quantity, results, on_compute_node, hdf_group)

    if computation_type == "extract-edge-populations":
        return collect_batched_edge_population(of_quantity, results, on_compute_node, hdf_group)

    in_connsense_h5 = on_compute_node / "connsense.h5"
    in_store = matrices.get_store(in_connsense_h5, hdf_group, for_matrix_type=of_output_type)

    batched = results.items()
    return in_store.collect({batch: matrices.get_store(connsense_h5, hdf_group, for_matrix_type=of_output_type)
                             for batch, (connsense_h5, group) in batched})

def collect_batched_edge_population(p, results, on_compute_node, hdf_group):
    """..."""

    in_connsense_h5 = on_compute_node / "connsense.h5"

    hdf_edges = hdf_group+'/'+p

    def collect_batch(b, output):
        """.."""
        try:
            from_connsense_h5_and_group = output["adj"]
        except KeyError as err:
            raise RuntimeError(f"No adjacencies registered in compute node batch {b} of_compute_node output {output}")

        LOG.info("collect batch %s of adjacencies at %s ", b, on_compute_node)

        batch_adj = read_toc_plus_payload(from_connsense_h5_and_group, for_step="extract-edge-populations")
        adj = write_toc_plus_payload(batch_adj, (in_connsense_h5, hdf_edges +'/adj'), append=True, format="table",
                                     min_itemsize=64)

        get_store = matrices.get_store

        if "props" in output:
            in_prop_store = get_store(in_connsense_h5, hdf_edges +'/props', "pandas.DataFrame",
                                       in_mode='a')
            batch_props, group = output["props"]
            props = in_prop_store.collect({b: get_store(batch_props, group, for_matrix_type="pandas.DataFrame",
                                                        in_mode='a')})
        else:
            props = pd.Series()

        return {"adj": adj, "props": props}

    LOG.info("collect batched extraction of edges at compute node %s", on_compute_node)
    collected = {b: collect_batch(b, output=o) for b, o in results.items()}
    LOG.info("DONE collecting %s", collected)

    return {"adj": (in_connsense_h5, hdf_edges+"/adj"), "props": (in_connsense_h5, hdf_edges+"/props")}


def collect_batched_node_population(p, results, on_compute_node, hdf_group):
    """..."""
    from connsense.io.write_results import read as read_batch, write as write_batch

    LOG.info("Collect batched node populations of %s %s results on compute-node %s to %s", p,
             len(results), on_compute_node, hdf_group)

    in_connsense_h5 = on_compute_node / "connsense.h5"
    in_hdf = (in_connsense_h5, hdf_group+"/"+p)

    def move(batch, from_path):
        """..."""
        LOG.info("Write batch %s read from %s", batch, from_path)
        result = read_batch(from_path, "extract-node-populations")
        return write_batch(result, to_path=in_hdf, append=True, format="table", min_itemsize={"subtarget": 64})

    for batch, hdf_path in results.items():
        move(batch, hdf_path)

    return in_hdf
