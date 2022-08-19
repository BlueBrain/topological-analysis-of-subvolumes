

# Within ~connsense.pipeline.parallelization~ we will have ~.parallelize_multinode~,


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
from connsense.pipeline import PARAMKEY, COMPKEYS
from connsense.io import logging, read_config as read_pipeline
from connsense.io.slurm import SlurmConfig
from connsense.io.write_results import read_toc_plus_payload, write_toc_plus_payload
from connsense.pipeline.workspace import find_base
from connsense.pipeline import NotConfiguredError
from connsense.pipeline.store.store import HDFStore
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


class IllegalParallelComputationError(ValueError):
    """..."""


def describe(computation):
    """..."""
    if isinstance(computation, str):
        description = computation.split('/')
        computation_type = description[0]
        quantity = '/'.join(description[1:])
    elif isinstance(computation, (tuple, list)):
        computation_type, quantity = computation
    else:
        raise TypeError(f"copmutation of illegal type {computation}")

    return (computation_type, quantity)


def run_multinode(process_of, computation, in_config, using_runtime, for_control=None, making_subgraphs=None):
    """..."""
    _, to_stage = get_workspace(computation, in_config, for_control, making_subgraphs)

    using_configs = run_multinode_configs(process_of, computation, in_config, for_control, making_subgraphs,
                                          at_dirpath=to_stage)

    computation_type, of_quantity = describe(computation)

    inputs = generate_inputs_of(computation, in_config)
    n_compute_nodes,  n_parallel_jobs = prepare_parallelization(computation, in_config, using_runtime)
    batched = batch_multinode(process_of, inputs, computation, in_config,
                              at_path=to_stage, using_parallelization=(n_compute_nodes, n_parallel_jobs))

    if process_of == setup_compute_node:
        using_configs["slurm_params"] = configure_slurm(computation, in_config, using_runtime).get("sbatch", None)

        compute_nodes = {c: setup_compute_node(c, inputs, (computation_type, of_quantity, to_stage), using_configs)
                         for c, inputs in batched.groupby("compute_node")}
        return {"configs": using_configs,
                "number_compute_nodes": n_compute_nodes, "number_total_jobs": n_parallel_jobs,
                "setup": write_multinode_setup(compute_nodes, inputs,  at_dirpath=to_stage)}

    if process_of == collect_multinode:
        _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
        at_path = output_paths["steps"][computation_type]

        setup = {c: read_setup_compute_node(c, for_quantity=to_stage) for c,_ in batched.groupby("compute_node")}
        return collect_multinode(computation_type, setup, from_dirpath=to_stage, in_connsense_store=at_path)

    raise ValueError(f"Unknown {process_of} multinode")


def run_multinode_configs(process_of, computation, in_config, for_control, making_subgraphs, at_dirpath):
    """..."""
    if process_of == setup_compute_node:
        return write_configs_of(computation, in_config, at_dirpath,
                                with_random_shuffle=for_control, in_the_subtarget=making_subgraphs)

    if process_of == collect_multinode:
        return read_configs_of(computation, in_config, at_dirpath,
                               with_random_shuffle=for_control, in_the_subtarget=making_subgraphs)

    raise ValueError(f"Unknown {process_of} multinode")


def batch_multinode(process_of, inputs, computation, in_config, at_path, using_parallelization):
    """..."""
    n_compute_nodes, n_parallel_jobs = using_parallelization

    if process_of == setup_compute_node:
        LOG.info("Assign batches to %s inputs", len(inputs))
        batches = assign_batches_to(inputs, upto_number=n_parallel_jobs)

        n_batches = batches.max() + 1
        LOG.info("Assign compute nodes to %s batches of %s inputs", len(batches), n_batches)
        compute_nodes = assign_compute_nodes(batches, upto_number=n_compute_nodes)

        assignment = pd.concat([batches, compute_nodes], axis=1)
        assignment_h5, dataset = COMPUTE_NODE_ASSIGNMENT
        assignment.to_hdf(at_path/assignment_h5, key=dataset)
        return assignment

    if process_of == collect_multinode:
        return read_compute_nodes_assignment(at_path)

    raise ValueError(f"Unknown {process_of} multinode")


def setup_compute_node(c, inputs, for_computation, using_configs):
    """..."""
    from connsense.apps import APPS
    LOG.info("Configure chunk %s with %s inputs to compute %s.", c, len(inputs), for_computation)

    computation_type, for_quantity, to_stage = for_computation

    for_compute_node = to_stage / f"compute-node-{c}"
    for_compute_node.mkdir(parents=False, exist_ok=True)
    configs = symlink_pipeline(configs=using_configs, at_dirpath=for_compute_node)

    inputs_to_read = write_compute(inputs, to_hdf=INPUTS, at_dirpath=for_compute_node)
    output_h5 = f"{for_compute_node}/connsense.h5"

    def cmd_sbatch(at_path, executable):
        """..."""
        try:
            slurm_params = using_configs["slurm_params"]
        except KeyError as kerr:
            raise RuntimeError("Missing slurm params") from kerr

        slurm_params.update({"name": computation_type, "executable": executable})
        slurm_config = SlurmConfig(slurm_params)
        return slurm_config.save(to_filepath=at_path/f"{computation_type}.sbatch")

    #of_executable = cmd_sbatch(at_path=for_compute_node, executable=APPS[computation_type])
    of_executable = cmd_sbatch(at_path=for_compute_node, executable=APPS["main"])

    def cmd_configs():
        """..."""
        return "--configure=pipeline.yaml --parallelize=runtime.yaml \\"

    def cmd_options():
        """..."""
        return None

    master_launchscript = to_stage / "launchscript.sh"

    with open(master_launchscript, 'a') as to_launch:

        def write(aline):
            if aline:
                to_launch.write(aline + '\n')

        write("#!/bin/bash")

        write(f"########################## LAUNCH {computation_type} for chunk {c}"
             f" of {len(inputs)} _inputs. #######################################")
        write(f"pushd {for_compute_node}")

        sbatch = f"sbatch {of_executable.name} run {computation_type} {for_quantity} \\"
        write(sbatch)
        write(cmd_configs())
        write(cmd_options())
        write(f"--input={inputs_to_read} \\")
        write(f"--output={output_h5}")

        write("popd")

    setup = {"dirpath": for_compute_node, "sbatch": of_executable, "input": inputs_to_read, "output": output_h5}

    return read_pipeline.write(setup, to_json=for_compute_node/"setup.json")


def write_multinode_setup(compute_nodes, inputs, at_dirpath):
    """..."""
    inputs_h5, dataset = INPUTS
    #inputs.to_hdf(at_dirpath/inputs_h5, key=dataset)

    return read_pipeline.write({"compute_nodes": compute_nodes, "inputs": at_dirpath/inputs_h5},
                                to_json=at_dirpath/"setup.json")


def collect_multinode(computation_type, setup, from_dirpath, in_connsense_store):
    """..."""
    #if not in_connsense_store.exists():
        #raise RuntimeError(f"NOTFOUND {in_connsense_h5_at_basedir}\n HDF5 for connsense in base dir must exist")

    if computation_type == "extract-node-populations":
        return collect_node_population(setup, from_dirpath, in_connsense_store)

    if computation_type == "extract-edge-populations":
        return collect_edge_population(setup, from_dirpath, in_connsense_store)

    if computation_type in ("analyze-connectivity", "analyze-node-types", "analyze-physiology", "sample-edge-populations"):
        return collect_analyze_step(setup, from_dirpath, in_connsense_store)

    raise NotImplementedError(f"INPROGRESS: {computation_type}")


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
        return write_toc_plus_payload(adj, (connsense_h5, hdf_edge_population), append=True, format="table")
        #return write_toc_plus_payload(adj, (in_connsense_store, hdf_edge_population), append=True, format="table")

    LOG.info("Collect adjacencies")
    for of_compute_node, output in outputs.items():
        collect_adjacencies(of_compute_node, output)

    LOG.info("Adjacencies collected: \n%s", len(outputs))

    return (in_connsense_store, hdf_edge_population)


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



def collect_node_population_single_dataframe(setup, from_dirpath, in_connsense_store):
    """...Deprecated"""
    from connsense.io.write_results import read as read_compute_node, write as write_compute_node
    LOG.info("Collect node population at %s using setup \n%s", from_dirpath, setup)

    try:
        with open(from_dirpath/"description.json", 'r') as f:
            population = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the population extracted: {from_dirpath}") from ferr

    def describe_output(of_compute_node):
        """..."""
        try:
            with open(Path(of_compute_node["dirpath"]) / "output.json", 'r') as f:
                output = json.load(f)
        except FileNotFoundError as ferr:
            raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
        return output

    #p = population["name"]
    #hdf_group = f"nodes/populations/{p}"
    connsense_h5, group = in_connsense_store

    def move(compute_node, from_path):
        """..."""
        LOG.info("Write batch %s read from %s", compute_node, from_path)
        compute_node_result = describe_output(from_path)
        result = read_compute_node(compute_node_result, "extract-node-populations")
        return write_compute_node(result, to_path=(connsense_h5, group+"/"+population["name"]),
                                  append=True, format="table")

    for compute_node, hdf_path in setup.items():
        move(compute_node, hdf_path)

    return (in_connsense_store, group+"/"+population["name"])


def collect_analyze_step(setup, from_dirpath, in_connsense_store):
    """..."""
    try:
        with open(from_dirpath/"description.json", 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the analysis extracted: {from_dirpath}") from ferr

    #a = analysis["name"]
    #hdf_analysis = f"analyses/connectivity/{a}"
    connsense_h5, group = in_connsense_store
    hdf_analysis = group + '/' + analysis["name"]
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

    def in_store(at_path, hdf_group=None):
        """..."""
        return matrices.get_store(at_path, hdf_group or hdf_analysis, output_type)

    def move(compute_node, output):
        """..."""
        LOG.info("Get analysis store for compute-node %s output %s", compute_node, output)
        h5, g = output
        return in_store(at_path=h5, hdf_group=g)

    return in_store(connsense_h5).collect({c: move(compute_node=c, output=o) for c, o in outputs.items()})



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


def get_workspace(for_computation, in_config, for_control=None, making_subgraphs=None, in_mode='r'):
    """..."""
    m = {'r': "test", 'w': "prod", 'a': "develop"}[in_mode]
    computation_type, of_quantity = describe(for_computation)
    rundir = workspace.get_rundir(in_config, computation_type, of_quantity, making_subgraphs, for_control, in_mode=m)
    basedir = workspace.find_base(rundir)
    return (basedir, rundir)


def write_configs_of(computation, in_config, at_dirpath, with_random_shuffle=None, in_the_subtarget=None):
    """..."""
    LOG.info("Write configs of %s at %s", computation, at_dirpath)
    return {"base": write_pipeline_base_configs(in_config, at_dirpath),
            "control": write_pipeline_control(with_random_shuffle, at_dirpath),
            "subgraphs": write_pipeline_subgraphs(in_the_subtarget, at_dirpath),
            "description": write_description(computation, in_config, at_dirpath)}

def read_configs_of(computation, in_config, at_dirpath, with_random_shuffle=None, in_the_subtarget=None):
    """..."""
    LOG.info("Read configs of %s at %s", computation, at_dirpath)
    return {"base": read_pipeline_base_configs(computation, in_config, at_dirpath),
            "control": read_pipeline_control(with_random_shuffle, at_dirpath),
            "subgraphs": read_pipeline_subgraphs(in_the_subtarget, at_dirpath)}

def write_pipeline_base_configs(in_config, at_dirpath): #pylint: disable=unused-argument
    """..."""
    basedir = find_base(rundir=at_dirpath)
    LOG.info("CHECK BASE CONFIGS AT %s", basedir)
    def write_config(c):
        def write_format(f):
            filename = f"{c}.{f}"
            base_config = basedir / filename
            if base_config.exists():
                run_config = at_dirpath / filename
                _remove_link(run_config)
                run_config.symlink_to(base_config)
                return  run_config
            LOG.info("Not found config %s", base_config)
            return None
        return {f: write_format(f) for f in ["json", "yaml"] if f}
    return {c: write_config(c) for c in ["pipeline", "runtime", "config", "parallel"]}


def read_pipeline_base_configs(of_computation, in_config, at_dirpath): #pylint: disable=unused-argument
    """..."""
    LOG.info("Look for basedir of %s", at_dirpath)
    basedir = find_base(rundir=at_dirpath)
    LOG.info("CHECK BASE CONFIGS AT %s", basedir)
    def read_config(c):
        def read_format(f):
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



def write_pipeline_control(algorithm, at_dirpath): #pylint: disable=unused-argument
    """..."""
    if not algorithm: return None

    if not at_dirpath.name.startswith("compute-node-"):
        control_json = at_dirpath / "control.json"
        description = deepcopy(algorithm.description)
        description["name"] = algorithm.name
        return read_pipeline.write(description, to_json=control_json)

    control_config = at_dirpath.parent / "control.json"
    if not control_config.exits():
        raise RuntimeError(f"InvalicComputeNode: {at_dirpath}. The directory's parent is missing a control config.")
    _remove_link(control_config)
    control_config.symlink_to(at_dirpath.parent / "control.json")
    return control_config

def read_pipeline_control(algorithm, at_dirpath): #pylint: disable=unused-argument
    """..."""
    if not algorithm: return None
    raise NotImplementedError("INRPOGRESS")


def write_pipeline_subgraphs(in_the_subtarget, at_dirpath): #pylint: disable=unused-argument
    """..."""
    return None


def read_pipeline_subgraphs(algorithm, at_dirpath): #pylint: disable=unused-argument
    """..."""
    if not algorithm: return None
    raise NotImplementedError("INRPOGRESS")

def write_description(computation, in_config, at_dirpath):
    """..."""
    computation_type, of_quantity = describe(computation)
    configured = parameterize(computation_type, of_quantity, in_config)
    configured["name"] = of_quantity
    return read_pipeline.write(configured, to_json=at_dirpath / "description.json")

def symlink_pipeline(configs, at_dirpath):
    """..."""
    to_base = symlink_pipeline_base(configs["base"], at_dirpath)
    to_control = symlink_pipeline_control(configs["control"], at_dirpath)
    to_subgraphs = symlink_pipeline_subgraphs(configs["subgraphs"], at_dirpath)
    return {"base": to_base, "control": to_control, "subgraphs": to_subgraphs}


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


def symlink_pipeline_control(to_config, at_dirpath):
    """..."""
    return create_symlink(at_dirpath)(to_config) if to_config else None


def symlink_pipeline_subgraphs(to_config, at_dirpath):
    """..."""
    return create_symlink(at_dirpath)(to_config) if to_config else None



def input_units(computation, to_tap):
    """..."""
    parameters = parameterize(*describe(computation), to_tap._config)
    index_vars = parameters.get("index", parameters["input"])
    index = pd.MultiIndex.from_product([to_tap.subset_index(var, values) for var, values in index_vars.items()])
    return index
    return index.to_frame().reset_index(drop=True)


def filter_input_datasets(described):
    """..."""
    return {var: val for var, val in described.items() if (var not in ("circuit", "connectome")
                                                           and isinstance(val, Mapping) and "dataset" in val)}

def generate_inputs_of(computation, in_config, on_compute_node=None, by_subtarget=False):
    """..."""
    LOG.info("Generate inputs for %s %s %s", computation,
             "by subtarget" if by_subtarget else "", on_compute_node if on_compute_node else "")

    computation_type, of_quantity = describe(computation)
    described = parameterize(computation_type, of_quantity, in_config)

    tap = HDFStore(in_config)

    inputs_h5, dataset = INPUTS
    input_batches = pd.read_hdf(on_compute_node/inputs_h5, dataset) if on_compute_node else None

    variables = filter_input_datasets(described["input"])

    unit_computations = input_units(computation, tap)
    LOG.info("There will be %s unit computations with index %s", len(unit_computations), unit_computations.names)
    unit_slices = [unit_computations[s:s+1] for s in range(0, len(unit_computations))]

    #input_datasets = pd.DataFrame({var: tap.pour_dataset(var, values["dataset"]).loc[unit_computations]
                                   #for var, values in variables.items()})
    dataflow = pour(tap, variables)
    input_datasets = pd.Series([dataflow(s) for s in unit_slices], index=unit_computations)
    return (lambda s: input_datasets.loc[s]) if by_subtarget else input_datasets


    #input_datasets = unit_computations.apply(dataflow, axis=1) #causes stop iteration for some cases!
    #input_datasets = pd.Series([dataflow(row) for i, row in unit_computations.iterrows()],
                               #index=pd.MultiIndex.from_frame(unit_computations))

    return get_subtarget(input_datasets) if by_subtarget else input_datasets



def pour(tap, variables):
    """.."""
    #input_datasets = {var: tap.pour_dataset(var, vals["dataset"]) for var, vals in variables.items()}

    def unpack(value):
        """..."""
        try:
            get = value.get_value
        except AttributeError:
            return value
        return get()

    def group_properties(var):
        """..."""
        properties = variables[var].get("properties", None)

        def apply(subtarget):
            """..."""
            if not properties:
                return subtarget

            return lambda index: subtarget[properties].loc[index]

        return apply

    def load_dataset(var, values):
        """..."""
        dataset = tap.pour_dataset(var, values["dataset"]).apply(unpack)
        if not "reindex" in values:
            return dataset

        original = dataset.apply(lambda subtarget: tap.reindex(subtarget, values["reindex"]))
        return pd.concat(original.values, axis=0, keys=original.index.values, names=original.index.names)

    input_datasets = {var: load_dataset(var, values).apply(group_properties(var)) for var, values in variables.items()}

    def loc(subtarget):
        """..."""
        return {variable: values.loc[subtarget] for variable, values in input_datasets.items()}

    def loc_0(subtarget):
        """..."""
        LOG.info("Locate subtarget \n%s ", subtarget)

        def get_dataset(var, value):
            """..."""
            LOG.info("To pour on \n%s\n, get dataset %s, %s", subtarget, var, value)
            #dataset = tap.pour_subtarget(value["dataset"], subset=lookup)
            dataset = tap.pour_dataset(var, value["dataset"])
            try:
                value = dataset.get_value
            except AttributeError:
                return dataset
            return value()

        def get_transformation(value):
            """..."""
            return {k: v for k, v in value.items() if k != "dataset"}

        return pd.Series({var: vals.loc[subtarget] for var, vals in input_datasets.items()})

        return pd.Series({var: evaluate(get_transformation(value), get_dataset(var, value))
                          for var, value in variables.items()})

    def evaluate(transformation, of_dataset):
        """..."""
        transform = resolve(transformation)
        return transform(of_dataset)

    def resolve(transformation):
        """..."""
        if not transformation:
            return lambda x: x

        try:
            _, transform = plugins.import_module(transformation)
        except plugins.ImportFailure:
            pass
        else:
            return transform

        to_filter = get_filter(transformation)
        to_type = get_properties(transformation)

        return lambda dataset: to_type(to_filter(dataset))

    return lazily(to_evaluate=loc)


def get_filter(transformation):
    """..."""
    if "filter" not in transformation:
        return lambda dataset: dataset

    def apply(dataset):
        """..."""
        raise NotImplementedError

    return apply


def get_properties(transformation):
    """..."""
    g = transformation.get("properties", None)

    if isinstance(g, (list, tuple)) and len(g) == 1:
        g = g[0]

    def apply(dataset):
        """..."""
        def to_node(ids):
            """..."""
            return dataset[g].loc[ids].reset_index(drop=True) if g is not None else ids
        return to_node
    return apply


    return lazily(loc)


def lazily(to_evaluate):
    """..."""
    LOG.info("Evaluate %s lazily", to_evaluate.__name__)
    return lambda subtarget: lambda: to_evaluate(subtarget)


def parameterize(computation_type, of_quantity, in_config):
    """..."""
    """..."""
    paramkey = PARAMKEY[computation_type]

    if not computation_type in in_config["parameters"]:
        raise RuntimeError(f"Unknown {computation_type}")

    configured = in_config["parameters"][computation_type][paramkey]
    if of_quantity not in configured:
        try:
            modeltype, component = of_quantity.split('/')
        except ValueError:
            raise RuntimeError(f"Unknown {paramkey} {of_quantity} for {computation_type}")
        configured_quantity =  configured[modeltype][component]

    else:
        configured_quantity = configured[of_quantity]

    return deepcopy(configured_quantity)

    if computation_type != "define-subtargets":
        if of_quantity not in in_config["parameters"][computation_type][paramkey]:
            raise RuntimeError(f"Unknown {paramkey[:-1]} {of_quantity} for {computation_type}")
        return deepcopy(in_config["parameters"][computation_type][paramkey][of_quantity])

    return deepcopy(in_config["parameters"]["define-subtargets"])

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
        return (1, 1)

    compute_nodes = p["number-compute-nodes"]
    tasks = p["number-tasks-per-node"]
    return (compute_nodes, compute_nodes * tasks)


def read_runtime_config(for_parallelization, of_pipeline=None, return_path=False):
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

            for c in decompose_quantity(q):
                q_cfg[c] = configure_component(c)

            return q_cfg

        return {q: configure_quantity(q) for q in quantities_to_configure if q != "description"}

    runtime_pipeline = {c: configure_slurm_for(computation_type=c) for c in of_pipeline["parameters"]}
    config = {"version": config["version"], "date": config["date"], "pipeline": runtime_pipeline}
    return (path, config) if return_path else config


def prepare_parallelization(computation, in_config, using_runtime):
    """.."""
    computation_type, quantity = describe(computation)
    from_runtime = (read_runtime_config(for_parallelization=using_runtime, of_pipeline=in_config)
                    if not isinstance(using_runtime, Mapping) else using_runtime)
    LOG.info("prepare parallelization %s using runtime \n%s", computation, pformat(from_runtime))
    configured = from_runtime["pipeline"].get(computation_type, {})
    LOG.info("\t Configured \n%s", configured)
    return read_njobs(to_parallelize=configured, computation_of=quantity)


def assign_batches_to(inputs, upto_number, return_load=False):
    """..."""
    def estimate_load(input_data): #pylint: disable=unused-argument
        """Needs improvement.."""
        if callable(input_data):
            return estimate_load(input_data())

        if isinstance(input_data, Mapping):
            first = next(v for v in input_data.values())
            return estimate_load(first)
        try:
            shape = input_data.shape
        except AttributeError:
            pass
        else:
            return np.prod(shape)

        try:
            return len(input_data)
        except TypeError:
            pass

        return 1.

    if isinstance(inputs, pd.Series):
        weights = inputs.apply(estimate_load).sort_values(ascending=True).rename("estimated_load")
    elif isinstance(inputs, pd.DataFrame):
        weights = inputs.apply(estimate_load, axis=1).sort_values(ascending=True).rename("estimated_load")
    else:
        raise TypeError(f"Unhandled type of input: {inputs}")

    computational_load = (np.cumsum(weights) / weights.sum()).rename("estimated_load")
    n = np.minimum(upto_number, len(inputs))
    batches = (n * (computational_load - computational_load.min())).apply(int).rename("batch")

    LOG.info("Load balanced batches for %s inputs: \n %s", len(inputs), batches.value_counts())
    return batches if not return_load else pd.concat([batches, weights/weights.sum()], axis=1)
    #return batches.loc[inputs.index]


def assign_compute_nodes(batches, upto_number):
    """..."""
    LOG.info("Assign compute nodes to batches \n%s", batches)
    _, dataset = COMPUTE_NODE_ASSIGNMENT

    assignment = pd.Series(np.linspace(0, upto_number - 1.e-6, batches.max() + 1, dtype=int)[batches.values],
                           name=dataset, index=batches.index)
    return assignment


def read_compute_nodes_assignment(at_dirpath):
    """..."""
    assignment_h5, dataset = COMPUTE_NODE_ASSIGNMENT

    if not (at_dirpath/assignment_h5).exists():
        raise RuntimeError(f"No compute node assignment saved at {at_dirpath}")

    return pd.read_hdf(at_dirpath / assignment_h5, key=dataset)


def write_compute(batches, to_hdf, at_dirpath):
    """..."""
    batches_h5, and_hdf_group = to_hdf
    batches.to_hdf(at_dirpath / batches_h5, key=and_hdf_group, format="fixed", mode='w')
    return at_dirpath / batches_h5



def load_kwargs(parameters, to_tap):
    """..."""
    def load(value):
        if not isinstance(value, Mapping) or "dataset" not in value:
            return value
        return to_tap.read_dataset(value["dataset"])

    kwargs = parameters.get("kwargs", {})
    kwargs.update({var: load(value) for var, value in parameters.items() if var not in COMPKEYS})
    kwargs.update({var: value for var, value in parameters.get("input", {}).items()
                   if var not in ("circuit", "connectome") and (
                           not isinstance(value, Mapping) or "dataset" not in value)})
    return kwargs


def run_multiprocess(of_computation, in_config, using_runtime, on_compute_node, inputs=None):
    """..."""
    execute, to_store_batch, to_store_one = configure_execution(of_computation, in_config, on_compute_node)

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

    kwargs = load_kwargs(parameters, to_tap=HDFStore(in_config))

    subset_input = generate_inputs_of(of_computation, in_config, on_compute_node, by_subtarget=True)

    collector = plugins.import_module(parameters["collector"]) if "collector" in parameters else None

    def collect_batch(results):
        """..."""
        if not collector:
            return results

        _, collect = collector
        return collect(results)

    def execute_one(lazy_subtarget):
        """..."""
        def unpack(value):
            if isinstance(value, pd.Series):
                assert len(value) == 1
                return value.iloc[0]
            return value

        return execute(*circuit_args_values, **{var: unpack(value) for var, value in lazy_subtarget().items()},
                       **kwargs)

    def run_batch(of_input, *, index, in_bowl):
        """..."""
        LOG.info("Run %s batch %s of %s inputs args, and circuit %s, \n with kwargs %s ", of_computation,
                 index, len(of_input), circuit_args_values, pformat(kwargs))

        def to_subtarget(s):
            """..."""
            return to_store_one(in_hdf.format(index), result=execute_one(lazy_subtarget=s))

        if to_store_batch:
            results = of_input.apply(execute_one)
            in_bowl[index] = to_store_batch(in_hdf.format(index), results=collect_batch(results))
            #framed = pd.concat([results], axis=0, keys=connsense_index.values, names=connsense_index.names)
            #in_bowl[index] = to_store_batch(in_hdf.format(index), results=collect_batch(framed))
        else:
            in_bowl[index] = to_store_one(in_hdf.format(index), update=of_input.apply(to_subtarget))

        return in_bowl[index]

    manager = Manager()
    bowl = manager.dict()
    processes = []

    #n_compute_nodes,  n_jobs = prepare_parallelization(of_computation, in_config, using_runtime)
    #batches = load_input_batches(on_compute_node, inputs, n_parallel_tasks=int(n_jobs/n_compute_nodes))
    batches = load_input_batches(on_compute_node)
    n_batches = batches.batch.max() - batches.batch.min() + 1

    for batch, subtargets in batches.groupby("batch"):
        LOG.info("Spawn compute node %s process %s / %s batches", on_compute_node, batch, n_batches)
        p = Process(target=run_batch,
                    args=(subset_input(subtargets.index),), kwargs={"index": batch, "in_bowl": bowl})
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
    circuit = sbtcfg.input_circuit[labeled]
    circuit.variant = labeled
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



def get_executable(computation_type, parameters):
    """..."""
    executable_type = EXECUTABLE[computation_type.split('-')[0]]

    try:
        executable = parameters[executable_type]
    except KeyError as err:
        raise RuntimeError(f"No {executable_type} defined for {computation_type}") from err

    _, execute = plugins.import_module(executable["source"], executable["method"])

    return execute


def configure_execution(computation, in_config, on_compute_node):
    """..."""
    computation_type, of_quantity = describe(computation)
    parameters = parameterize(computation_type, of_quantity, in_config)
    _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
    _, at_path = output_paths["steps"][computation_type]

    execute = get_executable(computation_type, parameters)
    if computation_type == "extract-node-populations":
        return (execute, None,  store_node_properties(of_quantity, on_compute_node, at_path))
        #return (execute, store_node_properties(of_quantity, on_compute_node, at_path), None)

    if computation_type == "extract-edge-populations":
        return (execute, store_edge_extraction(of_quantity, on_compute_node, at_path), None)

    return (execute, None, store_matrix_data(of_quantity, parameters, on_compute_node, at_path))


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


def store_matrix_data(of_quantity, parameters, on_compute_node, in_hdf_group):
    """..."""
    LOG.info("Store matrix data for %s", parameters)
    of_output = parameters["output"]
    hdf_quantity = f"{in_hdf_group}/{of_quantity}"

    cached_stores = {}

    def write_hdf(at_path, *, result=None, update=None):
        """..."""
        assert at_path
        assert not(result is None and update is None)
        assert result is not None or update is not None

        p = on_compute_node/at_path
        if p not in cached_stores:
            cached_stores[p] = matrices.get_store(p, hdf_quantity, for_matrix_type=of_output)

        if result is not None:
            return cached_stores[p].write(result)

        cached_stores[p].append_toc(cached_stores[p].prepare_toc(of_paths=update))
        return (at_path, hdf_quantity)

    return write_hdf


def collect_batches(of_computation, results, on_compute_node, hdf_group, of_output_type):
    """..."""
    computation_type, of_quantity = describe(of_computation)


    #if computation_type == "extract-node-populations":
        #return collect_batched_node_population(of_quantity, results, on_compute_node, hdf_group)

    if computation_type == "extract-edge-populations":
        return collect_batched_edge_population(of_quantity, results, on_compute_node, hdf_group)

    hdf_quantity = hdf_group+"/"+of_quantity
    in_connsense_h5 = on_compute_node / "connsense.h5"
    in_store = matrices.get_store(in_connsense_h5, hdf_quantity, for_matrix_type=of_output_type)

    batched = results.items()
    in_store.collect({batch: matrices.get_store(connsense_h5, hdf_quantity, for_matrix_type=of_output_type)
                      for batch, (connsense_h5, group) in batched})
    return (in_connsense_h5, hdf_quantity)

def collect_batched_edge_population(p, results, on_compute_node, hdf_group):
    """..."""

    in_connsense_h5 = on_compute_node / "connsense.h5"

    hdf_edge_population = (in_connsense_h5, hdf_group+'/'+p)

    def move(batch, output):
        """.."""
        LOG.info("collect batch %s of adjacencies at %s output %s ", batch, on_compute_node, output)
        adjmats = read_toc_plus_payload(output, for_step="extract-edge-populations")
        return write_toc_plus_payload(adjmats, hdf_edge_population, append=True, format="table")

    LOG.info("collect batched extraction of edges at compute node %s", on_compute_node)
    for batch, output in results.items():
        move(batch, output)

    LOG.info("DONE collecting %s", results)
    return hdf_edge_population


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
