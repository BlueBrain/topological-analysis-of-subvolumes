# Putting it together
# We can now list the code that can configure a multinode computation.
# which we do to keep the output Python code clean.


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
    n_compute_nodes, n_jobs = prepare_parallelization(computation, in_config, using_runtime)

    computation_type, _ = describe(computation)

    inputs = generate_inputs_of(computation, in_config)
    batched = batch_multinode(process_of, inputs, computation, in_config,
                              at_path=to_stage, using_parallelization=(n_compute_nodes, n_jobs))

    if process_of == setup_compute_node:
        using_configs["slurm_params"] = configure_slurm(computation, in_config, using_runtime)

        compute_nodes = {c: setup_compute_node(c, inputs, (computation_type, to_stage), using_configs)
                         for c, inputs in batched.groupby("compute_node")}
        return {"configs": using_configs,
                "number_compute_nodes": n_compute_nodes, "number_total_jobs": n_jobs,
                "setup": write_multinode_setup(compute_nodes, inputs,  at_dirpath=to_stage)}

    if process_of == collect_multinode:
        setup = {c: read_setup_compute_node(c, for_quantity=to_stage) for c,_ in batched.groupby("compute_node")}
        at_base = Path(in_config["paths"]["output"]["store"])
        return collect_multinode(computation_type, setup, from_dirpath=to_stage, in_connsense_store=at_base)

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
    n_compute_nodes, n_total_jobs = using_parallelization

    if process_of == setup_compute_node:
        LOG.info("Assign batches to %s inputs", len(inputs))
        batches = assign_batches_to(inputs, upto_number=n_total_jobs)

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
    LOG.info("Configure chunk %s with %s inputs to compute %s.", c, len(inputs), for_computation)

    computation_type, for_quantity = describe(for_computation)

    for_compute_node = for_quantity / f"compute-node-{c}"
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
        return {"configure": "pipeline.yaml", "parallelize": "runtime.yaml"}

    def cmd_options():
        """..."""
        paramkey = PARAMKEY[computation_type]
        if paramkey[-1] == 's':
            paramkey = paramkey[:-1]
        return {paramkey: for_quantity.name}

    master_launchscript = for_quantity / "launchscript.sh"

    with open(master_launchscript, 'a') as to_launch:
        def write(aline):
            to_launch.write(aline + '\n')

        write("#!/bin/bash")

        write(f"########################## LAUNCH {computation_type} for chunk {c}"
            f" of {len(inputs)} _inputs. #######################################")
        write(f"pushd {for_compute_node}")

        sbatch = f"sbatch {of_executable.name} run {computation_type} \\"
        configs = ' '.join([f"--{config}={value}" for config, value in cmd_configs().items()]) + " \\"
        options = ' '.join([f"--{option}={value}" for option, value in cmd_options().items()]) + " \\"
        #batches = f"--input={inputs_to_read} \\"
        batches = f"--input={inputs_to_read} \\"
        output = f"--output={output_h5}"
        write(f"{sbatch}\n {configs}\n {options}\n {batches}\n {output}")

        write("popd")

    setup = {"dirpath": for_compute_node, "sbatch": of_executable, "input": inputs_to_read, "output": output_h5}

    return read_pipeline.write(setup, to_json=for_compute_node/"setup.json")


def write_multinode_setup(compute_nodes, inputs, at_dirpath):
    """..."""
    inputs_h5, dataset = INPUTS
    inputs.to_hdf(at_dirpath/inputs_h5, key=dataset)

    return read_pipeline.write({"compute_nodes": compute_nodes, "inputs": at_dirpath/inputs_h5},
                                to_json=at_dirpath/"setup.json")


def collect_multinode(computation_type, setup, from_dirpath, in_connsense_store):
    """..."""
    if not in_connsense_store.exists():
        raise RuntimeError(f"NOTFOUND {in_connsense_h5_at_basedir}\n HDF5 for connsense in base dir must exist")

    if computation_type == "extract-node-populations":
        return collect_node_population(setup, from_dirpath, in_connsense_store)

    if computation_type == "extract-edge-populations":
        return collect_edge_population(setup, from_dirpath, in_connsense_store)

    if computation_type == "analyze-connectivity":
        return collect_analyze_connectivity(setup, from_dirpath, in_connsense_store)

    raise NotImplementedError(f"INPROGRESS: {computation_type}")


def collect_edge_population(setup, from_dirpath, in_connsense_store):
    """..."""
    LOG.info("Collect edge population at %s using setup \n%s", from_dirpath, setup)

    try:
        with open(from_dirpath/"description.json", 'r') as f:
            population = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the population extracted: {at_basedir}") from ferr

    p = population["name"]
    adj_group = f"edges/populations/{p}/adj"
    props_group = f"edges/populations/{p}/props"

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
        try:
            from_connsense_h5 = output["adj"]
        except KeyError as kerr:
            raise RuntimeError(f"No adjacencies registered in compute node {of_compute_node}/output.json") from kerr

        adj = read_toc_plus_payload(from_connsense_h5, for_step="extract-edge-populations")
        return write_toc_plus_payload(adj, (in_connsense_store, adj_group), append=True)

    LOG.info("Collect adjacencies")
    adjacencies = {c: collect_adjacencies(of_compute_node=c, output=o) for c, o in outputs.items()}
    LOG.info("Adjacencies collected: \n%s", adjacencies)

    if "properties" not in population:
        LOG.info("No properties were extracted")
        return adjacencies

    def props_store(compute_node, output):
        """..."""
        try:
            props = output["props"]
        except KeyError as kerr:
            raise RuntimeError(f"No properties for compute node {compute_node} in its output {output}") from kerr

        hdf, group = props
        return matrices.get_store(hdf, group, for_matrix_type="pandas.DataFrame")

    in_base_connsense_props = props_store("base", {"props": (in_connsense_store, props_group)})

    LOG.info("Collect properties")
    properties = in_base_connsense_props.collect({of_compute_node: props_store(of_compute_node, output)
                                                  for of_compute_node, output in outputs.items()})
    LOG.info("Properties collected \n%s", properties)
    return {"adj": adjacencies, "props": properties}


def collect_node_population(setup, from_dirpath, in_connsense_store):
    """..."""
    from connsense.io.write_results import read as read_compute_node, write as write_compute_node
    LOG.info("Collect node population at %s using setup \n%s", from_dirpath, setup)

    try:
        with open(from_dirpath/"description.json", 'r') as f:
            population = json.load(f)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the population extracted: {at_basedir}") from ferr

    def describe_output(of_compute_node):
        """..."""
        try:
            with open(Path(of_compute_node["dirpath"]) / "output.json", 'r') as f:
                output = json.load(f)
        except FileNotFoundError as ferr:
            raise RuntimeError(f"No output configured for compute node {of_compute_node}") from ferr
        return output

    p = population["name"]
    hdf_group = f"nodes/populations/{p}"

    def move(compute_node, from_path):
        """..."""
        LOG.info("Write batch %s read from %s", compute_node, from_path)
        compute_node_result = describe_output(from_path)
        result = read_compute_node(compute_node_result, "extract-node-populations")
        return write_compute_node(result, to_path=(in_connsense_store, hdf_group), append=True, format="table",
                                  min_itemsize={"subtarget": 128})

    for compute_node, hdf_path in setup.items():
        move(compute_node, hdf_path)

    return (in_connsense_store, hdf_group)


def collect_analyze_connectivity(setup, from_dirpath, in_connsense_store):
    """..."""
    try:
        with open(at_basedir/"description.json", 'r') as f:
            config = json.load(f)
        analysis = SingleMethodAnalysisFromSource(at_basedir.name, config)
    except FileNotFoundError as ferr:
        raise RuntimeError(f"NOTFOUND a description of the analysis: {at_basedir}") from ferr

    of_quantity = analysis.name

    def in_store(at_path):
        """..."""
        return matrices.get_store(at_path, f"analysis/{of_quantity}", analysis.output_type)

    return in_store(in_connsense_store).collect({compute_node: in_store(at_its_rundir/"connsense.h5")
                                                         for compute_node, at_its_rundir in setup.items()})


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
    paramkey = PARAMKEY[computation_type]
    configured = in_config["parameters"][computation_type][paramkey][of_quantity]
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


def generate_inputs_of(computation, in_config):
    """..."""
    LOG.info("Generate inputs for  %s", computation)

    computation_type, of_quantity = describe(computation)
    if computation_type == "extract-edge-populations":
        return input_subtargets(in_config)

    input_paths, _ = read_pipeline.check_paths(in_config, step=computation_type)

    parameters = parameterize(computation_type, of_quantity, in_config)

    if computation_type == "extract-node-types":
        circuit = input_circuit(parameters["input"], in_config)
        if not circuit:
            raise RuntimeError("MIssing circuit to extract-node-types")

        if parameters["input"]:
            raise RuntimeError(f"UNADMISSABLE arguments {parameters['input']} to extract-node-types")

        extractor = parameters["extractor"]
        _, extract = plugins.import_module(extractor["source"], extractor["method"])
        return extract(circuit)

    cfg_inputs = parameters["input"].items()
    inputs = [load_connsense_input(dset, in_config, with_name=arg) for arg, dset in cfg_inputs  if arg != "circuit"]
    if len(inputs) == 1:
        return inputs[0]

    return pd.concat(inputs, axis=1)


def load_connsense_input(computation, in_config, with_name):
    """..."""
    from ..io.write_results import read
    computation_type, of_quantity = describe(computation)
    LOG.info("Load connsense input %s %s", computation_type, of_quantity)

    input_paths, _ = read_pipeline.check_paths(in_config, step=computation_type)
    hdf_path, group = input_paths["steps"][computation_type]

    if computation_type == "define-subtargets":
        assert not of_quantity or of_quantity == ""
        subtargets = read((hdf_path, group), for_step=computation_type)

        # subset for testing
        # from bluepy import Cell
        # circuit = input_circuit("Bio_M", in_config)
        # def filter_l1(gids):
        #     layers = circuit.cells.get(gids, Cell.LAYER)
        #     return list(layers[layers==1].index.values)
        # return subtargets.apply(filter_l1)


        return subtargets

    key = f"{group}/{of_quantity}"

    if computation_type in ("extract-node-types", "extract-node-populations"):
        return read((hdf_path, key), for_step=computation_type)

    if computation_type == "extract-edge_populations":

        if dataset.endswith("/adj"):
            return read_toc_plus_payload((hdf_path, key)).rename(with_name)

        if dataset.endswith("/props"):
            return (matrices.get_store(hdf_path, key, for_matrix_type="pandas.DataFrame", in_mode='r').toc
                    .rename(with_name))

        raise RutimeError(f"Unknown dataset for results of extract-edge-population: {dataset}")

    parameters = parameterize(computation_type, of_quantity, in_config)

    if computation_type.startswith("analyze-"):
        return (matrices.get_store(hdf_path, key, for_matrix_type=parameters["output"], in_mode='r').toc.
                rename(with_name))

    raise NotImplementedError(f"computation type of {computation_type}")


def input_subtargets(in_config):
    """..."""
    _, output_paths = read_pipeline.check_paths(in_config, "define-subtargets")
    path_subtargets = output_paths["steps"]["define-subtargets"]
    LOG.info("Read subtargets from %s", path_subtargets)

    subtargets = read_results(path_subtargets, for_step="define-subtargets")
    LOG.info("Read %s subtargets", len(subtargets))

    #subset for testing
    # from bluepy import Cell
    # circuit = input_circuit("Bio_M", in_config)
    # def filter_l1(gids):
    #     layers = circuit.cells.get(gids, Cell.LAYER)
    #     return list(layers[layers==1].index.values)
    # return subtargets.apply(filter_l1)

    return subtargets


def input_networks(in_config, to_analyze): #pylint: disable=unused-argument
    """..."""
    raise NotImplementedError("INPROGRESS")


def parameterize(computation_type, of_quantity, in_config):
    """..."""
    """..."""
    paramkey = PARAMKEY[computation_type]

    if not computation_type in in_config["parameters"]:
        raise RuntimeError(f"Unknown {computation_type}")

    if computation_type != "define-subtargets":
        if of_quantity not in in_config["parameters"][computation_type][paramkey]:
            raise RuntimeError(f"Unknown {paramkey[:-1]} {of_quantity} for {computation_type}")
        return deepcopy(in_config["parameters"][computation_type][paramkey][of_quantity])

    return deepcopy(in_config["parameters"]["define-subtargets"])

def configure_slurm(computation, in_config, using_runtime):
    """..."""
    computation_type, quantity = computation.split('/')
    pipeline_config = in_config if isinstance(in_config, Mapping) else read_pipeline.read(in_config)
    from_runtime = (read_runtime_config(for_parallelization=using_runtime, of_pipeline=pipeline_config)
                    if not isinstance(using_runtime, Mapping) else using_runtime)
    return from_runtime["pipeline"].get(computation_type, {}).get(quantity, None).get("sbatch", None)


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


def read_runtime_config(for_parallelization, of_pipeline=None):
    """..."""
    assert not of_pipeline or isinstance(of_pipeline, Mapping), of_pipeline

    if not for_parallelization:
        return None

    try:
        path = Path(for_parallelization)
    except TypeError:
        assert isinstance(for_parallelization, Mapping)
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
        return config

    from_runtime = config["pipeline"]
    default_sbatch = lambda : deepcopy(config["slurm"]["sbatch"])

    def configure_slurm_for(computation_type):
        """..."""
        try:
            cfg_computation_type = of_pipeline["parameters"][computation_type]
        except KeyError:
            return None

        paramkey = PARAMKEY[computation_type]
        quantities_to_configure = cfg_computation_type[paramkey]
        configured = from_runtime.get(computation_type, {})[paramkey]

        def configure_quantity(q):
            cfg = deepcopy(configured.get(q) or {})
            if "sbatch" not in cfg:
                cfg["sbatch"] = default_sbatch()
            if "number-compute-nodes" not in cfg:
                cfg["number-compute-nodes"] = 1
            if "number-tasks-per-node" not in cfg:
                cfg["number-tasks-per-node"] = 1
            return cfg

        return {q: configure_quantity(q) for q in quantities_to_configure if q != "description"}

    runtime_pipeline = {c: configure_slurm_for(computation_type=c) for c in of_pipeline["parameters"]}
    return {"version": config["version"], "date": config["date"], "pipeline": runtime_pipeline}


def prepare_parallelization(computation, in_config, using_runtime):
    """.."""
    computation_type, quantity = computation.split('/')
    from_runtime = (read_runtime_config(for_parallelization=using_runtime, of_pipeline=in_config)
                    if not isinstance(using_runtime, Mapping) else using_runtime)
    LOG.info("prepare parallelization %s using runtime \n%s", computation, pformat(from_runtime))
    configured = from_runtime["pipeline"].get(computation_type, {})
    LOG.info("\t Configured \n%s", configured)
    return read_njobs(to_parallelize=configured, computation_of=quantity)


def assign_batches_to(inputs, upto_number):
    """..."""
    def estimate_load(input_data): #pylint: disable=unused-argument
        """Needs improvement.."""
        try:
            shape = input_data.shape
        except AttributeError:
            return len(input_data)
        return np.prod(shape)

    if isinstance(inputs, pd.Series):
        weights = inputs.apply(estimate_load).sort_values(ascending=True)
    elif isinstance(inputs, pd.DataFrame):
        weights = inputs.apply(estimate_load, axis=1).sort_values(ascending=True)
    else:
        raise TypeError(f"Unhandled type of input: {inputs}")

    computational_load = np.cumsum(weights) / weights.sum()
    batches = (upto_number * (computational_load - computational_load.min())).apply(int).rename("batch")

    LOG.info("Load balanced batches for %s inputs: \n %s", len(inputs), batches)
    return batches
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
    kwargs = {key: value for key, value in parameters.items() if key not in ("description", "input",
                                                                             "extractor", "computation", "output")}
    def compute_batch(of_input, *, index, in_bowl):
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

    input_data = generate_inputs_of(of_computation, in_config)

    batches = load_inputs(on_compute_node, inputs)
    n_batches = batches.batch.max() - batches.batch.min() + 1

    for index, batch in batches.groupby("batch"):
        LOG.info("Spawn compute node %s process %s / %s batches", on_compute_node, index, n_batches)
        batch_inputs = input_data.loc[batch.index]
        p = Process(target=compute_batch, args=(batch_inputs,), kwargs={"index": index, "in_bowl": bowl})
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
    c = computation_inputs.pop("circuit", None)
    circuit = input_circuit(c, in_config)
    connectome = computation_inputs.pop("connectome", None)
    return tuple(x for x in (circuit, connectome) if x)


def load_inputs(on_compute_node, inputs=None):
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

    executable_type = {"extract": "extractor", "analyze": "computation"}[computation_type.split('-')[0]]

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
        adj = write_toc_plus_payload(batch_adj, (in_connsense_h5, hdf_edges +'/adj'), append=True, format="table")

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
        return write_batch(result, to_path=in_hdf, append=True, format="table", min_itemsize={"subtarget": 128})

    for batch, hdf_path in results.items():
        move(batch, hdf_path)

    return in_hdf