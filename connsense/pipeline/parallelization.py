# Putting it together
# We can now list the code that can configure a multinode computation.
# which we do to keep the output Python code clean.


from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from pprint import pformat

import json
import yaml

import numpy as np
import pandas as pd

from connsense.pipeline import workspace
from connsense.pipeline.pipeline import PARAMKEY
from connsense.io import logging, read_config as read_pipeline
from connsense.io.slurm import SlurmConfig
from connsense.io.write_results import read_toc_plus_payload, write_toc_plus_payload
from connsense.pipeline.workspace import find_base
from connsense.analyze_connectivity import check_paths, matrices
from connsense.analyze_connectivity.analysis import SingleMethodAnalysisFromSource
from connsense.apps import APPS
from connsense.extract_connectivity import read_results

# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, too-many-locals, comparison-with-callable, too-many-arguments, invalid-name, unspecified-encoding, unnecessary-lambda-assignment

LOG = logging.get_logger("connsense pipeline")

def run_multinode(process_of, *, computation, in_config, using_runtime, for_control=None, making_subgraphs=None):
    """..."""
    _, to_stage = get_workspace(computation, in_config, for_control, making_subgraphs)

    using_configs = run_multinode_configs(process_of, computation, in_config, for_control, making_subgraphs,
                                          at_dirpath=to_stage)
    n_compute_nodes, n_jobs = prepare_parallelization(computation, in_config, using_runtime)

    computation_type, _ = describe(computation)

    batched_inputs = run_multinode_inputs(process_of, computation, in_config, to_stage, with_number_jobs=n_jobs)

    chunked = run_multinode_compute_nodes(process_of, batched_inputs, numbering_upto=n_compute_nodes,
                                          at_dirpath=to_stage)

    if process_of == setup_compute_node:
        using_configs["slurm_params"] = configure_slurm(computation, in_config, using_runtime)
        compute_nodes = {c: setup_compute_node(c, inputs, (computation_type, to_stage), using_configs)
                         for c, inputs in chunked.groupby("compute_node")}
        return {"configs": using_configs,
                "number_compute_nodes": n_compute_nodes, "number_total_jobs": n_jobs,
                "setup": write_multinode_setup(compute_nodes, at_dirpath=to_stage)}

    if process_of == collect_multinode:
        setup = {c: read_setup_compute_node(c, for_quantity=to_stage) for c,_ in chunked.groupby("compute_node")}
        at_base = in_config["paths"]["output"]["store"]
        return collect_multinode(computation_type, setup, at_dirpath=to_stage, in_connsense_store=at_base)

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


def run_multinode_inputs(process_of, computation, in_config, to_stage, with_number_jobs):
    """..."""
    if process_of == setup_compute_node:
        inputs = generate_inputs_of(computation, in_config)
        batches = assign_batches_to(inputs, with_number_jobs)
        batched_inputs = pd.concat([inputs, batches], axis=1)
        write_compute(batched_inputs, to_hdf=BATCH_SUBTARGETS, at_dirpath=to_stage)
        return batched_inputs

    if process_of == collect_multinode:
        subtargets_h5, dset = BATCH_SUBTARGETS
        path_subtargets = to_stage / subtargets_h5
        if not path_subtargets.exists():
            raise RuntimeError(f"No subtargets run by TAP at {to_stage}"
                               f" Expecting an HDF5 file created during the TAP run of {computation}")
        return pd.read_hdf(path_subtargets, key=dset)

    raise ValueError(f"Unknown {process_of} multinode")


def run_multinode_compute_nodes(process_of, batched_inputs, numbering_upto, at_dirpath):
    """..."""
    if process_of == setup_compute_node:
        return assign_compute_nodes(batched_inputs, numbering_upto, at_dirpath)

    if process_of == collect_multinode:
        return read_compute_nodes_assignment(at_dirpath)

    raise ValueError(f"Unknown {process_of} multinode")


def setup_compute_node(c, inputs, for_computation, using_configs):
    """..."""
    LOG.info("Configure chunk %s with %s inputs to compute %s.", c, len(inputs), for_computation)

    computation_type, for_quantity = for_computation

    for_compute_node = for_quantity / f"compute-node-{c}"
    for_compute_node.mkdir(parents=False, exist_ok=True)
    configs = symlink_pipeline(configs=using_configs, at_dirpath=for_compute_node)

    inputs_to_read = write_compute(inputs, to_hdf=COMPUTE_NODE_SUBTARGETS, at_dirpath=for_compute_node)
    output_h5 = f"{for_compute_node}/connsense.h5"

    def cmd_sbatch(at_path):
        """..."""
        try:
            slurm_params = using_configs["slurm_params"]
        except KeyError as kerr:
            raise RuntimeError("Missing slurm params") from kerr

        slurm_params.update({"name": computation_type, "executable": APPS[computation_type]})
        slurm_config = SlurmConfig(slurm_params)
        return slurm_config.save(to_filepath=at_path/f"{computation_type}.sbatch")

    of_executable = cmd_sbatch(at_path=for_compute_node)

    def cmd_configs():
        """..."""
        if computation_type == "extract-edge-populations":
            return {"configure": "pipeline.yaml", "parallelize": "runtime.yaml"}
        raise NotImplementedError("Will do when the need arises a.k.a when we get there.")

    def cmd_options():
        """..."""
        if computation_type == "extract-edge-populations":
            return {"connectome": for_quantity.name}
        raise NotImplementedError("Will do when the need arises a.k.a when we get there.")

    master_launchscript = for_quantity / "launchscript.sh"

    with open(master_launchscript, 'a') as to_launch:
        def write(aline):
            to_launch.write(aline + '\n')

        write("#!/bin/bash")

        write(f"########################## LAUNCH {computation_type} for chunk {c}"
            f" of {len(inputs)} _inputs. #######################################")
        write(f"pushd {for_compute_node}")

        sbatch = f"sbatch {of_executable.name} run \\"
        configs = ' '.join([f"--{config}={value}" for config, value in cmd_configs().items()]) + " \\"
        options = ' '.join([f"--{option}={value}" for option, value in cmd_options().items()]) + " \\"
        batches = f"--batch={inputs_to_read} \\"
        output = f"--output={output_h5}"
        write(f"{sbatch}\n {configs}\n {options}\n {batches}\n {output}")

        write("popd")

    setup = {"dirpath": for_compute_node, "sbatch": of_executable, "input": inputs_to_read, "output": output_h5}

    return read_pipeline.write(setup, to_json=for_compute_node/"setup.json")


def write_multinode_setup(config, at_dirpath):
    """..."""
    return read_pipeline.write(config, to_json=at_dirpath/"setup.json")


def collect_multinode(computation_type, setup, at_dirpath, in_connsense_store):
    """..."""
    if not in_connsense_store.exists():
        raise RuntimeError(f"NOTFOUND {in_connsense_h5_at_basedir}\n HDF5 for connsense in base dir must exist")

    if computation_type == "extract-edge-populations":
        return collect_edge_population(setup, at_dirpath, in_connsense_store)

    if computation_type == "analyze-connectivity":
        return collect_analyze_connectivity(setup, at_dirpath, in_connsense_store)

    raise NotImplementedError(f"INPROGRESS: {computation_type}")


def collect_edge_population(setup, at_dirpath, in_connsense_store):
    """..."""
    LOG.info("Collect edge population at %s using setup \n%s", at_dirpath, setup)

    try:
        with open(at_dirpath/"description.json", 'r') as f:
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


def collect_analyze_connectivity(setup, at_dirpath, in_connsense_store):
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


def _remove_link(path):
    try:
        return path.unlink()
    except FileNotFoundError:
        pass
    return None


BATCH_SUBTARGETS = ("subtargets.h5", "batch")
COMPUTE_NODE_SUBTARGETS = ("batches.h5", "compute_node")


def describe(computation):
    """..."""
    computation_type, quantity = computation.split('/')
    return (computation_type, quantity)


def configure_multinode(computation, in_config, using_runtime, for_control=None, making_subgraphs=None):
    """..."""
    _, to_run_quantity = get_workspace(computation, in_config, for_control, making_subgraphs)

    def _write_configs(at_dirpath):
        return write_configs_of(computation, in_config, at_dirpath, with_random_shuffle=for_control,
                                in_the_subtarget=making_subgraphs)

    _write_configs(at_dirpath=to_run_quantity)
    n_compute_nodes, n_jobs = prepare_parallelization(computation, in_config, using_runtime)

    computation_type, quantity = describe(computation)


    def cmd_sbatch(at_path):
        """..."""
        slurm_params = configure_slurm(computation, in_config, using_runtime)
        slurm_params.update({"name": computation_type, "executable": APPS[computation_type]})
        slurm_config = SlurmConfig(slurm_params)
        return slurm_config.save(to_filepath=at_path/f"{computation_type}.sbatch")

    def cmd_configs():
        """..."""
        if computation_type == "extract-edge-populations":
            return {"configure": "pipeline.yaml", "parallelize": "runtime.yaml"}
        raise NotImplementedError("Will do when the need arises a.k.a when we get there.")

    def cmd_options():
        """..."""
        if computation_type == "extract-edge-populations":
            return {"connectome": quantity}
        raise NotImplementedError("Will do when the need arises a.k.a when we get there.")

    master_launchscript = to_run_quantity / "launchscript.sh"

    inputs = generate_inputs_of(computation, in_config)

    def configure_chunk(c, _inputs):
        """..."""
        LOG.info("Configure chunk %s with %s inputs %s.", c, len(_inputs), list(_inputs.keys()))

        for_compute_node = to_run_quantity / f"compute-node-{c}"
        for_compute_node.mkdir(parents=False, exist_ok=True)
        _write_configs(at_dirpath=for_compute_node)

        read_input_batches = write_compute(_inputs, to_hdf=COMPUTE_NODE_SUBTARGETS, at_dirpath=for_compute_node)

        with open(master_launchscript, 'a') as to_launch:
            script = cmd_sbatch(at_path=for_compute_node).name

            def write(aline):
                to_launch.write(aline + '\n')

            write("#!/bin/bash")

            write(f"########################## LAUNCH {computation_type} for chunk {c}"
                f" of {len(_inputs)} _inputs. #######################################")
            write(f"pushd {for_compute_node}")

            sbatch = f"sbatch {script} run \\"
            configs = ' '.join([f"--{config}={value}" for config, value in cmd_configs().items()]) + " \\"
            options = ' '.join([f"--{option}={value}" for option, value in cmd_options().items()]) + " \\"
            batches = f"--batch={for_compute_node/read_input_batches} \\"
            output = f"--output={for_compute_node}/compute_node_connsense.h5"
            write(f"{sbatch}\n {configs}\n {options}\n {batches}\n {output}")

            write("popd")

        return to_run_quantity

    batched_inputs = assign_batches_to(inputs, n_jobs)
    write_compute(batched_inputs, to_hdf=BATCH_SUBTARGETS, at_dirpath=to_run_quantity)

    chunked = assign_compute_nodes(batched_inputs, n_compute_nodes, at_dirpath=to_run_quantity)
    return {c: configure_chunk(c, inputs) for c, inputs in chunked.groupby("compute_node")}


def get_workspace(for_computation, in_config, for_control=None, making_subgraphs=None, in_mode='r'):
    """..."""
    m = {'r': "test", 'w': "prod", 'a': "develop"}[in_mode]
    computation_type, of_quantity = for_computation.split('/')
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
            "runtime": {fmt: symlink_to(config_at_path=p) for fmt, p in configs["pipeline"].items() if p}}


def symlink_pipeline_control(to_config, at_dirpath):
    """..."""
    return create_symlink(at_dirpath)(to_config) if to_config else None


def symlink_pipeline_subgraphs(to_config, at_dirpath):
    """..."""
    return create_symlink(at_dirpath)(to_config) if to_config else None


def generate_inputs_of(computation, in_config):
    """..."""
    LOG.info("Generate inputs for  %s", computation)

    computation_type, _ = describe(computation)

    if computation_type == "extract-edge-populations":
        return input_subtargets(in_config)

    if computation_type == "analyze-connectivity":
        raise NotImplementedError("INPROGRESS")

    raise NotImplementedError(f"inputs to {computation}: INPROGRESS")


def input_subtargets(in_config):
    """..."""
    _, output_paths = read_pipeline.check_paths(in_config, "define-subtargets")
    path_subtargets = output_paths["steps"]["define-subtargets"]
    LOG.info("Read subtargets from %s", path_subtargets)

    subtargets = read_results(path_subtargets, for_step="define-subtargets")
    LOG.info("Read %s subtargets", len(subtargets))
    return subtargets


def input_networks(in_config, to_analyze): #pylint: disable=unused-argument
    """..."""
    raise NotImplementedError("INPROGRESS")


def parameterize(computation_type, of_quantity, in_config):
    """..."""
    parameters = in_config["parameters"][computation_type]
    return parameters[PARAMKEY[computation_type]][of_quantity]


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
        return 1.

    weights = inputs.apply(estimate_load).sort_values(ascending=True)
    computational_load = np.cumsum(weights) / weights.sum()
    batches = ((upto_number - 1) * computational_load).apply(int).rename("batch")

    LOG.info("Load balanced batches for %s inputs: \n %s", len(inputs), batches)
    return batches.loc[inputs.index]


def assign_compute_nodes(batched_inputs, n_compute_nodes, at_dirpath):
    """..."""
    batches = batched_inputs.batch
    assignment = pd.Series(np.linspace(0, n_compute_nodes - 1.e-6, batches.max() + 1, dtype=int)[batches.values],
                           name="compute_node", index=batched_inputs.index)
    LOG.info("Assign compute nodes to \n%s", batched_inputs)
    LOG.info("with batches \n%s", batches)

    assignment = pd.concat([batched_inputs, assignment], axis=1)
    assignment_h5, dataset = COMPUTE_NODE_SUBTARGETS
    assignment.to_hdf(at_dirpath / assignment_h5, key=dataset)
    return assignment


def read_compute_nodes_assignment(at_dirpath):
    """..."""
    assignment_h5, dataset = COMPUTE_NODE_SUBTARGETS

    if not (at_dirpath/assignment_h5).exists():
        raise RuntimeError(f"No compute node assignment saved at {at_dirpath}")

    return pd.read_hdf(at_dirpath / assignment_h5, key=dataset)


def write_compute(batches, to_hdf, at_dirpath):
    """..."""
    batches_h5, and_hdf_group = to_hdf
    batches.to_hdf(at_dirpath / batches_h5, key=and_hdf_group, format="fixed", mode='w')
    return at_dirpath / batches_h5
