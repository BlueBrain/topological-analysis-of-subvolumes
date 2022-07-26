# Putting it together
# We can now list the code that can configure a multinode computation. Excuse the double lines bettween individual entries,
# which we do to keep the output Python code clean.


from collections.abc import Mapping
from pathlib import Path
from pprint import pformat

import yaml
import json

import numpy as np
import pandas as pd

from connsense.pipeline import workspace
from connsense.pipeline.pipeline import PARAMKEY
from connsense.io import logging
from connsense.io.slurm import SlurmConfig
from connsense.apps import APPS

LOG = logging.get_logger("connsense pipeline")

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
    workspace = get_workspace(computation, in_config, for_control, making_subgraphs)
    computation_base, to_run_quantity = workspace

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

        batches_h5, dataset = COMPUTE_NODE_SUBTARGETS
        write_compute(_inputs, to_dirpath=for_compute_node, for_hdf=(for_compute_node/batches_h5, dataset))

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
            batches = f"--batch={for_compute_node/batches_h5} \\"
            output = f"--output={for_compute_node}/compute_node_connsense.h5"
            write(f"{sbatch}\n {configs}\n {options}\n {batches}\n {output}")

            write("popd")

        return to_run_quantity

    batches = assign_batches_to(inputs, n_jobs)
    write_compute(batches, to_dirpath=to_run_quantity, for_hdf=BATCH_SUBTARGETS)

    chunked = assign_compute_nodes(inputs, batches, n_compute_nodes)
    return {c: configure_chunk(c, inputs) for c, inputs in chunked.groupby("compute_node")}


def get_workspace(for_computation, in_config, for_control=None, making_subgraphs=None, in_mode='r'):
    """..."""
    m = {'r': "test", 'w': "prod", 'a': "develop"}
    computation_type, of_quantity = for_computation.split('/')
    rundir = workspace.get_rundir(in_config, computation_type, of_quantity, making_subgraphs, for_control, in_mode=m)
    basedir = workspace.find_base(rundir)
    return (basedir, rundir)


def write_configs_of(computation, in_config, at_dirpath, with_random_shuffle=None, in_the_subtarget=None):
    """..."""
    return {"configs": write_base_configs(computation, in_config, at_dirpath, with_random_shuffle, in_the_subtarget),
            "controls": write_control(with_random_shuffle, at_dirpath),
            "subgraphs": write_subgraphs(in_the_subtarget, at_dirpath)}


def write_base_configs(of_computation, in_config, at_dirpath, controlling, subgraphing):
    """..."""
    from connsense.pipeline.workspace import find_base
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
            else:
                LOG.info("Not found config %s", base_config)
            return None
        return {f: write_format(f) for f in ["json", "yaml"]}
    return {c: write_config(c) for c in ["pipeline", "runtime", "config", "parallel"]}


def write_control(algorithm, at_dirpath):
    """..."""
    if not algorithm: return None

    from connsense.io.read_config import write
    from copy import deepcopy
    control_json = at_dirpath / "control.json"
    description = deepcopy(algorithm.description)
    description["name"] = algorithm.name
    return write(description, control_json)


def write_subgraphs(in_the_subtarget, at_dirpath):
    """..."""
    return None


def generate_inputs_of(computation, in_config, for_batch=None, selecting=None):
    """..."""
    from connsense.io import read_config as read_pipeline

    computation_type, quantity = computation.split('/')


    if computation_type == "extract-edge-populations":
        population = quantity
        LOG.warning("Generate inputs to %s extract-connectivity for batch %s and selection %s",
                    population, for_batch, selecting)
        from connsense.extract_connectivity import read_results

        _, output_paths = read_pipeline.check_paths(in_config, "define-subtargets")
        path_subtargets = output_paths["steps"]["define-subtargets"]
        LOG.info("Read subtargets from %s", path_subtargets)

        subtargets = read_results(path_subtargets, for_step="define-subtargets")
        LOG.info("Read %s subtargets", len(subtargets))
        return subtargets

    if computation_type == "analyze-connectivity":
        raise NotImplementedError("Needs fiz for edge population to analyze")
        LOG.warning("Generate inputs to analyze-connectivity for batch %s and selection %s", for_batch, selecting)
        from connsense.analyze_connectivity import check_paths, load_adjacencies, load_neurons
        toc_dispatch = load_adjacencies(input_paths, for_batch, return_batches=False, sample=selecting)
        input_paths, output_paths = check_paths(in_config, "analyze-connectivity")
        toc_dispatch = load_adjacencies(input_paths, for_batch, return_batches=False, sample=selecting)

        if toc_dispatch is None:
            LOG.warning("Donw, with no connectivity matrices available to analyze for batch %s and selection %s",
                        for_batch, selecting)

        neurons = load_neurons(input_paths, index_with_connectome=substep, and_flatxy=False)
        return pd.concat([toc_dispatch, neurons.reindex(for_batch.index)], axis=1)


def parameterize(computation_type, of_quantity, in_config):
    """..."""
    parameters = in_config["parameters"][computation_type]
    return parameters[PARAMKEY[computation_type]][of_quantity]


def configure_slurm(computation, in_config, using_runtime):
    """..."""
    from connsense.io.read_config import read as read_pipeline
    computation_type, quantity = computation.split('/')
    pipeline_config = in_config if isinstance(in_config, Mapping) else read_pipeline(in_config)
    from_runtime = (read_config(for_parallelization=using_runtime, of_pipeline=pipeline_config)
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


def read_config(for_parallelization, of_pipeline=None):
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

    version = config["version"]
    date = config["date"]
    from_runtime = config["pipeline"]
    default_sbatch = lambda: {key: value for key, value in config["slurm"]["sbatch"].items()}

    def configure_slurm_for(computation_type):
        """..."""
        try:
            cfg_computaiton_type = of_pipeline["parameters"][computation_type]
        except KeyError:
            return None

        paramkey = PARAMKEY[computation_type]
        quantities_to_configure = cfg_computaiton_type[paramkey]
        configured = from_runtime.get(computation_type, {})[paramkey]

        def configure_quantity(q):
            cfg = {key: value for key, value in (configured.get(q) or {}).items()}
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
    from_runtime = (read_config(for_parallelization=using_runtime, of_pipeline=in_config)
                    if not isinstance(using_runtime, Mapping) else using_runtime)
    LOG.info("prepare parallelization %s using runtime \n%s", computation, pformat(from_runtime))
    configured = from_runtime["pipeline"].get(computation_type, {})
    LOG.info("\t Configured \n%s", configured)
    return read_njobs(to_parallelize=configured, computation_of=quantity)


def assign_batches_to(inputs, upto_number):
    """..."""
    def estimate_load(input):
        return 1.

    weights = inputs.apply(estimate_load).sort_values(ascending=True)
    computational_load = np.cumsum(weights) / weights.sum()
    batches = ((upto_number - 1) * computational_load).apply(int).rename("batch")

    LOG.info("Load balanced batches for %s inputs: \n %s", len(inputs), batches)
    return batches.loc[inputs.index]


def assign_compute_nodes(inputs, batches, n_compute_nodes):
    """..."""
    assignment = pd.Series(np.linspace(0, n_compute_nodes - 1.e-6, batches.max() + 1, dtype=int)[batches.values],
                           name="compute_node", index=inputs.index)
    LOG.info("Assign compute nodes to \n%s", inputs)
    LOG.info("with batches \n%s", batches)
    return pd.concat([inputs, batches, assignment], axis=1)


def write_compute(batches, to_dirpath, for_hdf):
    """..."""
    file_h5, and_hdf_group = for_hdf
    batches.to_hdf(to_dirpath / file_h5, key=and_hdf_group, format="fixed", mode='w')
