

# #+RESULTS:

# For development we can use a develop version,


from collections.abc import Mapping
from copy import deepcopy
import shutil
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
from connsense.io import logging, time, read_config as read_pipeline
from connsense.io.slurm import SlurmConfig
from connsense.io.write_results import read_toc_plus_payload, write_toc_plus_payload
from connsense.pipeline.workspace import find_base
from connsense.pipeline import ConfigurationError, NotConfiguredError
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


def batch_multinode(computation, of_inputs, in_config, at_dirpath, using_parallelization, single_submission=500):
    """...Just read the method definition above,
    and code below
    """
    n_compute_nodes, n_parallel_jobs = using_parallelization

    LOG.info("Assign batches to %s inputs", len(of_inputs))
    batches = batch_parallel_groups(of_inputs, upto_number=n_parallel_jobs)

    n_batches = batches.max() + 1
    LOG.info("Assign compute nodes to %s batches of %s of_inputs", len(batches), n_batches)
    compute_nodes = distribute_compute_nodes(batches, upto_number=n_compute_nodes)

    LOG.info("Group %s compute node into launchscript submissions", compute_nodes.nunique())
    submissions = group_launchscripts(compute_nodes, max_entries=single_submission)

    assignment = pd.concat([batches, compute_nodes, submissions], axis=1)
    assignment_h5, dataset = COMPUTE_NODE_ASSIGNMENT
    assignment.to_hdf(at_dirpath/assignment_h5, key=dataset)

    return assignment



def batch_parallel_groups(of_inputs, upto_number, to_compute=None, return_load=False):
    """..."""
    if isinstance(of_inputs, pd.Series):
        weights = of_inputs.apply(estimate_load(to_compute)).sort_values(ascending=True).rename("load")
    elif isinstance(of_inputs, pd.DataFrame):
        weights = of_inputs.apply(estimate_load(to_compute), axis=1).sort_values(ascending=True).rename("load")
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

        if callable(d): return of_input_data(d())

        if isinstance(d, Mapping):
            if not d: return 1.
            first = next(v for v in d.values())
            return of_input_data(first)

        if isinstance(d, pd.Series): return d.apply(of_input_data).sum()

        try:
            shape = d.shape
        except AttributeError:
            pass
        else:
            return np.prod(shape)

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
    assignment = pd.Series(compute_nodes[parallel_batches], name=dset, index=parallel_batches.index)
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
            if var not in ("circuit", "connectome") and isinstance(val, Mapping) and "dataset" in val}


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
    dataset = tap.pour_dataset(variable, values["dataset"])

    def unpack_value(v):
        """..."""
        try:
            get = v.get_value
        except AttributeError:
            return v
        return get()

    LOG.info("Pour %s dataset ", variable)

    return dataset.apply(lazily(to_evaluate=unpack_value))


def pour(tap, datasets):
    """..."""
    LOG.info("Pour tap \n%s\n to get values for variables:\n%s", tap._root, pformat(datasets))
    datasets = sorted([(variable, load_dataset(tap, variable, values)) for variable, values in datasets.items()],
                      key=lambda x: len(x[1].index.names), reverse=True)

    def collect(aggregated, datasets):
        """..."""
        if not datasets:
            return aggregated

        head, dset_head = datasets[0]
        vars, dset_agg = aggregated
        combined = pd.concat([dset_agg, dset_head.loc[dset_agg.index]], axis=1, keys=vars+[head])
        return collect((vars+[head], combined), datasets[1:])

    variable, primary = datasets[0]
    return collect(([variable], primary), datasets[1:])[1].apply(lambda row: row.to_dict(), axis=1)


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

def generate_inputs(of_computation, in_config):
    """..."""
    LOG.info("Generate inputs for %s.", of_computation)

    computation_type, of_quantity = describe(of_computation)
    params = parameterize(computation_type, of_quantity, in_config)
    tap = HDFStore(in_config)

    return pour(tap=HDFStore(in_config), datasets=filter_datasets(params["input"]))


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


def setup_compute_node(c, of_computation, with_inputs, using_configs, at_dirpath):
    """..."""
    from connsense.apps import APPS
    LOG.info("Configure chunk %s with %s inputs to compute %s", c, len(with_inputs), of_computation)

    computation_type, of_quantity = describe(of_computation)
    for_compute_node = at_dirpath / f"compute-node-{c}"
    for_compute_node.mkdir(parents=False, exist_ok=True)
    configs = symlink_pipeline(using_configs, at_dirpath=for_compute_node)

    inputs_to_read = write_compute(batches=with_inputs, to_hdf=INPUTS, at_dirpath=for_compute_node)
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

    of_executable = cmd_sbatch(at_path=for_compute_node, executable=APPS["main"])

    def cmd_configs():
        """..."""
        return "--configure=pipeline.yaml --parallelize=runtime.yaml \\"

    def cmd_options():
        """..."""
        return None

    if "submission" not in with_inputs:
        launchscript = at_dirpath / "launchscript.sh"
    else:
        submission = with_inputs.submission.unique()
        assert len(submission) == 1
        launchscript = at_dirpath / f"launchscript-{submission[0]}.sh"

    command_lines = ["#!/bin/bash",
                    (f"########################## LAUNCH {computation_type} for chunk {c}"
                     f" of {len(with_inputs)} _inputs. #######################################"),
                    f"pushd {for_compute_node}",
                    f"sbatch {of_executable.name} run {computation_type} {of_quantity} \\",
                    cmd_configs(),
                    cmd_options(),
                    f"--input={inputs_to_read} \\",
                    f"--output={output_h5}",
                    "popd"]

    with open(launchscript, 'a') as to_launch:
        to_launch.write('\n'.join(l for l in command_lines if l) + "\n")

    setup = {"dirpath": for_compute_node, "sbatch": of_executable, "input": inputs_to_read, "output": output_h5}

    return read_pipeline.write(setup, to_json=for_compute_node/"setup.json")


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


def setup_multinode(process, of_computation, in_config, using_runtime):
    """Setup a multinode process.
    """
    _, to_stage = get_workspace(of_computation, in_config)

    using_configs = configure_multinode(process, of_computation, in_config, at_dirpath=to_stage)

    computation_type, of_quantity = describe(of_computation)

    inputs = generate_inputs(of_computation, in_config)
    n_compute_nodes, n_parallel_jobs = prepare_parallelization(of_computation, in_config, using_runtime)

    if process == setup_compute_node:
        batched = batch_multinode(of_computation, inputs, in_config,
                                  at_dirpath=to_stage, using_parallelization=(n_compute_nodes, n_parallel_jobs))
        using_configs["slurm_params"] = configure_slurm(of_computation, in_config, using_runtime).get("sbatch", None)
        compute_nodes = {c: setup_compute_node(c, of_computation, inputs, using_configs, at_dirpath=to_stage)
                         for c, inputs in batched.groupby("compute_node")}
        return {"configs": using_configs,
                "number_compute_nodes": n_compute_nodes, "number_total_jobs": n_parallel_jobs,
                "setup": write_multinode_setup(compute_nodes, inputs, at_dirpath=to_stage)}

    if process == collect_results:
        batched = read_compute_nodes_assignment(at_dirpath)
        _, output_paths = read_pipeline.check_paths(in_config, step=computation_type)
        at_path = output_paths["steps"][computation_type]

        setup = {c: read_setup_compute_node(c, for_quantity=to_stage) for c, _ in batched.groupby("compute_node")}
        return collect_results(computation_type, setup, from_dirpath=to_stage, in_connsense_store=at_path)

    return ValueError(f"Unknown multinode {process}")


def reindex_input(transformations, to_tap, variable, dataset, of_analysis):
    """..."""
    try:
        to_reindex = transformations["reindex"]
    except KeyError:
        LOG.info("No reindexing configured for %s of analysis %s", variable, of_analysis)
        return dataset

    original = dataset.apply(lambda subtarget: to_tap.reindex(subtarget, variables=to_reindex))
    return pd.concat(original.values, axis=0, keys=original.index.values, names=original.index.values)



def apply_control(transformations, to_tap, variable, dataset, of_analysis):
    """..."""
    try:
        configured = transformations["controls"]
    except KeyError:
        LOG.info("No controls configured for %s", variable)
        return dataset

    original = [dataset]
    controls = load_controls(configured)

    return (pd.concat(original + [dataset.apply(control) for control in controls], axis=0,
                      keys=(["original"] + [control.__name__ for control in controls]), names=["control"])
            .reorder_levels(dataset.index.names + ["control"]).sort_index())


def load_controls(configured):
    """..."""
    def load_shufflers(control, labeled):
        """..."""
        LOG.info("Load shufflers to control %s: \n%s", labeled, pformat(control))
        _, algorithm = plugins.import_module(control["algorithm"])

        kwargs = deepcopy(control["kwargs"])
        seeds = kwargs.pop("seeds", None)

        def seed_shuffler(s):
            """..."""
            def shuffle(subgraph, seed=s, **kwargs):
                """..."""
                return algorithm(subgraph(), seed=s)

            lazy_shuffle = lazily(to_evaluate=shuffle)
            lazy_shuffle.__name__  = f"{labeled}-{s}"
            return lazy_shuffle

        return [seed_shuffler(s) for s in seeds]

    return [shuffler for shufflers in (load_shufflers(control, labeled=c) for c, control in configured.items())
            for shuffler in shufflers]



def apply_transformations(in_values, to_tap, variable, dataset, of_analysis):
    """..."""
    try:
        transformations = in_values["transformations"]
    except KeyError:
        LOG.info("No transformations configured for %s", variable)
        return dataset

    reindexed = reindex_input(transformations, to_tap, variable, dataset, of_analysis)
    controlled = apply_control(transformations, to_tap, variable, reindexed, of_analysis)

    return controlled



def input_units(computation, to_tap):
    """..."""
    described = parameterize(*describe(computation), to_tap._config)
    datasets = {variable: apply_transformations(in_values, to_tap, variable,
                                                load_dataset(to_tap, variable, in_values), of_analysis=computation)
                for variable, in_values in filter_datasets(described["input"]).items()}
    return datasets
