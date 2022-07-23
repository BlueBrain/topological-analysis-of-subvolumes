# Putting it together
# We can now list the code that can configure a multinode computation. Excuse the double lines bettween individual entries,
# which we do to keep the output Python code clean.


from connsense.pipeline import workspace


def _remove_link(path):
    try:
        return path.unlink()
    except FileNotFoundError:
        pass
    return None

BATCH_SUBTARGETS = ("subtargets.h5", "batch")


def write_configs(of_computation, at_dirpath):
    """..."""
    raise NotImplementedError


def configure_multinode(computation, in_config, using_runtime, for_control=None, making_subgraphs=None):
    """..."""

    workspace = get_workspace(computation, in_config, for_control, making_subgraphs)
    proj_dir, computation_base = workspace

    def write_configs(at_dirpath):
     return write_configs_of(computation, in_config, at_dirpath, with_random_shuffle=for_control,
                             and_in_the_subtarget=making_subgraphs)

    write_configs(at_dirpath=computation_base)
    slurm_config = configure_slurm(computation, in_config, using_runtime)
    n_compute_nodes, n_jobs = prepare_parallelization(computation, using_runtime)

    master_launchscript = to_run_in / "launchscript.sh"

    def configure_chunk(c, inputs):
        """..."""
        LOG.info("Configure chunk %s with %s inputs %s.", c, len(inputs), list(inputs.keys()))

        to_run_ = to_run_in / f"compute-node-{c}"
        rundir.mkdir(parents=False, exist_ok=True)
        write_configs(at_dirpath=rundir)

        with open(master_launchscript, 'w') as to_launch:
            script = cmd_sbatch(slurm_config, at_path=rundir).name

            def write(aline):
                to_launch.write(aline + '\n')

            write(f"########################## LAUNCH {name(computation)} for chunk {c}"
                f" of {len(inputs)} inputs. #######################################")
            write(f"pushd {rundir}")

            sbatch = f"sbatch {script} "
            configs = ' '.join(["--{config}={value}" for config, value in cmd_configs(computation, inputs).items()])
            options = ' '.join(["--{option}={value}" for option, value in cmd_options(computation, inputs).items()])
            write(f"{sbatch} {configs} {options} run")

            write("popd")

        return rundir

    inputs = generate_inputs_of(computation, in_config)

    batches = assign_batches(inputs, n_jobs)
    write_compute(batches, at_dirpath=computation_base)

    chunked = assign_compute_nodes(inputs, batches, n_compute_nodes)
    return {c: configure_chunk(c, inputs=i) for c, i in chunked.groupyby("compute_node")}


def get_workspace(for_computation, in_config, for_control, making_subgraphs, in_mode='r'):
    """..."""
    rundir = workspace.get_rundir(in_config, step, substep, making_subgraphs, for_controls, in_mode)
    basdir = workspace.find_base(rundir)
    return (basedir, rundir)


def write_configs_of(computation, in_config, at_dirpath, for_control, in_the_subtarget):
    """..."""
    return {"configs": write_configs(of_computation, in_config, at_dirpath),
            "controls": write_controls(algorithm=for_control, at_dirpath),
            "subgraphs": write_subgraphs(in_the_subtarget, at_dirpath)}


def write_configs(of_computation, in_config, at_dirpath, controlling, subgraphing):
    """..."""
    basedir = find_base(rundir=at_dirpath)
    def write(config):
        def write_format(f):
            filename = "{c}.{f}"
            base_config = basedir / filename
            if base_config.exists():
                run_config = at_dirpath / filename
                _remove_link(run_config)
                run_config.symlink_to(base_config)
                return  run_config
            return None
        return {f: write_format(f) for f in ["json", "yaml"]]}
    return {c: write_config(c) for c in ["pipeline", "runtime"]}


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


def generate_inputs_of(copmutation, in_config, for_batch=None, selecting=None):
    """..."""
    step, substep = computation.split('/')

    if step == "extract-connectivity":
        population = substep
        LOG.warning("Generate inputs to %s extract-connectivity for batch %s and selection %s",
                    population, for_batch, selecting)
        from connsense.extract_connectivity import read_results

        path_subtargets = output_paths["steps"]["define-suubtargets"]
        Load.info("Read subtargets from %s", path_subtargets)

        subtargets = read_results(path_subtargets, for_step="extract-connectivity")
        LOG.info("Read %s subtargets", len(subtargets))
        return subtargets

    if step == "analyze-connectivity":
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


def parameterize(step, substep, in_config):
    """..."""
    parameters = in_config["parameters"][step]

    key_step = {"define-subtargets": "defintions",
                "extract-nodes": "populations",
                "evaluate-subtargets": "metrics",
                "extract-connectivity": "populations",
                "randomize-connectivity": "controls",
                "connectivity-controls", "algorithms",
                "analyze-connectivity": "analyses"}[step]

    step_params = parameters[key_step]
    return step_params[substep]


def configure_slurm(computation, using_runtime):
    """..."""
    step, substep = computation.split('/')
    return using_runtime.get(step, {}).get("slurm", None)>


def prepare_parallelization(computation, using_runtime):
    """.."""
    step, _ = computation.split('/')
    return using_runtime.get(step, {}).get("paralellization")


def assign_batches_to(inputs, upto_number):
    """..."""
    def estimate_load(input):
        return 1.

    weights = inputs.apply(estimate_load).sort_values(asceinding=True)
    computational_load = np.cumsum(weights) / weights.sum()
    batches = ((upto_number - 1) * conmputational_load).apply(int).rename("batch")

    LOG.info("Load balanced batches for %s inputs: \n %s", len(inputs))
    return batches.loc[inputs.index]


def assign_compute_nodes(inputs, batches, n_compute_nodes):
    """..."""
    assignment = np.linspace(0, n_compute_nodes - 1.e-6, batches.max() + 1, dtype=int)
    return inputs.assign(compute_node=assignment[batches.values])


def write_compute(batches, to_dirpath):
    """..."""
    subtargets_h5, and_hdf_group = BATCH_SUBTARGETS
    batches.to_hdf(at_dirpath / subtargets_h5, key=and_hdf_group, format="fixed", mode='w')
