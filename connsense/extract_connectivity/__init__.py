"""Extract subtargets' connectivity.
"""
from collections.abc import Mapping
from pathlib import Path

from ..io.write_results import (read as read_results,
                                write_toc_plus_payload as write,
                                default_hdf)
from ..io import read_config
from ..io import logging
from ..io.slurm import SlurmConfig

from connsense.pipeline import workspace

STEP = "extract-edge-populations"
LOG = logging.get_logger(STEP)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)


def cmd_sbatch_extraction(slurm_params, at_path):
    """..."""
    LOG.info("Prepare a Slurm command using sbatch params: \n%s", slurm_params)
    slurm_params["executable"] = "tap-connectivity"
    slurm_config = SlurmConfig(slurm_params)
    return slurm_config.save(to_filepath=at_path/"tap-connectivity.sbatch")


def _remove_link(path):
    try:
        return path.unlink()
    except FileNotFoundError:
        pass
    return None


def get_current(action, mode, config, step, substep, controls=None, with_parallelization=None):
    """..."""
    current_run = workspace.initialize if is_to_init(action) else workspace.current
    return current_run(config, step, substep, controls, mode, with_parallelization)


def run(config, action, substep=None, in_mode=None,  parallelize=None, output=None, **kwargs):
    """..."""
    from connsense.pipeline import workspace

    connectome = substep
    LOG.warning("Extract connectivity of subtargets")

    cfg = read(config)
    input_paths, output_paths = read_config.check_paths(cfg, STEP)

    LOG.warning("Extract subtarget connectivity from connectome %s", connectome)

    stage_dir = workspace.get_rundir(config, STEP, substep)
    basedir = workspace.find_base(stage_dir)

    copy_config = stage_dir.joinpath("config.json")
    _remove_link(copy_config)
    copy_config.symlink_to(basedir/"config.json")

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))

    configured = cfg["parameters"][STEP]["populations"]

    assert connectome in configured, (
        f"Argued connectome {connectome} was not among those configured {configured}")

    assert stage_dir.exists()

    slurm_config = {
        "name": "extract-connectivity",
        "account": "proj83",
        "partition": "prod",
        "time": "24:00:00",
        "constraint": "cpu",
        "venv": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/topological-analysis-subvolumes/test/load_env.sh"
    }

    launchscript = stage_dir / "launchscript.sh"
    with open(launchscript, 'w') as to_launch:
        script = cmd_sbatch_extraction(slurm_config, at_path=stage_dir)

        LOG.info("Created a sbatch script at %s", script)

        def write(aline):
            to_launch.write(aline + '\n')

        write(f"#################### EXTRACT connectivity ######################")

        write(f"pushd {stage_dir}")

        if in_mode:
            write(f"sbatch {script.name} --configure=config.json --mode={in_mode}"
                  f" --connectome={connectome} {action}\n")
        else:
            write(f"sbatch {script.name} --configure=config.json"
                  f" --connectome={connectome} {action}\n")

        write(f"popd")

    return launchscript

    # subtarget_cfg = SubtargetsConfig(cfg)

    # path_targets = output_paths["steps"]["define-subtargets"]
    # LOG.info("READ targets from path %s", path_targets)
    # subtargets = read_results(path_targets, for_step="define-subtargets")
    # LOG.info("DONE read number of targets read: %s", subtargets.shape[0])


    # extracted = run_extraction_from_full_matrix(subtarget_cfg.input_circuit, subtargets, connectomes)

    # to_output = output_specified_in(output_paths, and_argued_to_be=output)
    # write(extracted, to_output, format="table")

    # LOG.warning("DONE, exctracting %s subtarget connectivity", len(extracted))
    # return to_output


def run_0(config, in_mode=None,  parallelize=None, *args, dry_run=False, **kwargs):
    """...
    TODO
    -----
    Use a `rundir` to run extraction of connectivity in.
    """
    paths = config["paths"]

    if parallelize and STEP in parallelize and parallelize[STEP]:
        LOG.error("NotImplemented yet, parallilization of %s", STEP)
        raise NotImplementedError(f"Parallilization of {STEP}")

    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in paths:
        raise RuntimeError("No defined columns in config!")
    if STEP not in paths:
        raise RuntimeError("No connection matrices in config!")

    config = config["parameters"].get("extract-connectivity", {})

    circuits = paths["circuit"]
    path_targets = paths["define-subtargets"]

    LOG.info("Read targets from path %s", path_targets)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        targets = read_results(path_targets, for_step="subtargets")
        LOG.info("Number of targets read: %s", targets.shape[0])

    connectomes = config.get("connectomes", [])
    LOG.info("Extract connevtivity from connectomes: %s", connectomes)
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        extracted = run_extraction_from_full_matrix(circuits, targets, connectomes)
        LOG.info("Done, extraction of %s connectivity matrices.", extracted.shape)

    output = paths.get(STEP, default_hdf(STEP))
    LOG.info("Write extracted matrices to %s\n\t group %s", output[0], output[1])
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        output = write(extracted, to_path=output, format="table")
        LOG.info("Done, writing %s connectivity matrices.", extracted)

    LOG.warning("DONE, extraction of matrices")
    return f"Output saved at {output}"
