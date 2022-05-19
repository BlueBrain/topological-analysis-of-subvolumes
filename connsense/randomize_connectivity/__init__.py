"""Randomize subtarget connectivity."""

from collections.abc import Mapping
from pathlib import Path
from argparse import ArgumentParser
from pprint import pformat

import numpy as np
import pandas as pd

from .algorithm import SingleMethodAlgorithmFromSource


from ..io.write_results import (read as read_results,
                                read_toc_plus_payload,
                                write_toc_plus_payload,
                                default_hdf)

from ..io.slurm import SlurmConfig
from ..io import read_config, logging
from ..analyze_connectivity import (default_hdf, read, _check_paths,
                                    load_neurons, load_adjacencies)
from ..analyze_connectivity.randomize import read_random_controls
from ..analyze_connectivity.analyze import (check_basedir, find_base,
                                            configure_launch_multi, read_njobs,
                                            append_batch)

from .randomize import get_neuron_properties, randomize_table_of_contents

STEP = "randomize-connectivity"
LOG = logging.get_logger(STEP)


def read_randomization(config, substep):
    """..."""
    step = config[STEP]
    controls = config[STEP]["controls"]
    try:
        return controls[substep]
    except KeyError:
        pass

    return controls["default"]


def cmd_sbatch_randomization(slurm_params, at_path):
    """..."""
    LOG.info("Prepare a Slurm command using sbatch params: \n%s", slurm_params)
    slurm_params["executable"] = "tap-randomization"
    slurm_config = SlurmConfig(slurm_params)
    return slurm_config.save(to_filepath=at_path / "tap-randomization.sbatch")


def parallely_randomize(controls, subtargets, neurons, action,
                        in_mode=None, to_parallelize=None, to_tap=None, to_save=None):
    """..."""
    rundir, hdf_group = check_basedir(to_save, controls, to_parallelize, mode=action,
                                      controls=None, return_hdf_group=True)

    base = find_base(rundir)
    LOG.info("Checked rundir %s under base at %s", rundir, base)

    assert rundir.exists(), f"Check basedir did not create {base}"

    compute_nodes, n = read_njobs(to_parallelize, for_quantity=controls)
    batched = append_batch(subtargets, using_basedir=rundir, njobs=n)
    to_sbatch = to_parallelize.get(controls.name, {}).get("sbatch", None)

    multirun = configure_launch_multi(compute_nodes, quantity=controls,
                                      using_subtargets=batched, control=None,
                                      at_workspace=(base, rundir), cmd_sbatch=cmd_sbatch_randomization,
                                      action=action, in_mode=in_mode, slurm_config=to_sbatch)
    LOG.info("Multinode randomization run: \n%s", pformat(multirun))
    return multirun


def dispatch(adjacencies, neurons, controls, action, in_mode, parallelize, output, tap=None):
    """..."""
    LOG.info("DISPATCH randomization of connectivity.")

    args = (adjacencies, neurons, action, in_mode, parallelize, tap, output)
    results = parallely_randomize(controls, *args)

    LOG.info("Done, randomizing %s matrices", len(adjacencies))
    return results


def read_parallelization(config):
    """..."""
    step = config.get(STEP, {})
    LOG.info("Read parallelization: \n%s", step)
    return step


def run(config, action, substep=None, in_mode=None, parallelize=None,
        output=None, tap=None, **kwargs):
    """Run an action such as `init, run, continue, merge or collect` on
    TAP step `randomization-connetivity`.
    All the relevant information must be provided in the TAP config.
    """
    from connsense.pipeline import workspace

    assert substep,\
        "Missing argument `substep`: TAP can run only one control's randomization at a time, not all"

    config = read(config)
    input_paths, output_paths = _check_paths(config)

    LOG.warning("%s randomize connectivity using control %s using config:\n%s",
                action, substep, config)

    rundir = workspace.get_rundir(config, mode=in_mode)

    neurons = load_neurons(input_paths)

    randomization = read_randomization(config["parameters"], substep)
    toc_sample = load_adjacencies(input_paths, from_batch=None, return_batches=False,
                                  sample=randomization["subtargets"])

    if toc_sample is None or len(toc_sample) == 0:
        LOG.warning("DONE randomization of connectivity: No matrices were sampled!")
        return None

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))

    controls = read_random_controls(argued=substep, in_config=config)
    LOG.info("Controls to run: %s", pformat([c.name for c in controls.algorithms]))

    basedir = workspace.locate_base(rundir, STEP, create=True)
    m = in_mode; p = read_parallelization(parallelize) if parallelize else None
    randomizations = dispatch(toc_sample, neurons, controls, action, in_mode=m,
                              parallelize=p["controls"], output=(basedir, hdf_group), tap=tap)

    LOG.warning("DONE %s controls for TAPconfig at %s:\n%s", len(controls), rundir, pformat(randomizations))

    LOG.warning("Don't forget to run the collection step to gather the parallel computation's results.")
    return randomizations
