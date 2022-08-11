#!/usr/bin/env python3

"""Extract node-types.
"""

from collections.abc import Mapping
from pathlib import Path
from pprint import pformat

import pandas as pd

from ..import plugins
from ..pipeline import workspace, parallelization
from ..io import read_config
from ..io.write_results import read as read_results, default_hdf
from ..io.slurm import SlurmConfig
from ..io import logging
from ..define_subtargets.config import SubtargetsConfig


LOG = logging.get_logger("Extract node types")

STEP = "extract-node-types"


def read_configuration(of_quantity, in_config):
    """..."""
    try:
        modeltype, component = of_quantity.split('/')
    except ValueError as err:
        raise ValueError(f"{STEP} {quantity} should have been of the form <modeltype> / <component>") from err

    return in_config["parameters"][STEP]["modeltypes"][modeltype][component]


def generate_inputs(of_quantity, in_config):
    """..."""
    configured = read_configuration(of_quantity, in_config)
    _, load_inputs = plugins.import_module(configured["inputs"]["source"], configured["inputs"]["method"])

    circuits = SubtargetsConfig(in_config).input_circuit
    inputs = [load_inputs(in_circuit=c) for _, c in circuits.items()]
    return pd.concat(inputs, axis=0, keys=[l for l, _ in circuits.items()], names=["circuit"])


def symlink(config, to_basedir):
    """..."""
    if (to_basedir/config.name).exists():
        _remove_link(config)
        config.symlink_t(to_basedir/config.name)
        return config
    LOG.info("No config %s at base-dir %s", config.name, to_basedir)


def cmd_sbatch_extraction(slurm_params, at_path):
    """..."""
    LOG.info("Prepare a slurm command using sbatch params: \n%s", pformat(slurm_params))
    slurm_params["executable"] = "tap-node-types"
    slurm_config = SlurmConfig(slurm_params)
    return slurm_config.save(to_filepath=at_path/"tap-node-types.sbatch")


def setup(config, substep, subgraphs=None, controls=None, in_mode=None, parallelize=None,
          output=None, batch=None, sample=None, tap=None, **kwargs):
    """..."""
    assert parallelize
    from connsense.pipeline.parallelization.parallelization import run_multinode, setup_compute_node
    return run_multinode(setup_compute_node, computation=f"analyze-node-types/{substep}",
                         in_config=config, using_runtime=parallelize)


def collect(config, substep,  parallelize, subgraphs=None, controls=None, in_mode=None, **kwargs):
    """..."""
    from connsense.pipeline.parallelization import run_multinode, collect_multinode

    modeltype = substep
    assert modeltype

    return run_parallel(process_of=collect_multinode,
                        computation=f"{STEP}/{modeltype}", in_config=config, using_runtime=parallelize,
                        for_control=controls, making_subgraphs=subgraphs)
