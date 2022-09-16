#!/usr/bin/env python3


"""Extract data on node types.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import numpy as np

from bluepy import Cell

from flatmap_utility import subtargets as flatmap_subtargets
from flatmap_utility.tessellate import TriTille

from ..import plugins
from ..define_subtargets.config import SubtargetsConfig
from ..pipeline import workspace
from ..io.write_results import write, default_hdf
from ..io import read_config, logging
from ..io.read_config import check_paths

STEP = "summarize-connectivity"

LOG = logging.get_logger(STEP)

def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, "summaries/connectivity")

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group)


def read(config):
    """..."""
    try:
        config = read_config.read(config)
    except TypeError:
        assert isinstance(config, Mapping)
    return config


def run(config, substep=None, output=None, **kwargs):
    """..."""
    config = read(config)
    input_paths, output_paths = check_paths(config, STEP)
    LOG.warning("Summarize connectivity %s  inside the circuit %s", substep, input_paths)

    parameters = config["parameters"][STEP]["analyses"][substep]

    sbtcfg = SubtargetsConfig(config)
    circuit_label = parameters["input"]["circuit"]
    circuit = sbtcfg.input_circuit[circuit_label]

    _, load = plugins.import_module(parameters["computation"])

    morphologies = load(circuit).rename("morphologies")
    morphologies.index.rename("subtarget_id", inplace=True)
    morphologies = pd.concat([morphologies], axis=0, keys=[circuit_label], names=["circuit"])

    LOG.info("Defined %s %s-morphoes.", len(morphologies), substep)
    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    out_h5, hdf_group = to_output
    output = write(morphologies, to_path=(out_h5, hdf_group+"/"+substep), format="table")
    LOG.info("DONE: define-subtargets %s %s", substep, output)
    return output
