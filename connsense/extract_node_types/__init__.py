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
from ..pipeline import PARAMKEY, NotConfiguredError, ConfigurationError, workspace
from ..pipeline.parallelization import parallelization as prl
from ..pipeline.store.store import HDFStore as TapStore
from ..io.write_results import write, default_hdf
from ..io import read_config, logging
from ..io.read_config import check_paths

STEP = "extract-node-types"

MODELTYPES = PARAMKEY[STEP]

LOG = logging.get_logger(STEP)

def output_specified_in(configured_paths, substep, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, "nodes/types")

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group + "/" + substep)


def read(config):
    """..."""
    try:
        config = read_config.read(config)
    except TypeError:
        assert isinstance(config, Mapping)
    return config


def extract_component(variable, component, from_tap, to_output):
    """..."""
    from_circuit = from_tap.get_circuit(component["input"]["circuit"])
    _, extract = plugins.import_module(component["extractor"])

    LOG.info("Extract node types %s from circuit %s to output %s", variable, from_circuit.variant, to_output)

    data = extract(from_circuit)

    connsense_h5, modeltypes = to_output

    def forge(*akey):
        """..."""
        return '/'.join(akey)

    varkey = forge(modeltypes, variable)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data.to_hdf(connsense_h5, key=forge(modeltypes, variable))

    elif isinstance(data, Mapping):
        for labeled, dataset in data.items():
            dataset.to_hdf(connsense_h5, key=forge(varkey, labeled))
    else:
        raise TypeError(f"Cannot extract coomponent {component} data of type {type(component)}")

    return (connsense_h5, varkey)


def read_components(config, modeltype):
    """..."""
    try:
        modeltypes = config["parameters"][STEP][MODELTYPES]
    except KeyError as kerr:
        raise ConfigurationError(f"{STEP} should specify {MODELTYPES}") from kerr

    try:
        read = modeltypes[modeltype]
    except KeyError as kerr:
        raise NotConfiguredError(f"MISSING argued {modeltype} in {STEP} config")

    return {component: description for component, description in read.items() if component != "description"}


def run(config, substep=None, output=None, **kwargs):
    """..."""
    config = read(config)
    input_paths, output_paths = check_paths(config, STEP)
    LOG.warning("Extract node-types %s  inside the circuit %s", substep, input_paths)

    from_tap = TapStore(config)
    to_output = output_specified_in(output_paths, substep, and_argued_to_be=output)

    try:
        modeltype, variable = substep.split('/')
    except ValueError:
        pass
    else:
        among_components_for = read_components(config, modeltype)
        component = among_components_for[variable]
        return extract_component(variable, component, from_tap, to_output)


    among_components = read_components(config, modeltype=substep)
    find_extracted = {variable: extract_component(variable, component, from_tap, to_output)
                      for variable, component in among_components.items()}

    return find_extracted
