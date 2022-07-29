#!/usr/bin/env python3

import pandas as pd

from ..import plugins
from ..io import read_config, logging

STEP = "extract-node-types"

LOG = logging.get_logger(STEP)


def extract_component(c, parameters):
    """..."""
    module, method = plugins.import_module(parameters["extractor"]["source"], parameters["extractor"]["method"])
    return method(parameters["metrics"])


def write(modeltype, components, to_path):
    """..."""
    connsense_h5, group = to_path

    def write_component(c, extraction):
        dataset = '/'.join([group, modeltype, component])
        extraction.to_hdf(connsense_h5, key=dataset)

    return {c: write_component(c, extraction) for c, extraction in components.itemns()}


def extract_node_types(in_config, substep, for_batch=None, output=None):
    """..."""
    LOG.warning("Extract node types")

    _, output_paths = read_config.check_paths(in_config, STEP)

    configured = in_config["parameters"][modeltype]

    components = {c: extract_component(c, parameters) for c, parameters in configured.items()}

    return write(modeltype, components, to_path)
