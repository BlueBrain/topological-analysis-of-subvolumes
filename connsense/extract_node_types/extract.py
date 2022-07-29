#!/usr/bin/env python3

import pandas as pd

from ..import plugins
from ..io import read_config, logging
from ..pipeline.parallelization import COMPUTE_NODE_SUBTARGETS
from ..define_subtargets.config import SubtargetsConfig
from .import read_configuration, generate_inputs

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


def filter_parallel(batches, at_path=None, among=None):
    """..."""
    assert at_path is not None or among_morphologies is not None
    assert not (at_path is None and among_morphologies is None)

    if among_morphologies is None:
        raise NotImplementedError("Morphological subtargets in the HDFstore\n"
                                  "Must be argued until morphological subtargets have been implmented")

    if batches is None:
        LOG.info("No batches for extract node types to filter.")
        return among_morphologies

    _, dataset = COMPUTE_NODE_SUBTARGETS
    assignment = pd.read_hdf(batches, key=dataset)
    batched = among_morphologies.loc[assignment.index]
    LOG.info("Done reading a batch of %s / %s morphological subtargets", len(batched), len(among_morphologies))
    return batched


def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, None)
    assert under_group

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group)


def extract_node_types(in_config, substep, for_batch=None, output=None):
    """..."""
    LOG.warning("Extract node types")

    _, output_paths = read_config.check_paths(in_config, STEP)

    modeltype, component = substep.split('/')
    circuit_morphologies = generate_inputs(f"{modeltype}/{component}", in_config)
    morphological_subtargets = filter_parallel(for_batch, among=circuit_morphologies)

    circuits = SubtargetsConfig(in_config).input_circuit
    configured = read_configuration(f"{modeltype}/{component}", in_config)
    _, extract = plugins.import_module(configured["extractor"]["source"], configured["extractor"]["method"])
    extracted = extract(circuits, configured["metrics"], morphological_subtargets)

    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    hdf, group = to_output
    write(extracted, to_output=(hdf, f"{group}/{modeltype}/{component}"))

    LOG.info("DONE extracting %s morphological subtargets.", len(extracted))

    return to_output
