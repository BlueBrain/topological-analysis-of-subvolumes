"""Connectivity in subtargets."""
from collections.abc import Mapping
import h5py
import numpy
from scipy import sparse
import pandas
from tqdm import tqdm
from bluepy import Circuit

from .. import plugins
from ..io.write_results import read as read_results, write_toc_plus_payload, default_hdf

from ..io.read_config import check_paths
from ..io import logging

from ..define_subtargets.config import SubtargetsConfig
from ..analyze_connectivity.matrices import get_store

def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, default_hdf(STEP))

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group)


def resolve_connectomes(in_argued):
    """..."""
    if isinstance(in_argued, str):
        return [[in_argued]]

    raise NotImplementedError(f"Argued type {type(in_argued)}."
                              " To do when ready to analyze local + mid-range.")


def write(edges, to_output):
    """..."""
    adj = write_toc_plus_payload(edges["adj"], to_output, format="table")

    hdf, group = to_output
    store = get_store(hdf, group, for_matrix_type="pandas.DataFrame", in_mode='a')
    contents = edges["props"].apply(store.write)
    update = store.prepare_toc(of_paths=contents)
    store.append_toc(update)
    return store


def extract_subtargets(in_config, population, for_batch=None, output=None):
    """Extract connectivity of a population of edges among subtargets.
    """
    LOG.warning("Extract conectivity of subtargets")

    input_paths, output_paths = check_paths(in_config, STEP)

    subtarget_cfg = SubtargetsConfig(in_config)

    path_subtargets = output_paths["steps"]["define-subtargets"]
    LOG.info("Read subtargets from %s", path_subtargets)
    subtargets = read_results(path_subtargets, for_step="extract-connectivity")
    LOG.info("Done reading subtargets %s", len(subtargets))

    parameters = in_config["parameters"]["extract-connectivity"]
    cfgpops = parameters["populations"]

    population = population or "local"
    assert population in cfgpops, f"Argued connectome {population} must be among {cfgpops}"
    LOG.info("Use a configured method to extract connectivity: %s", cfgpops[population])

    extractor = cfgpops[population]["extractor"]
    _, extract = plugins.import_module(extractor["source"], extractor["method"])
    connectivity = extract(subtarget_cfg.input_circuit, cfgpops[population], subtargets)

    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    write(connectivity, to_output)

    LOG.warning("DONE, exctracting %s subtarget connectivity", len(extracted))
    return to_output
