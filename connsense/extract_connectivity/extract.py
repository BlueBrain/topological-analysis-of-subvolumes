"""Connectivity in subtargets."""
import pandas as pd

from .. import plugins
from ..io.write_results import read as read_results, write_toc_plus_payload, default_hdf

from ..io.read_config import check_paths
from ..io import logging

from ..define_subtargets.config import SubtargetsConfig
from ..analyze_connectivity.matrices import get_store
from ..pipeline.parallelization import COMPUTE_NODE_SUBTARGETS

STEP = "extract-edge-populations"

LOG = logging.get_logger("connsense " + STEP)

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
    adj = edges["adj"]
    if adj:
        LOG.info("Write adjacencies like %s", adj.head())
        hdf, group = to_output
        astore = write_toc_plus_payload(edges["adj"], (hdf, group+"/adj"), format="table")
    else:
        LOG.warning("No adjacency matrices to write.")
        astore = None

    props = edges["props"]
    if props:
        LOG.info("Write edge-properties like %s", props.head())
        pstore = get_store(hdf, group+"/props", for_matrix_type="pandas.DataFrame", in_mode='a')
        contents = edges["props"].apply(pstore.write)
        update = pstore.prepare_toc(of_paths=contents)
        pstore.append_toc(update)
    else:
        LOG.warning("No edge properties to write.")
        pstore = None

    return {"adj": astore, "props": pstore}


def filter_parallel(batches, at_path):
    """..."""
    LOG.info("Read subtargets from %s", at_path)
    among_subtargets = read_results(at_path, for_step="extract-connectivity")

    if batches is None:
        LOG.info("No batches for extract connectivity to filter.")
        return among_subtargets

    _, dataset = COMPUTE_NODE_SUBTARGETS
    assignment = pd.read_hdf(batches, key=dataset)
    subtargets = among_subtargets.loc[assignment.index]
    LOG.info("Done reading a batch of %s / %s subtargets", len(subtargets), len(among_subtargets))
    return subtargets


def extract_subtargets(in_config, population, for_batch=None, output=None):
    """Extract connectivity of a population of edges among subtargets.
    """
    LOG.warning("Extract conectivity of subtargets")

    _, output_paths = check_paths(in_config, STEP)

    subtarget_cfg = SubtargetsConfig(in_config)

    at_path = output_paths["steps"]["define-subtargets"]
    subtargets = filter_parallel(for_batch, at_path)

    parameters = in_config["parameters"][STEP]
    cfgpops = parameters["populations"]

    population = population or "local"
    assert population in cfgpops, f"Argued connectome {population} must be among {cfgpops}"
    LOG.info("Use a configured method to extract connectivity: %s", cfgpops[population])

    extractor = cfgpops[population]["extractor"]
    _, extract = plugins.import_module(extractor["source"], extractor["method"])
    connectivity = extract(subtarget_cfg.input_circuit, cfgpops[population], subtargets)

    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    hdf, group = to_output
    write(connectivity, to_output=(hdf, group+'/'+population))

    LOG.warning("DONE, extracting %s subtarget connectivity: adj: %s, props: %s", len(subtargets),
                len(connectivity["adj"]), len(connectivity["props"]))

    return to_output
