"""Connectivity in subtargets."""
import pandas as pd
from pathlib import Path

from .. import plugins
from ..io.write_results import read as read_results, write_toc_plus_payload, default_hdf

from ..io.read_config import check_paths, write as write_config
from ..io import logging

from ..define_subtargets.config import SubtargetsConfig
from ..analyze_connectivity.matrices import get_store

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


def write_adj(matrices, to_output, append=False, format=True, return_config=False):
    """..."""
    hdf, group = to_output
    LOG.info("Write adjacencies like %s", matrices.head())

    write_toc_plus_payload(matrices, (hdf, group), append=append, format=format)
    return (hdf, group)


def write(edges, to_output, append=False, format=True, return_config=False):
    """..."""
    hdf, group = to_output

    output_config = {}
    adj = edges["adj"]
    if adj is not None:
        LOG.info("Write adjacencies like %s", adj.head())
        hdf_adj = (hdf, group+"/adj")
        write_toc_plus_payload(edges["adj"], hdf_adj , append=append, format=format)
        output_config["adj"] = hdf_adj
    else:
        LOG.warning("No adjacency matrices to write.")

    props = edges["props"]
    if props is not None:
        LOG.info("Write edge-properties like %s", props.head())
        hdf_props = (hdf, group+"/props")
        pstore = get_store(*hdf_props, for_matrix_type="pandas.DataFrame", in_mode='a')
        contents = edges["props"].apply(pstore.write)
        update = pstore.prepare_toc(of_paths=contents)
        pstore.append_toc(update)
        output_config["props"] = hdf_props
    else:
        LOG.warning("No edge properties to write.")

    if return_config:
        return output_config
    return write_config(output_config, to_json=Path(hdf).parent/"output.json")


def filter_parallel(batches, at_path):
    """..."""
    from ..pipeline.parallelization import COMPUTE_NODE_SUBTARGETS
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


def extract_subtargets(in_config, population, for_input=None, output=None):
    """Extract connectivity of a population of edges among subtargets.
    """
    LOG.warning("Extract conectivity of subtargets")

    _, output_paths = check_paths(in_config, STEP)

    subtarget_cfg = SubtargetsConfig(in_config)

    subtargets = filter_parallel(batches=for_input, at_path=output_paths["steps"]["define-subtargets"])

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

    count_subtargets = lambda dset: 0 if dset is None else len(dset)
    LOG.warning("DONE, extracting %s subtarget connectivity: adj: %s, props: %s", len(subtargets),
                count_subtargets(connectivity["adj"]), count_subtargets(connectivity["props"]))

    return to_output
