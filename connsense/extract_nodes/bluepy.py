#!/usr/bin/env python3

"""Extract nodes from a bluepy circuit.
"""
from pprint import pformat

import pandas as pd

from connsense.io import logging
from connsense.define_subtargets.config import attribute_depths

LOG = logging.get_logger("Extract node populations.")


def get_node_depths(circuit):
    LOG.info("RUN neuron depths extraction")
    from flatmap_utility import supersampled_neuron_locations
    #  TODO: Use config-provided flatmap, if possible
    #  TODO: Could offer diffent ways to get the depths values here, such as, using [PH]y
    orient = circuit.atlas.load_data("orientation")
    flatmap = circuit.atlas.load_data("flatmap")
    flat_and_depths = supersampled_neuron_locations(circuit, flatmap, orient, include_depth=True)
    depths = flat_and_depths[["depth"]]
    LOG.info("DONE neuron depths extractions")
    return depths


def extract_node_properties_batch(circuit, subtargets, properties):
    """...Expect subtargets to be lazy."""

    if "depth" in properties:
        LOG.info("Compute depths as node properties")
        circuit_depths = get_node_depths(circuit)
        properties.remove("depth")
    else:
        circuit_depths = None

    def get_gids(subtarget):
        """..."""
        try:
            return subtarget["gids"]
        except KeyError:
            return subtarget
        raise RuntimeError("Python execution must not reach here.")

    def get_props(subtarget_gids):
        """..."""
        props = circuit.cells.get(subtarget_gids, properties=properties)
        if circuit_depths is not None:
            depths = circuit_depths.loc[circuit_depths.index.intersection(subtarget_gids)]
            props = pd.concat([props, depths], axis=1)
        return props.reset_index().rename(columns={"index": "gid"})

    dataframes = subtargets.apply(get_gids).apply(get_props)
    node_properties = (pd.concat(dataframes.values, keys=subtargets.index.values, names=subtargets.index.names)
                       .droplevel(None))
    LOG.info("Extracted node populations: %s, \n%s", node_properties.shape, pformat(node_properties))
    return node_properties


def extract_node_properties_0(circuit, subtarget, properties, depths=None):
    """..."""
    if "depth" in properties:
        LOG.info("Compute depths as node properties for subtarget %s", subtarget)
        circuit_depths = get_node_depths(circuit) if depths is None else depths
    else:
        circuit_depths = None

    props = circuit.cells.get(subtarget, [p for p in properties if p != "depth"])
    if circuit_depths is not None:
        depths = circuit_depths.loc[circuit_depths.index.intersection(subtarget)]
        props = pd.concat([props, depths], axis=1)
    props = props.reset_index().rename(columns={"index": "gid"})
    props.index.name = "node_id"
    return props


def extract_node_properties(circuit, subtarget, properties):
    """..."""
    props = circuit.cells.get(subtarget, [p for p in properties if p != "depth"])

    if "depth" in properties:
        try:
            circuit_cells_depths = circuit.cells.depths
        except AttributeError:
            circuit_cells_depths = attribute_depths(circuit)
            depths = circuit_cells_depths.reindex(subtarget)
        else:
            depths = circuit_cells_depths(subtarget)

        props = pd.concat([props, depths], axis=1)

    props = props.reset_index().rename(columns={"index": "gid"})
    props.index.name = "node_id"
    return props


def collect_node_properties(of_subtargets):
    """..."""
    return pd.concat(of_subtargets.values, keys=of_subtargets.index.values, names=of_subtargets.index.names)
