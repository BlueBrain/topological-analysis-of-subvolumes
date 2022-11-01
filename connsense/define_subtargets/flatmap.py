#!/usr/bin/env python3

"""Load subtargets from an NRRD with subtarget info.
"""

from conntility.circuit_models.neuron_groups import load_group_filter
from connsense.io import logging

LOG = logging.get_logger("Define subtargets in circuit flatmap")

import pandas as pd

def read_subtargets(info):
    """..."""
    LOG.info("Read subtarget info from %s", info)
    to_flatspace = {"nrrd-file-id": "subtarget_id", "grid-i": "flat_i", "grid-j": "flat_j",
                    "grid-x": "flat_x", "grid-y": "flat_y", "grid-subtarget": "subtarget"}
    reorder_columns = ["subtarget", "flat_i", "flat_j", "flat_x", "flat_y"]
    return (pd.read_hdf(info, key="grid-info").rename(columns=to_flatspace)
            .set_index("subtarget_id")[reorder_columns])


def assign_cells(in_circuit, to_subtargets_in_nrrd):
    """..."""
    LOG.info("Load cells with subtarget-id using NRRD: %s", to_subtargets_in_nrrd)
    loader_cfg = {"loading": {"properties": ["x", "y", "z", "layer", "synapse_class"],
                              "atlas": [{"data": to_subtargets_in_nrrd, "properties": ["column-id"]}]}}
    neurons = load_group_filter(in_circuit, loader_cfg).rename(columns={"column-id": "subtarget_id"})

    that_were_assigned_to_voxels = neurons.subtarget_id > 0
    return neurons[that_were_assigned_to_voxels].set_index("subtarget_id").sort_index()


def load_nrrd(circuit, *, path):
    """...Load an NRRD of subtargets."""
    LOG.info("Define subtargets to be loaded from an NRRD")
    cells = assign_cells(circuit, to_subtargets_in_nrrd=path)
    return cells.groupby("subtarget_id").gid.apply(list).rename("gids")
