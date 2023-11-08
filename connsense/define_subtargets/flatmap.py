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
    to_flatspace = {"nrrd-file-id": "subtarget_id",
                    "grid-i": "flat_i", "grid-j": "flat_j",
                    "grid-x": "flat_x", "grid-y": "flat_y",
                    "grid-subtarget": "subtarget"}
    info_read = (pd.read_hdf(info, key="grid-info").rename(columns=to_flatspace)
                 .set_index("subtarget_id"))
    info_columns = info_read.columns
    coord_columns = ["subtarget", "flat_i", "flat_j", "flat_x", "flat_y"]
    reorder_columns = coord_columns + [c for c in info_columns if c not in coord_columns]
    return info_read[reorder_columns]


def assign_cells(in_circuit, to_subtargets_in_nrrd):
    """..."""
    LOG.info("Load cells with subtarget-id using NRRD: %s", to_subtargets_in_nrrd)
    loader_cfg = {
        "loading": {
            "properties": ["x", "y", "z", "layer", "synapse_class"],
            "atlas": [{"data": to_subtargets_in_nrrd, "properties": ["column-id"]}]
        }
    }
    neurons = (load_group_filter(in_circuit, loader_cfg)
               .rename(columns={"column-id": "subtarget_id"}))
    that_were_assigned_to_voxels = neurons.subtarget_id > 0
    return neurons[that_were_assigned_to_voxels].set_index("subtarget_id").sort_index()


def load_nrrd(circuit, *, path):
    """...Load an NRRD of subtargets."""
    LOG.info("Define subtargets to be loaded from an NRRD")
    cells = assign_cells(circuit, to_subtargets_in_nrrd=path)
    return cells.groupby("subtarget_id").gid.apply(list).rename("gids")


def ensure_list(xs):
    """..."""
    return None if not xs else (list([xs]) if not isinstance(xs, list) else xs)

def conntility_grid(circuit, node_population=None, conntility_loader_cfg=None):
    """..."""
    import numpy as np
    from copy import deepcopy
    import bluepysnap as snap
    from conntility.circuit_models.neuron_groups.grouping_config import load_group_filter

    if conntility_loader_cfg is None:
        raise Exception("Loader Config cannot be empty")

    if node_population is None:
        if len(circuit.nodes.population_names) == 1:
            node_population = circuit.nodes.population_names[0]
        else:
            raise Exception("There are multiple node populations. Specify one")

    assert len(conntility_loader_cfg["grouping"]) == 1
    radii = ensure_list(conntility_loader_cfg["grouping"][0].get("radius", None))

    def configure_radius(r):
        config = deepcopy(conntility_loader_cfg)
        config["grouping"][0]["args"] = [r]
        return config

    if radii:

        subtargets = [(load_group_filter(circuit, configure_radius(r), node_population)
                      .reset_index(drop=True)[["node_ids",
                                               "grid-subtarget", "grid-x", "grid-y"]]
                       .rename(columns={"grid-subtarget": "subtarget",
                                        "grid-x": "grid_x", "grid-y": "grid_y"}))
                      for r in radii]
        columns = (pd.concat(subtargets, axis=0, keys=radii, names=["radius"])
                   .droplevel(None).reset_index()
                   .assign(subtarget=lambda d: d.radius.astype(str) + ";" + d.subtarget))

        info = (columns[["subtarget", "radius", "grid_x", "grid_y"]]
                .drop_duplicates().reset_index(drop=True))
        info.index.rename("subtarget_id", inplace=True)

        colgids = (columns.set_index("subtarget")
                   .join(info.subtarget.reset_index().set_index("subtarget"))
                   .groupby("subtarget_id").node_ids.apply(list))

        return (info.subtarget, info[["radius", "grid_x", "grid_y"]], colgids)

    assert "args" in conntility_loader_cfg["grouping"][0]

    #Get the nodes dataframe based on the loader_config
    df_out = load_group_filter(circuit, conntility_loader_cfg, node_population)

    #Rename some columns and reset index
    result = (df_out.reset_index()
              .rename(columns={'grid-subtarget':'subtarget',
                               'grid-x':'grid_x','grid-y':'grid_y',
                               'grid-i':'grid_i','grid-j':'grid_j'}))

    #Get the members dataset
    members = result.subtarget.drop_duplicates().reset_index(drop=True)
    members.index.rename('subtarget_id', inplace=True)

    #Use the sub_id_map to index the original dataframe output from connitility
    sub_id_map = members.reset_index().set_index('subtarget').subtarget_id

    result['subtarget_id'] =  result['subtarget'].apply(lambda x: sub_id_map[x])

    #Subtargets
    subtargets = result.groupby('subtarget_id')['node_ids'].apply(list).rename('gids')

    #Subtarget_info
    subtarget_info = (result.groupby('subtarget_id')
                      [['x','y','z','grid_x','grid_y','grid_i','grid_j']].mean())


    return (members, subtarget_info, subtargets)
