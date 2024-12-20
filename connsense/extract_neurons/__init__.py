"""Extract neuron properties."""
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import numpy

from bluepy import Circuit

from ..io import read_config as read_cfg
from ..io.read_config import check_paths
from ..io.write_results import read as read_results, write, default_hdf
from ..io import logging

from ..define_subtargets.config import SubtargetsConfig

STEP = "extract-neurons"
LOG = logging.get_logger(STEP)
NEURON_XYZ = ["x", "y", "z"]


def get_neuron_depths(circuit):
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


def extract(circuits, subtargets, params):
    """Run the extractoin for 1 circuit.
    """
    LOG.info("RUN neuron properties extractions")
    if len(params) == 0:
        print("Warning: No properties to extract given. This step will do nothing!")

    circuits = {k: c if isinstance(c, Circuit) else Circuit(c) for k, c in circuits.items()}

    # TODO: find a better way
    if "depth" in params:
        LOG.info("Compute depths as neuron properties")
        depths = dict([(k, get_neuron_depths(v)) for k, v in circuits.items()])
        params.remove("depth")
        include_depth = True
    else:
        include_depth = False

    #circuit_frame = subtargets.index.to_frame().apply(lambda x: circuits[x["circuit"]], axis=1)

    def get_props(index, gids):
        circuit = circuits[index[0]]
        props = circuit.cells.get(gids, properties=params)
        if include_depth:
            circ_depth = depths[index[0]]
            nrn_depths = circ_depth.loc[circ_depth.index.intersection(gids)]
            props = pd.concat([props, nrn_depths], axis=1)  # Should fill missing gids with NaN
        props.index = pd.MultiIndex.from_tuples([index + (gid,) for gid in gids],
                                                names=["circuit", "subtarget",
                                                       "flat_x", "flat_y", "gid"])
        return props

    neuron_properties = pd.concat([get_props(index, gids) for index, gids in subtargets.iteritems()])
    LOG.info("DONE neuron properties extractions: %s", neuron_properties.shape)

    return neuron_properties


def read(config):
    """..."""
    try:
        return read_cfg.read(config)
    except TypeError:
        pass

    assert isinstance(config, Mapping)
    return config


def _resolve_hdf(location, paths):
    """..."""
    path, group = paths.get(STEP, default_hdf(STEP))
    LOG.info("resolve HDF paths for %s, %s", location, (path, group))


    if location:
        try:
            path = Path(location)
        except TypeError as terror:
            LOG.info("No path from %s, \n\t because %s", output, terror)
            try:
                hdf_path, hdf_group = location
            except (TypeError, ValueError):
                raise ValueError("output should be a tuple(hdf_path, hdf_group)"
                                 "Found %s", output)

    return (path,group)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)


def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, default_hdf(STEP))

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group)


def run(config, in_mode=None, parallelize=None, output=None, **kwargs):
    """Launch extraction of  neurons.

    TODO
    ---------
    Use a `rundir` to run extraction of neurons in.
    """
    LOG.warning("Extract neurons for subtargets.")

    if parallelize and STEP in parallelize and parallelize[STEP]:
        LOG.error("NotImplemented yet, parallilization of %s", STEP)
        raise NotImplementedError(f"Parallilization of {STEP}")

    cfg = read(config)
    input_paths, output_paths = check_paths(cfg, STEP)

    subtarget_cfg = SubtargetsConfig(cfg)

    if "circuit" not in cfg["paths"]:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in input_paths["steps"] or "define-subtargets" not in output_paths["steps"]:
        raise RuntimeError("Missing subtarget definitions in config.")
    if "extract-neurons" not in input_paths["steps"] or "extract-neurons" not in output_paths["steps"]:
        raise RuntimeError("Missing neuron extraction in config!")

    #path_targets = cfg["paths"]["define-subtargets"]
    path_targets = output_paths["steps"]["define-subtargets"]

    LOG.info("READ targets from path %s", path_targets)
    subtargets = read_results(path_targets, for_step="define-subtargets")
    LOG.info("DONE read number of targets read: %s", subtargets.shape[0])

    cfg = cfg["parameters"].get(STEP, {})
    params = cfg.get("properties", [])

    LOG.info("Cell properties to extract: %s", params)
    extracted = extract(subtarget_cfg.input_circuit, subtargets, params)
    LOG.info("DONE, extracting %s", params)

    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    LOG.info("WRITE neuron properties to archive %s\n\t under group %s",
             to_output[0], to_output[1])
    write(extracted, to_path=to_output, format="table")
    LOG.info("DONE neuron properties to archive.")

    LOG.warning("DONE extract neurons for subtargets, with %s neurons in a dataframe", extracted.shape[0])

    return f"Saved output {to_output}"
