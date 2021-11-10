"""Extract neuron properties."""
from collections.abc import Mapping

import pandas as pd

from bluepy import Circuit

from ..io import read_config as read_cfg
from ..io.write_results import read as read_results, write, default_hdf
from ..io import logging

STEP = "extract-neurons"
LOG = logging.get_logger(STEP)


def extract(circuits, subtargets, params):
    """Run the extractoin for 1 circuit.
    """
    LOG.info("RUN neuron properties extractions")
    if len(params) == 0:
        print("Warning: No properties to extract given. This step will do nothing!")

    circuits = dict([(k, Circuit(v)) for k, v in circuits.items()])

    #circuit_frame = subtargets.index.to_frame().apply(lambda x: circuits[x["circuit"]], axis=1)

    def get_props(index, gids):
        circuit = circuits[index[0]]
        props = circuit.cells.get(gids, properties=params)
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


def run(config, *args, dry_run=False, **kwargs):
    """Launch extraction of  neurons."""
    LOG.warning("Extract neurons for subtargets.")

    cfg = read(config)
    paths = cfg["paths"]

    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "extract-neurons" not in paths:
        raise RuntimeError("No neurons in config!")

    if not dry_run:
        circuits = cfg["paths"]["circuit"]

    path_targets = cfg["paths"]["define-subtargets"]

    LOG.info("READ targets from path %s", path_targets)
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        targets = read_results(path_targets, for_step="define-subtargets")
        LOG.info("DONE read number of targets read: %s", targets.shape[0])

    cfg = cfg["parameters"].get(STEP, {})
    params = cfg.get("properties", [])

    LOG.info("Cell properties to extract: %s", params)
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        extracted = extract(circuits, targets, params)
        LOG.info("DONE, extracting %s", params)

    output = paths.get(STEP, default_hdf(STEP))
    LOG.info("WRITE neuron properties to archive %s\n\t under group %s",
             output[0], output[1])
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        write(extracted, to_path=output, format="table")
        LOG.info("DONE neuron properties to archive.")

    if dry_run:
        LOG.info("DONE dry-run testing the pipeline's plumbing")
    else:
        LOG.warning("DONE extract neurons for %s subtargets", targets.shape[0])

    return f"Saved output {output}"
