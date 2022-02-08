"""Randomize subtarget connectivity."""

from collections.abc import Mapping
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from .algorithm import SingleMethodAlgorithmFromSource


from ..io.write_results import (read as read_results,
                                read_toc_plus_payload,
                                write_toc_plus_payload,
                                default_hdf)

from ..io import read_config
from ..io import logging

from .randomize import get_neuron_properties, randomize_table_of_contents

STEP = "randomize-connectivity"
LOG = logging.get_logger(STEP)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)


def get_algorithms(config):
    """..."""
    all_parameters = config["parameters"]

    randomize_params = all_parameters[STEP]

    configured = randomize_params["algorithms"]

    LOG.warning("configured algorithms %s", configured )

    return [SingleMethodAlgorithmFromSource(name, description)
            for name, description in configured.items()]


def run(config, parallelize=None, *args, output=None, batch_size=None, sample=None,  dry_run=None,
        **kwargs):
    """..."""
    config = read(config)
    paths = config["paths"]

    if parallelize and STEP in parallelize and parallelize[STEP]:
        LOG.error("NotImplemented yet, parallilization of %s", STEP)
        raise NotImplementedError(f"Parallilization of {STEP}")

    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "extract-neurons" not in paths:
        raise RuntimeError("No neurons in config!")
    if "extract-connectivity" not in paths:
        raise RuntimeError("No connection matrices in config!")
    if STEP not in paths:
        raise RuntimeError("No randomized matrices in config!")

    hdf_path, hdf_group = paths["extract-neurons"]
    LOG.info("Load extracted neuron properties from %s\n\t, group %s",
             hdf_path, hdf_group)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        neurons = get_neuron_properties(hdf_path, hdf_group)
        LOG.info("Done loading extracted neuron properties: %s", neurons.shape)

    hdf_path, hdf_group = paths["extract-connectivity"]
    LOG.info("Load extracted connectivity from %s\n\t, group %s",
             hdf_path, hdf_group)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        toc = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
        LOG.info("Done reading %s table of contents for connectivity matrices",
                 toc.shape)

    LOG.info("DISPATCH randomization of connecivity matrices.")
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        if sample:
            S = np.float(sample)
            toc = toc.sample(frac=S) if S < 1 else toc.sample(n=int(S))
        algorithms = get_algorithms(config)
        randomized = randomize_table_of_contents(toc, neurons, algorithms,
                                                 batch_size)
        LOG.info("Done, randomizing %s matrices.", randomized.shape)

    hdf_path, hdf_key = paths.get(STEP, default_hdf(STEP))
    if output:
        hdf_path = output

    output = (hdf_path, hdf_key)
    LOG.info("Write randomized matrices to path %s.",  output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        output = write_toc_plus_payload(randomized, to_path=output, format="table",)
        LOG.info("Done writing %s randomized matrices: to %s", randomized.shape, output)

    LOG.warning("DONE randomizing: %s", config)
    return f"Result saved {output}"
