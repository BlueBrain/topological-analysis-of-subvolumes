"""Analyze connectivity."""


from collections.abc import Mapping
from pathlib import Path
from argparse import ArgumentParser


import pandas as pd
import numpy as np
from scipy import sparse

from ..io.write_results import (read as read_results,
                                read_toc_plus_payload,
                                write as write_dataframe,
                                write_toc_plus_payload,
                                default_hdf)

from ..io import read_config
from ..io import logging

from ..randomize_connectivity.randomize import get_neuron_properties
from .analysis import SingleMethodAnalysisFromSource
from .analyze import analyze_table_of_contents
from .import matrices


STEP = "analyze-connectivity"
LOG = logging.get_logger(STEP)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)


def write(analysis, data, to_path):
    """..."""
    hdf_path, hdf_group = to_path
    hdf_group += f"/{analysis.name}"

    LOG.info("Write analysis size %s to %s/%s",
             data.shape, hdf_group, analysis.name)

    return write_dataframe(data, to_path=(hdf_path, hdf_group),
                           format=None,
                           metadata=analysis.description)


def store(analysis, data, to_hdf_at_path, under_group):
    """..."""
    store = matrices.get_store(to_hdf_at_path, under_group + "/" + analysis.name,
                               for_matrix_type=analysis.output_type)
                               
    if not store:
        return write(a, data, (to_hdf_at_path, under_group))

    return store.dump(data)

def subset_subtargets(original, randomized, sample):
    """..."""
    all_matrices = ((None if randomized is None else randomized.rename("matrix"))
                    if original is None
                     else (original.rename("matrix") if randomized is None
                           else pd.concat([original, randomized]).rename("matrix")))
    if all_matrices is None:
        LOG.error("No matrices to subset")
        return None

    if not sample:
        return all_matrices

    S = np.float(sample)
    if S > 1:
        subset = all_matrices.sample(n=int(S))
    elif S > 0:
        subset = all_matrices.sample(frac=S)
    else:
        raise ValueError(f"Illegal sample={sample}")

    return subset


def get_analyses(config, as_dict=False):
    """..."""
    all_parameters = config["parameters"]

    analyze_params = all_parameters[STEP]

    configured = analyze_params["analyses"]

    LOG.warning("configured analyses %s", configured )

    if not as_dict:
        return [SingleMethodAnalysisFromSource(name, description)
                for name, description in configured.items()]
    return {name: SingleMethodAnalysisFromSource(name, description)
            for name, description in configured.items()}


def run(config, *args, output=None, batch_size=None, sample=None,  dry_run=None,
        **kwargs):
    """..."""
    config = read(config)
    paths = config["paths"]

    if "circuit" not in paths:
        raise RuntimeError("No circuits defined in config!")
    if "define-subtargets" not in paths:
        raise RuntimeError("No defined columns in config!")
    if "extract-neurons" not in paths:
        raise RuntimeError("No neurons in config!")
    if "extract-connectivity" not in paths:
        raise RuntimeError("No connection matrices in config!")
    if "randomize-connectivity" not in paths:
        raise RuntimeError("No randomized matrices in config paths: {list(paths.keys()}!")
    if STEP not in paths:
        raise RuntimeError(f"No {STEP} in config!")

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
        try:
            toc = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
        except KeyError:
            LOG.warning("No original connectivity data in the store.")
            toc_orig = None
        else:
            toc_orig = pd.concat([toc], keys=["original"], names=["algorithm"])
            LOG.info("Done loading  %s table of contents of original connectivity matrices",
                     toc_orig.shape)

    hdf_path, hdf_group = paths["randomize-connectivity"]
    LOG.info("Load randomized connectivity from %s\n\t, group %s",
             hdf_path, hdf_group)
    if dry_run:
        LOG.info("Test pipeline plumbing")
    else:
        try:
            toc_rand = read_toc_plus_payload((hdf_path, hdf_group), STEP).rename("matrix")
        except KeyError:
            LOG.warning("No randomized-connectivity data in the store.")
            toc_rand = None
        else:
            LOG.info("Done loading  %s table of contents of randomized connevitiy matrices",
                     toc_rand.shape)

    LOG.info("DISPATCH analyses of connectivity.")
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        analyses = get_analyses(config)
        lookup_analysis = {a.name: a for a in analyses}

        toc_dispatch = subset_subtargets(toc_orig, toc_rand, sample)

        if toc_dispatch is None:
            LOG.warning("Done, with no connectivity matrices available to analyze.")
            return None

        analyzed = analyze_table_of_contents(toc_dispatch, neurons, analyses,
                                             batch_size)
        LOG.info("Done, analyzing %s matrices.", len(analyzed))


    to_hdf_at_path, under_group = paths.get(STEP, default_hdf(STEP))
    if output:
        to_hdf_at_path = output

    output = (to_hdf_at_path, under_group)

    LOG.info("Write analyses to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        for a, data in analyzed.items():
            analysis = lookup_analysis[a]
            store(analysis, data, to_hdf_at_path, under_group)
            #write(analysis, data, to_path=(hdf_path, hdf_group))
        output = hdf_path
        LOG.info("Done writing %s analyzed matrices: to %s", len(analyzed), output)

    LOG.warning("DONE analyzing: %s", config)
    return f"Result saved {output}"
