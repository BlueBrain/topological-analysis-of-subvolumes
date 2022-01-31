"""Analyze connectivity."""


from collections.abc import Mapping
from collections import OrderedDict
from pathlib import Path
from argparse import ArgumentParser
import tempfile


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


def output_specified_in(paths, and_argued_to_be):
    """..."""
    to_hdf_at_path, under_group = paths.get(STEP, default_hdf(STEP))
    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be
    return (to_hdf_at_path, under_group)


def store(analysis, data, of_type, at_path):
    """..."""
    to_hdf_at_path, under_group = at_path
    store = matrices.get_store(to_hdf_at_path, under_group + "/" + analysis,
                               for_matrix_type=of_type)
                               
    if not store:
        return write(a, data, (to_hdf_at_path, under_group))

    return store.dump(data)


def store_batch(analysis, data, to_hdf_in_file, basedir=None):
    """..."""
    #basedir = tempfile.TemporaryDirectory(dir=Path(basedir)
    #                                      if basedir else Path.cwd())
    basedir = Path(basedir) if basedir else Path.cwd()
    store = matrices.get_store(to_hdf_at_path=basedir/to_hdf_in_file,
                               under_group=analysis.name,
                               for_matrix_type=analysis.output_type)
    if not store:
        raise NotImplementedError("Cannot save batches if analysis of output type %s",
                                  analysis.output_type)
    store.dump(data)
    return (basedir/to_hdf_in_file, analysis.name, analysis.output_type)


def move_hdf(analyzed_results, to_path, and_cleanup_original=False):
    """..."""
    moved = {analysis: store(analysis,
                             data=(matrices.get_store(*with_description)
                                   .toc.apply(lambda lazym: lazym.value)),
                             of_type=with_description[2],
                             at_path=to_path)
             for analysis, with_description in analyzed_results.items()}

    if and_cleanup_original:
        raise NotImplementedError("Please clean up after testing the results")

    return moved


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


def get_value_store(analysis, at_path, from_cache=None):
    """..."""
    if not from_cache:
        to_hdf_at_path, under_group = at_path
        return matrices.get_store(to_hdf_at_path, under_group + "/" + analysis.name,
                                  for_matrix_type=analysis.output_type)
    try:
        store = from_cache[analysis]
    except KeyError:
        store = get_value_store(analysis, at_path)
        from_cache[analysis] = store
    return store


def run(config, *args, output=None, batch_size=None, sample=None,
        njobs=None, dry_run=None, **kwargs):
    """..."""
    config = read(config)
    paths = config["paths"]

    at_path = output_specified_in(paths, and_argued_to_be=output)
    analysis_value_stores = {}

    def store_at_path(analysis, data):
        value_store = get_value_store(analysis, at_path,
                                      from_cache=analysis_value_stores)

        if not value_store:
            return write(analysis, data, (to_hdf_at_path, under_group))
        return value_store.dump(data)

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
        #analysis_with_name = OrderedDict([(a.name, a) for a in analyses])

        toc_dispatch = subset_subtargets(toc_orig, toc_rand, sample)

        if toc_dispatch is None:
            LOG.warning("Done, with no connectivity matrices available to analyze.")
            return None

        analyzed_results = analyze_table_of_contents(toc_dispatch, neurons, analyses,
                                                     store_batch, batch_size, njobs)
        LOG.info("Done, analyzing %s matrices.", len(analyzed_results))


    output = output_specified_in(paths, and_argued_to_be=output)
    LOG.info("Write analyses to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
    else:
        saved = {}
        for batch, analyses_hdf_paths in analyzed_results.items():
            saved[batch] = move_hdf(analyses_hdf_paths, to_path=output)
        n_results = len(analyzed_results)
        LOG.info("Done writing %s analyzed matrices: to %s", n_results, saved)


    LOG.warning("DONE analyzing: %s", config)
    return f"Result saved {output}"


