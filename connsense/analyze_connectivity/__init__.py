"""Analyze connectivity."""


from collections.abc import Mapping
from collections import OrderedDict
from pathlib import Path
from argparse import ArgumentParser
import tempfile
from pprint import pformat

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

from ..randomize_connectivity.randomize import load_neuron_properties
from .analysis import SingleMethodAnalysisFromSource
from .analyze import (parallely_analyze, analyze_table_of_contents)
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
    hdf, group = to_path
    group += f"/{analysis.name}"

    LOG.info("Write analysis size %s to %s/%s", data.shape, group, analysis.name)

    return write_dataframe(data, to_path=(hdf, group), format=None,
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


def _to_remove_use_store_batch(analysis, data, to_hdf_in_file, basedir=None):
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


def store_analysis(a, at_path):
    """..."""
    hdf, group = at_path

    def store_batch(b):
        """..."""
        p = hdf(b)
        m = a.output_type
        g = f"{group}/{a.name}"
        return matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)

    return store_batch


def collect(batched_stores, in_store):
    """..."""
    LOG.info("Collect %s batched stores together in hdf %s, group %s",
             in_store._root, in_store._group)

    def frame(batch):
        colvars = batch.toc.index.get_level_values(-1).unique()
        colidxname = batch.toc.index.names[-1]
        return pd.concat([batch.toc.xs(d, level=colidxname) for d in colvars],
                         axis=1, keys=list(colvars), names=[colidxname])
    def move(batch):
        """..."""
        framed = frame(batch)
        saved = framed.apply(lambda r: in_store.write(r.apply(lambda l: l.value).dropna()),
                             axis=1)
        update = in_store.prepare_toc(of_paths=saved)
        in_store.append_toc(update)
        return update

    return {b: move(batch) for b, batch in batched_stores.items()}



def subset_subtargets(toc, sample, dry_run):
    """..."""
    if dry_run:
        LOG.info("Test plumbing: analyze_connectivity: subset_subtargets")
        return None

    if isinstance(toc, tuple):
        original, randomized = toc
        all_matrices = ((None if randomized is None else randomized.rename("matrix"))
                        if original is None
                        else (original.rename("matrix") if randomized is None
                              else pd.concat([original, randomized]).rename("matrix")))
    else:
        all_matrices = toc.rename("matrix")

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


def _check_paths(p):
    """..."""
    if "circuit" not in p:
        raise RuntimeError("No circuits defined in config!")

    if "define-subtargets" not in p["input"]:
        raise RuntimeError("No defined columns in config!")

    if "extract-neurons" not in p["input"]:
        raise RuntimeError("No neurons in config!")

    if "extract-connectivity" not in p["input"]:
        raise RuntimeError("No connection matrices in config!")

    if "randomize-connectivity" not in p["input"]:
        raise RuntimeError("No randomized matrices in config paths: {list(paths.keys()}!")

    if STEP not in p["output"]:
        raise RuntimeError(f"No {STEP} in config!")

    return p


def load_neurons(paths, dry_run=False):
    """..."""
    hdf, group = paths["extract-neurons"]
    LOG.info("Load neuron properties from %s/%s", hdf, group)

    if dry_run:
        LOG.info("Test plumbing: analyze: load neurons")
        return None

    neurons = load_neuron_properties(hdf, group)
    LOG.info("Done loading extracted neuron properties: %s", neurons.shape)
    return neurons


def load_connectivity_original(paths, dry_run=False):
    """Add connectivity TOC from the HDF store and append a level to
    the TOC index to indicate that it is the original connecitity.
    This is needed to concat original and randomized connectivities.
    """
    hdf, group = paths["extract-connectivity"]
    LOG.info("Load original connectivity from %s/%s", hdf, group)

    if dry_run:
        LOG.info("Test plumbing: analyze : load original connectivity")
        return None

    try:
        toc = read_toc_plus_payload((hdf, group), STEP).rename("matrix")
    except KeyError:
        LOG.warning("No original connectivity data in store %s/%s", path, group)
        toc_orig = None
    else:
        toc_orig = pd.concat([toc], keys=["original"], names=["algorithm"])
        LOG.info("Done loading  %s original connectivity TOC", toc_orig.shape)

    return toc_orig


def load_connectivity_randomized(paths, dry_run):
    """..."""
    hdf, group = paths["randomize-connectivity"]
    LOG.info("Load randomized connectivity from %s / %s", hdf, group)

    if dry_run:
        LOG.info("Test plumbing: analyze: load randomized connectivity")
        return None

    try:
        toc_rand = read_toc_plus_payload((hdf, group), STEP).rename("matrix")
    except KeyError:
        LOG.warning("No randomized-connectivity data in the store.")
        toc_rand = None
    else:
        LOG.info("Done loading  %s randomized connectivity TOC", toc_rand.shape)

    return toc_rand


def load_adjacencies(paths, dry_run=False):
    """..."""
    LOG.info("Load all adjacencies")

    toc_orig = load_connectivity_original(paths, dry_run)

    toc_rand = load_connectivity_randomized(paths, dry_run)

    if dry_run:
        LOG.info("Test plumbing: analyze: load connectivity")
        return None

    LOG.info("Done loading connectivity.")
    return (toc_orig, toc_rand)


def dispatch(adjacencies, neurons, analyses, batch_size, njobs,
             output=None, dry_run=False):
    """Dispatch a table of contents of adjacencies, ..."""
    LOG.info("DISPATCH analyses of connectivity.")
    if dry_run:
        LOG.info("Test plumbing: analyze: dispatch toc")
        return None

    args = (adjacencies, neurons, output, batch_size, njobs, output)
    results = {quantity: parallely_analyze(quantity, *args) for quantity in analyses}

    LOG.info("Done, analyzing %s matrices", len(adjacencies))
    return results


def save_output(results, to_path):
    """..."""
    LOG.info("Write analyses results to %s", to_path)

    p, group = to_path

    def in_store(analysis):
        g = f"{group}/{analysis.name}"
        m = analysis.output_type
        return matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)

    saved = {a: collect(batched_stores, in_store(a)) for a, batched_stores in results.items()}
    LOG.info("Done saving %s analyses of results")
    for a, saved_analysis in saved.items():
        LOG.info("Analysis %s saved %s values", a.name, len(saved_analysis))

    return saved


def _prepare_workspace(paths):
    """..."""
    from connsense.io import time as timing

    batches = Path(paths["root"]) / "batches"
    batches.mkdir(parents=False, exist_ok=True)

    workspace = batches / timing.stamp(now=True)
    workspace.mkdir(parents=False, exist_ok=True)

    analyses = workspace / "analysis"
    analyses.mkdir(parents=False, exist_ok=False)

    return analyses


def run(config, *args, output=None, batch_size=None, sample=None,
        njobs=None, dry_run=None, **kwargs):
    """..."""
    config = read(config)
    paths = _check_paths(config["paths"])
    input_paths = paths["input"]
    output_paths = paths["output"]

    # REMOVE the following ---  it is not used and has undefined variables
    # def store_at_path(analysis, data):
    #     value_store = get_value_store(analysis, at_path,
    #                                   from_cache=analysis_value_stores)

    #     if not value_store:
    #         return write(analysis, data, (to_hdf_at_path, under_group))
    #     return value_store.dump(data)

    neurons = load_neurons(input_paths, dry_run)

    toc_adjs = load_adjacencies(input_paths, dry_run)
    toc_dispatch = subset_subtargets(toc_adjs, sample, dry_run)

    if toc_dispatch is None:
        if not dry_run:
            LOG.warning("Done, with no connectivity matrices available to analyze.")
            return None

    _, hdf_group = output_paths.get(STEP, default_hdf(STEP))
    basedir = _prepare_workspace(paths)
    analyses = get_analyses(config)
    analyzed_results = dispatch(toc_dispatch, neurons, analyses, batch_size, njobs,
                                (basedir, hdf_group), dry_run)

    output = output_specified_in(output_paths, and_argued_to_be=output)
    LOG.info("Write analyses to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing")
        return "TESTED: Pipeline plumbing in place: TESTED"

    saved = save_output(analyzed_results, to_path=output)

    LOG.warning("DONE analyzing: %s", config)
    return f"Result saved {output}"
