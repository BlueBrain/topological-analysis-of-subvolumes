"""Analyze connectivity.
"""


from collections.abc import Mapping
from collections import OrderedDict
from pathlib import Path
from argparse import ArgumentParser
import tempfile
from pprint import pformat

import h5py
import numpy as np
import pandas as pd
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
from .analyze import BATCHED_SUBTARGETS, parallely_analyze
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


def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["output"]["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, default_hdf(STEP))

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

def __collect_to_remove(batched_stores, in_store):
    """..."""
    LOG.info("Collect %s batched stores together in hdf %s, group %s",
             len(batched_stores), in_store._root, in_store._group)

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


def _read_sample_description(d):
    """..."""
    d_split = d.split('-')
    try:
        by_size, amount = d_split
    except ValueError:
        try:
            amount = np.float(d)
        except ValueError:
            by_size = d.lower(); amount = None
        else:
            by_size = None
    return (by_size, amount)


def sample_subtarget(adjacency_matrices, by_description):
    """..."""
    adjmats = adjacency_matrices

    d = by_description
    by_size, amount = _read_sample_description(d)

    if not by_size:
        S = np.float(amount)
        if S > 1:
            subset = adjmats.sample(n=int(S))
        elif S > 0:
            subset = adjmats.sample(frac=S)
        else:
            raise ValueError(f"Illegal sample {amount}")

        return subset

    def count_nodes(matrix):
        """..."""
        return matrix.value.shape[0]

    if by_size == "largest":
        N = np.int(amount or 1)
        subtarget_sizes = adjmats.apply(count_nodes).sort_values(ascending=False)
        return adjmats.loc[subtarget_sizes.iloc[0:N].index]

    if by_size == "smallest":
        N = np.int(amount or 1)
        subtarget_sizes = adjmats.apply(count_nodes).sort_values(ascending=True)
        return adjmats.loc[subtarget_sizes.iloc[0:N].index]

    raise ValueError(f"Unhandled arguments by_size={by_size}, amount={amount}")


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
        if all_matrices is None:
            LOG.error("No matrices to subset")
            return None
    else:
        all_matrices = toc.rename("matrix")

    if not sample:
        return all_matrices

    return sample_subtarget(all_matrices, by_description=sample)

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

    LOG.warning("configured analyses %s", configured)

    if not as_dict:
        return [SingleMethodAnalysisFromSource(name, description)
                for name, description in configured.items()]
    return {name: SingleMethodAnalysisFromSource(name, description)
            for name, description in configured.items()}


def get_value_store(analysis, at_path, from_cache=None, in_mode='a'):
    """..."""
    if not from_cache:
        to_hdf_at_path, under_group = at_path; m = in_mode
        return matrices.get_store(to_hdf_at_path, under_group + "/" + analysis.name,
                                  for_matrix_type=analysis.output_type, in_mode=m)
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

    if "define-subtargets" not in p["input"]["steps"]:
        raise RuntimeError("No defined columns in config!")

    if "extract-neurons" not in p["input"]["steps"]:
        raise RuntimeError("No neurons in config!")

    if "extract-connectivity" not in p["input"]["steps"]:
        raise RuntimeError("No connection matrices in config!")

    if "randomize-connectivity" not in p["input"]["steps"]:
        raise RuntimeError("No randomized matrices in config paths: {list(paths.keys()}!")

    if STEP not in p["output"]["steps"]:
        raise RuntimeError(f"No {STEP} in config output!")

    return p


def load_neurons(paths, dry_run=False):
    """..."""
    hdf, group = paths["steps"]["extract-neurons"]
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
    hdf, group = paths["steps"]["extract-connectivity"]
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
    hdf, group = paths["steps"]["randomize-connectivity"]
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


def load_adjacencies(paths, from_batch, dry_run=False):
    """..."""
    LOG.info("Load all adjacencies")

    if from_batch is None:
        toc_orig = load_connectivity_original(paths, dry_run)

        toc_rand = load_connectivity_randomized(paths, dry_run)

        if dry_run:
            LOG.info("Test plumbing: analyze: load connectivity")
            return None

        LOG.info("Done loading connectivity.")
        return (toc_orig, toc_rand)

    if isinstance(from_batch, pd.DataFrame):
        return from_batch

    try:
        path = Path(from_batch)
    except TypeError:
        raise TypeError(f"Expecting a path to dataframe, not {from_batch}")

    batches_h5, and_hdf_group = BATCHED_SUBTARGETS
    if path.is_dir():
        path_h5 = path / batches_h5
        if not path_h5.exists():
            raise RuntimeError(f"Missing batch HDF {batches_h5} in dir {path}")
    else:
        path_h5 = path

    return pd.read_hdf(path_h5, and_hdf_group)


def dispatch(adjacencies, neurons, analyses, action=None, in_mode=None, parallelize=None,
             output=None, tap=None, dry_run=False):
    """Dispatch a table of contents of adjacencies, ...

    Computation for an analysis will be run in parallel over the subtargets,
    with results for each batch saved in an independent archive under an
    independent folder for each analysis to compute.
    Once the computation is done, a mapping of batch of the temporary HDF-stores will be
    returned for each analysis.

    Arguments
    ------------
    output :: Tuple(basedir, hdf_group) where
    ~         basedir :: Path to the directory where analysis results will be assembled,
    ~                    i.e. where the temporary HDF for each subtarget analysis will be written
    ~                    into an HDF store, for example `temp.h5` under the specified hdf-group.
    ~                    Example: (<tap-root>/"analysis", "analysis") where
    ~                             <tap-root> is specified in the config as `config[paths][root]`.
    ~                             For an analysis `A` then results for a given batch `B` of
    ~                             subtargets results should go to
    ~                             <tap-root>/"analysis"/<analysis-name>/<batch-index>/"tap.h5"
    ~                             under the HDF group
    ~                             "analysis"/<analysis-name>

    parallelize :: A mapping that describes how to parallelize an analysis.

    More on how parallelization is done in another place.
    """
    LOG.info("DISPATCH analyses of connectivity.")
    if dry_run:
        LOG.info("Test plumbing: analyze: dispatch toc")
        return None

    args = (adjacencies, neurons, action, in_mode, parallelize, tap, output)
    results = {quantity: parallely_analyze(quantity, *args) for quantity in analyses}

    LOG.info("Done, analyzing %s matrices", len(adjacencies))
    return results


def save_output(results, to_path):
    """...
    Save the results if they contain data-stores
    """
    LOG.info("Write analyses results to %s", to_path)

    p, group = to_path

    def in_store(analysis):
        g = f"{group}/{analysis.name}"
        m = analysis.output_type
        return matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)

    def save_analysis(a, batches):
        """..."""




    saved = {a: in_store(a).collect(batched_stores) for a, batched_stores in results.items()}
    LOG.info("Done saving %s analyses of results")
    for a, saved_analysis in saved.items():
        LOG.info("Analysis %s saved %s values", a.name, len(saved_analysis))

    return saved


def run(config, action, in_mode=None, parallelize=None, output=None, batch=None,
        sample=None, tap=None, dry_run=None, **kwargs):
    """..."""
    from connsense.pipeline import workspace

    config = read(config)
    paths = _check_paths(config["paths"])
    input_paths = paths["input"]
    output_paths = paths["output"]

    LOG.warning("DONE analyzing: %s", pformat(config))

    rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)

    neurons = load_neurons(input_paths, dry_run)

    toc_adjs = load_adjacencies(input_paths, batch, dry_run)

    toc_dispatch = subset_subtargets(toc_adjs, sample, dry_run)

    if toc_dispatch is None:
        if not dry_run:
            LOG.warning("Done, with no connectivity matrices available to analyze.")
            return None

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))
    analyses = get_analyses(config)
    basedir = workspace.locate_base(rundir, STEP)
    m = in_mode; p = parallelize.get(STEP, {}) if parallelize else None
    analyzed_results = dispatch(toc_dispatch, neurons, analyses, action, in_mode, parallelize=p,
                                output=(basedir, hdf_group), tap=tap, dry_run=dry_run)

    LOG.warning("DONE %s analyses for TAP config at %s:\n %s", len(analyses), rundir,
                pformat({a.name: len(s) for a, s in analyzed_results.items()}))
    LOG.warning("Run the collection step to gather the parallel computation's results.")
    return analyzed_results


def load_batched_results(analyses, parallelization, output):
    """..."""
    from .analyze import load_parallel_run_analysis
    return {a: load_parallel_run_analysis(a, parallelization, output) for a in analyses}


def collect(config, in_mode, parallelize, *args, output=None, **kwargs):
    """..."""
    from connsense.pipeline import workspace

    config = read(config)
    LOG.info("Collect batched results of analyses of subtargets in config: \n %s",
             pformat(config))

    paths = _check_paths(config["paths"])
    output_paths = paths["output"]
    to_parallelize = parallelize.get(STEP, {}) if parallelize else None

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))
    analyses = get_analyses(config)
    rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)
    basedir = workspace.locate_base(rundir, STEP)
    batched_results = load_batched_results(analyses, to_parallelize, (basedir, hdf_group))

    output = output_specified_in(paths, and_argued_to_be=output)
    saved = save_output(batched_results, to_path=output)

    collected = {a.name: len(s) for a, s in saved.items()}
    LOG.info("DONE collection of batched results at %s: \n %s", rundir, pformat(collected))

    return saved
