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


def sample_by_size(nmin, nmax, subtargets):
    """Sample number of subtargets with size in [nmin, nmax)
    """
    def _(subtarget_sizes):
        """..."""
        in_range = subtarget_sizes[np.logical_and(nmin <= subtarget_sizes, subtarget_sizes < nmax)]
        if len(in_range) <= subtargets:
            return in_range

        return in_range.sample(n=subtargets)

    return _


def sample_subtargets(adjacency_matrices, by_description):
    """...
    sample may be described as:
    1. A list of dicts, with each entry describing subtargets to be selected.
    ~  Results will be concatenated.
    2. A string description such as 'largest-1'...
    3. A float less than 1 providing the fraction to sample, or a n int giving the number to sample

    TODO: Generalize the description provided as a list.
    """
    if isinstance(by_description, list):
        LOG.info("Sample subtargets among %s by description: \n%s",
                 len(adjacency_matrices), pformat(by_description))
        samples = [sample_by_size(**s) for s in by_description]
        subtarget_sizes = adjacency_matrices.apply(lambda m: m.get_value().shape[0])
        by_size = pd.concat([sample(subtarget_sizes) for sample in samples])
        return adjacency_matrices.reindex(by_size.index)

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
        return matrix.get_value().shape[0]

    if by_size == "largest":
        N = np.int(amount or 1)
        subtarget_sizes = adjmats.apply(count_nodes).sort_values(ascending=False)
        return adjmats.loc[subtarget_sizes.iloc[0:N].index]

    if by_size == "smallest":
        N = np.int(amount or 1)
        subtarget_sizes = adjmats.apply(count_nodes).sort_values(ascending=True)
        return adjmats.loc[subtarget_sizes.iloc[0:N].index]

    raise ValueError(f"Unhandled arguments by_size={by_size}, amount={amount}")


def subset_subtargets(toc_plus, sample, dry_run=False):
    """..."""
    if isinstance(toc_plus, tuple):
        toc, batches = toc_plus
    else:
        toc = toc_plus; batches = None

    LOG.info("Subset %s subtargets sampling %s descriptions", 0 if toc is None else len(toc), sample)
    if dry_run:
        LOG.info("Test plumbing: analyze_connectivity: subset_subtargets")
        return None

    if toc is None:
        return None

    all_matrices = toc.rename("matrix")

    if not sample:
        return all_matrices

    toc_sample = sample_subtargets(all_matrices, by_description=sample)
    return (toc_sample, batches.reindex(toc_sample.index)) if batches else toc_sample


def get_analyses(config, names=False, as_dict=False):
    """..."""
    all_parameters = config["parameters"]

    analyze_params = all_parameters[STEP]

    configured = analyze_params["analyses"]

    LOG.warning("configured analyses %s", configured)

    if names:
        return list(configured.keys())

    if not as_dict:
        return [SingleMethodAnalysisFromSource(name, description)
                for name, description in configured.items()]
    return {name: SingleMethodAnalysisFromSource(name, description)
            for name, description in configured.items() if name != "COMMEMT"}


def filter_analyses(ns, substep):
    """Filter an analyze-connectivity substep --- provided at the CLI.
    """
    analyses = ns
    if not substep:
        return analyses

    try:
        substep_analysis = analyses[substep]
    except KeyError as kerr:
        raise KeyError(f"analyze-connectivity <substep> {substep}"
                       " must be missing in the config.") from kerr
    return [substep_analysis]


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


def _check_paths(in_config):
    """..."""
    p = in_config["paths"]

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

    return (p["input"], p["output"])


def load_neurons(paths, dry_run=False):
    """..."""
    hdf, group = paths["steps"]["extract-neurons"]
    LOG.info("Load neuron properties from %s/%s", hdf, group)

    if dry_run:
        LOG.info("Test plumbing: analyze: load neurons")
        return None

    neurons = (read_results((hdf, group), STEP)
               .reset_index().set_index(["circuit", "subtarget", "flat_x", "flat_y"]))

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


def load_adjacencies(paths, from_batch=None, return_batches=True, sample=None):
    """Adjacencies can be loaded from the HDF-store specified in the paths,
    or from path to a batch that contains a store and a JSON file to control
    the adjacency before an analysis.

    Return
    ---------
    If not from batch, the original table of contents (TOC)

    If from batch, the original TOC's slice for that batch,
    along with the `batch` that assign a compute batch to each subtarget if so argued
    """
    LOG.info("Load all adjacencies")

    toc_orig = load_connectivity_original(paths).rename("matrix")

    if from_batch is None:
        assert not return_batches, "Cannot return batches if not argued as `from_batch`!"
        LOG.info("Done loading connectivity.")
        return subset_subtargets(toc_orig, sample)

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

    batches =  pd.read_hdf(path_h5, and_hdf_group)
    LOG.info("Done loading batched connectivity with %s entries: \n%s.", len(batches), pformat(batches))

    toc = toc_orig.reindex(batches.index)
    return subset_subtargets((toc, batches) if return_batches else toc, sample)


def dispatch(adjacencies, neurons, analyses, action=None, in_mode=None, controls=None,
             parallelize=None, output=None, tap=None, dry_run=False):
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

    args = (adjacencies, neurons, action, in_mode, controls, parallelize["analyses"], tap, output)
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

    saved = {a: in_store(a).collect(batched_stores) for a, batched_stores in results.items()}
    LOG.info("Done saving %s analyses of results")
    for a, saved_analysis in saved.items():
        LOG.info("Analysis %s saved %s values", a.name, len(saved_analysis))

    return saved


def read_controls(configured, argued):
    """..."""
    from .randomize import read_random_controls

    if not argued:
        LOG.info("No controls were argued.")
        return None

    LOG.info("Apply %s controls to a table of contents.", argued)

    return read_random_controls(argued, in_config=configured)


def apply_controls(configured, toc, log_info=None, **kwargs):
    """Apply configured controls to an adjacency table ot contents, as argued in  `kwargs``.

    A control is configured as :
       "erdos-renyi": {
          "source": "/path/to/the/source/file.py",
          "method": "name-of the shuffling method in the source file",
          "number_samples": 5,
          "seeds": [0, 1, 2, 3, 4, 5],
          "kwargs": {}
        }


    NOTE: This may not apply anymore. Keep this until no need as a reference
    """
    controls = read_controls(configured, **kwargs)
    if controls is None:
        return toc

    controlled = toc.droplevel("algorithm").apply(controls)

    return pd.concat([v for _, v in controlled.items()], keys=controlled.columns)


def read_parallelization(config):
    """..."""
    step = config.get(STEP, {})
    return step


def run(config, action, substep=None, controls=None, in_mode=None, parallelize=None,
        output=None, batch=None, sample=None, tap=None, dry_run=None, **kwargs):
    """..."""
    from connsense.pipeline import workspace

    assert substep,\
        "Missing argument `substep`: TAP can run only one argued analysis, not all at once!"

    config = read(config)
    input_paths, output_paths = _check_paths(config)

    LOG.warning("%s analyze connectivity %s using config:\n%s", action.capitalize(), substep, config)

    rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)

    neurons = load_neurons(input_paths, dry_run)

    toc_dispatch = load_adjacencies(input_paths, batch, return_batches=False, sample=sample)

    if toc_dispatch is None:
        if not dry_run:
            LOG.warning("Done, with no connectivity matrices available to analyze.")
            return None

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))

    configured = get_analyses(config, as_dict=True)
    LOG.info("Analyses in the configuration %s", pformat(configured.keys()))
    analyses = filter_analyses(configured, substep)
    LOG.info("Analyses to run %s", pformat(analyses))

    basedir = workspace.locate_base(rundir, STEP)
    m =in_mode; p = read_parallelization(parallelize) if parallelize else None
    analyzed_results = dispatch(toc_dispatch, neurons, analyses, action, in_mode,
                                controls=read_controls(config, controls),
                                parallelize=p,
                                output=(basedir, hdf_group),
                                tap=tap, dry_run=dry_run)

    LOG.warning("DONE %s analyses for TAP config at %s:\n %s", len(analyses), rundir,
                pformat({a.name: len(s) for a, s in analyzed_results.items()}))
    LOG.warning("Run the collection step to gather the parallel computation's results.")
    return analyzed_results


def load_batched_results(analyses, controls, parallelization, output):
    """..."""
    from .analyze import load_parallel_run_analysis
    return {a: load_parallel_run_analysis(a, controls, parallelization, output) for a in analyses}


def collect(config, in_mode, parallelize, substep=None, controls=None, output=None, **kwargs):
    """Collect batched results into a single store.

    substep :: Name of the analysis to store that is provided at the CLI as analyze-connectivity substep.
    ~         If `None` is provided, all the analyses configured will be run.
    """
    from connsense.pipeline import workspace

    config = read(config)
    LOG.info("Collect batched results of analyses of subtargets in config: \n %s",
             pformat(config))

    controls = read_controls(config, controls)
    if controls:
        LOG.info("Collect batched results of analyses of subtargets for controls: \n %s",
                 pformat(controls))

    _, output_paths = _check_paths(config)
    to_parallelize = parallelize.get(STEP, {}) if parallelize else None

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))
    configured = get_analyses(config, as_dict=True)
    analyses = filter_analyses(configured, substep)
    LOG.info("Collect analyses %s", pformat([a.name for a in analyses]))

    rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)
    basedir = workspace.locate_base(rundir, STEP)
    batched_results = load_batched_results(analyses, controls, to_parallelize, (basedir, hdf_group))

    LOG.info("Loaded results : \n%s ", pformat(batched_results))

    output = output_specified_in(paths, and_argued_to_be=output)
    saved = save_output(batched_results, to_path=output)

    collected = {a.name: len(s) for a, s in saved.items()}
    LOG.info("DONE collection of batched results at %s: \n %s", rundir, pformat(collected))

    return saved
