"""Analyze connectivity of subtargets
"""
import sys
import os
from pprint import pformat
from multiprocessing import Process, Manager
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import  Analysis
from ..io.read_config import write as write_config
from ..io.write_results import read as read_results
from ..io import logging
from .import matrices

STEP = "analyze-connectivity"

LOG = logging.get_logger(STEP)


def get_neuron_properties(hdf_path, hdf_group):
    """..."""
    return (read_results((hdf_path, hdf_group), STEP)
            .droplevel(["flat_x", "flat_y"])
            .reset_index()
            .set_index(["circuit", "subtarget"]))


def apply(analyses, to_batch, using_neurons,
          n_batches=None, label=None, bowl=None):
    """..."""
    LOG.info("ANALYZE %s \t batch %s / %s with %s targets",
             [a.name for a in analyses],
             label, n_batches, to_batch.shape[0])

    def get_neurons(row):
        """..."""
        index = dict(zip(to_batch.index.names, row.name))
        return (using_neurons.loc[index["circuit"], index["subtarget"]]
                .reset_index(drop=True))

    n_analyses = len(analyses)

    def apply(analysis, at_index):
        """..."""
        LOG.info("Apply analysis %s to batch %s", analysis.name, label)

        memory_used = 0
        def to_row(r):
            """..."""
            nrows = to_batch.shape[0]
            log_info = (f"Batch {label} Analysis {analysis.name}"
                        f" ({at_index} / {n_analyses})"
                        f" matrix {r.idx} / {nrows}")
            result = analysis.apply(r.matrix, get_neurons(r), log_info)
            memory_result = sys.getsizeof(result)
            memory_used += memory_result
            LOG.info("\t\t\t MEMORY USAGE"
                     " for analysis %s batch %s matrix %s / %s: %s / %s",
                     analysis.name, label, r.idx, nrows, memory_result, memory_used)

        n_batch = to_batch.shape[0]
        return to_batch.assign(idx=range(n_batch)).apply(to_row, axis=1)

    analyzed = {a.name: apply(a, i) for i, a in enumerate(analyses)}

    LOG.info("DONE batch %s / %s with %s targets, columns %s: analyzed %s",
             label, n_batches, batch.shape[0], batch.columns, len(analyzed))
    if bowl:
        assert label
        bowl[label] = analyzed
    return analyzed



GIGABYTE = 2 ** 30

BATCHED_SUBTARGETS = "batched_subtargets.h5"


def subset_subtarget(among_neurons):
    """..."""
    def _(circuit, subtarget):
        """..."""
        return among_neurons.loc[circuit, subtarget].reset_index(drop=True)
    return _


def getsizeof_measurement(result):
    """..."""
    if isinstance(result, np.ndarray):
        in_bytes = result.nbytes

    elif isinstance(result, pd.Series):
        in_bytes = result.memory_usage(index=True, deep=True)

    elif isinstance(result, pd.DataFrame):
        in_bytes = result.memory_usage(index=True, deep=True).sum()

    else:
        in_bytes = sys.getsizeof(result)

    return in_bytes / GIGABYTE


def measure_quantity(a, of_subtarget, index_entry=None, using_neuron_properties=None,
                     batch_size=None, log_info=None):
    """Apply an analysis to a subtarget in row of a batch.
    """
    analysis = a
    neurons = using_neuron_properties
    s = of_subtarget
    i = s.idx + 1
    l = index_entry["subtarget"]
    LOG.info("Apply analysis %s to subtarget %s (%s / %s) %s",
             analysis.name, l, i, batch_size or "",
             log_info or "")

    result = analysis.apply(s.matrix, neurons, log_info)

    return result


def apply_analysis(a, to_batch, among_neurons, in_mode=None, using_store=None,
                   batch_index=None, log_info=None):
    """..."""
    label, batch = to_batch
    subset = subset_subtarget(among_neurons)

    LOG.info("Run analysis %s to %s subtargets in batch %s",
             a.name, batch.shape[0], batch_index)

    mem_tot = 0

    def to_subtarget(row):
        """..."""
        nonlocal mem_tot
        index = dict(zip(batch.index.names, row.name))
        subtarget = (index["circuit"], index["subtarget"])
        value = measure_quantity(a, of_subtarget=row, index_entry=index,
                                 using_neuron_properties=subset(*subtarget),
                                 batch_size=batch.shape[0],
                                 log_info=f"Batch: {batch_index}")
        mem_row = getsizeof_measurement(value)
        mem_tot += mem_row
        LOG.info("MEMORY USAGE by subtarget %s (%s / %s): %sGB / total %sGB",
                 index["subtarget"], row.idx + 1, batch.shape[0] or "batches", mem_row, mem_tot)

        return using_store.write(value) if using_store else value

    saved = batch.assign(idx=range(batch.shape[0])).apply(to_subtarget, axis=1)
    update = using_store.prepare_toc(of_paths=saved)
    using_store.append_toc(update) if using_store else matrices
    return  using_store


def parallely_analyze(quantity, subtargets, neuron_properties, in_mode=None,
                      to_parallelize=None, to_save=None, log_info=None):
    """Run an analysis of quantity over all the subtargets in a table of contents.

    Computation for an analysis will be run in parallel over the subtargets,
    with results for each batch saved in an independent archive under an
    independent folder for each analysis to compute.
    Once the computation is done, a mapping of batch of the temporary HDF-stores will be
    returned for each analysis.

    Arguments
    ------------
    subtargets :: A TOC of adjacency matrices

    to_save :: Tuple(basedir, hdf_group) where
    ~            basedir :: Path to the directory where analysis results will be assembled,
    ~                       i.e. where the temporary HDF for each subtarget analysis will be written
    ~                       into an HDF store, for example `temp.h5` under the specified hdf-group.
    ~                       Example: (<tap-root>/"analysis", "analysis") where
    ~                                <tap-root> is specified in the config as `config[paths][root]`.
    ~                                For an analysis `A` then results for a given batch `B` of
    ~                                subtargets results should go to
    ~                                <tap-root>/"analysis"/<analysis-name>/<batch-index>/"tap.h5"
    ~                                under the HDF group
    ~                                "analysis"/<analysis-name>

    to_parallelize :: A mapping that describes how to parallelize an analysis.
    """
    a = quantity
    toc = subtargets
    properties = neuron_properties

    N = toc.shape[0]
    LOG.info("Analyze connectivity for %s subtargets", N)
    LOG.info("Example subtarget adjacency TOC: \n %s", pformat(toc.head()))

    in_mode_for_basedir = {"run": 'w', "resume": 'a'}.get(in_mode, 'r')
    b = check_basedir(to_save, quantity, to_parallelize, in_mode_for_basedir)
    q = quantity.name
    compute_nodes, njobs = read_njobs(to_parallelize, for_quantity=q)
    n = njobs
    batched = append_batch(toc, using_basedir=b, njobs=n)
    n_batches = batched.batch.max() + 1

    if compute_nodes > 1:
        raise NotImplementedError("Multi-node parallelization, not yet.")

    manager = Manager()
    bowl = manager.dict()
    processes = []

    basedir, group = to_save if to_save else (Path.cwd(), "analysis")
    abase = Path(basedir) / a.name
    assert abase.exists(), f"A workspace dir at {abase} must exist exist to run analysis {a.name}."

    def measure_batch(subtargets, *, index, bowl=None):
        """..."""
        p = b / f"{index}.h5"; g = f"{group}/{a.name}"; t = a.output_type
        s = matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=t)
        of_subtargets = (index, filter_pending(subtargets, in_store=s))

        m = in_mode
        result = apply_analysis(a, to_batch=of_subtargets, among_neurons=properties,
                                in_mode=m, using_store=s, batch_index=index,
                                log_info=log_info)
        if bowl is not None:
            bowl[index] = result
        return result

    for batch, of_subtargets in batched.groupby("batch"):
        LOG.info("Spawn process %s / %s", batch+1, n_batches)
        p = Process(target=measure_batch,
                    args=(of_subtargets,), kwargs={"index": batch, "bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED")

    for p in processes:
        p.join()

    LOG.info("Parallely analyze %s: obtained %s chunks.", a.name, len(bowl))
    batch_stores = {batch: stores for batch, stores in bowl.items()}
    LOG.info("DONE analyzing %s in %s subtargets.", a.name, N)

    return batch_stores


def check_basedir(to_save, quantity, to_parallelize=None, mode=None,
                  return_hdf_group=False):
    """The locaiton `to_save` in could have been used in a previous run.

    By default, the base directory will be prepared and returned.
    However, use `mode='r'` to get path to an existing basedir.
    """
    assert not mode or mode in ('w', 'r', 'a')
    mode = mode or 'w'

    try:
        p, hdf_group = to_save
    except TypeError:
        if return_hdf_group:
            raise TypeError(f"Expecting Tuple(basedir, hdf_group) to_save, not {to_save}")
        path = Path(to_save)
    else:
        path = Path(p)

    assert path.exists(), "Cannot save in a directory that does not exist."

    if to_parallelize:
        LOG.info("Check analysis basedir: \n %s", pformat(to_parallelize))
        config_already = path/"parallelize.json"
        if mode == 'w':
            if config_already.exists():
                raise ValueError(f"Location {to_save} has already been used to run.\n"
                                 "Please use a different location for the TAP workspace,\n"
                                 "or `tap resume ...` instead of `tap run ...`"
                                 f" to resume the run from the current state in {to_save}")
            write_config(to_parallelize, to_json=config_already)

    if not quantity:
        return path

    sq = path / quantity.name
    if mode == 'w':
        sq.mkdir(parents=False, exist_ok=True)

    if to_parallelize and quantity in to_parallelize["analyses"]:
        cpath = sq / "parallelize.json"
        if mode == 'w' and not cpath.exists():
            write_config(to_parallelize[quantity], to_json=cpath)

    q = quantity.name
    _, j = read_njobs(to_parallelize, for_quantity=q)
    njobs = sq / f"njobs-{j}"
    if mode == 'w':
        njobs.mkdir(parents=False, exist_ok=True)
    return (njobs, hdf_group) if return_hdf_group else njobs


def append_batch(toc, using_basedir=None, njobs=1):
    """Append a column of batches to a table of contents..."""
    LOG.info("Append %s batches to %s subtargets in TOC: \n %s %s using basedir",
             njobs, len(toc), pformat(toc.head()), "not" if not using_basedir else "")

    def redo_batches():
        return load_balance(toc, njobs, by="edge_count")

    if not using_basedir:
        return pd.concat([toc, redo_batches()], axis=1)

    path = Path(using_basedir)/BATCHED_SUBTARGETS
    previously_run = path.exists()
    if previously_run:
        previously = pd.read_hdf(path, key=f"njobs-{njobs}")
        batches = previously.reindex(toc.index)
        LOG.info("Found previously run batches of %s subtargets: \n %s",
                 len(batches), pformat(batches))
        if batches.isna().sum() != 0:
            raise ValueError(f"Found a previously run  batches in {using_basedir}"
                             " that are not the same as the current run's batches.\n"
                             "The input HDF store may be different than the previous run.\n"
                             "Please clean this location or use a different one.")
        return pd.concat([toc, batches], axis=1)

    batches = redo_batches()
    LOG.info("Computed %s subtarget batches: \n %s", len(batches), pformat(batches.head()))
    batches.to_hdf(path, key=f"njobs-{njobs}", format="table", mode='w')
    return pd.concat([toc, batches], axis=1)


def load_balance(toc, njobs, by=None):
    """..."""
    edge_counts =  toc.apply(lambda adj: adj.matrix.sum())
    computational_load = (np.cumsum(edge_counts.sort_values(ascending=True))
                          / edge_counts.sum())
    batches = ((njobs - 1) * computational_load).apply(int)

    LOG.info("Load balanced batches for %s subtargets: \n %s", len(toc),
             pformat(batches.head()))
    return batches.loc[edge_counts.index].rename("batch")


def read_njobs(to_parallelize, for_quantity):
    """..."""
    if not to_parallelize:
        return (1, 1)

    try:
        q = for_quantity.name
    except AttributeError:
        q = for_quantity

    if q not in to_parallelize["analyses"]:
        return (1, 1)

    p = to_parallelize["analyses"][q]
    n = p["number-compute-nodes"]; t = p["number-tasks-per-node"]
    return (n, n * t)


def filter_pending(subtargets, in_store):
    """Data `in_store` might already have results for subtargets.
    Filter out the subtargets that are already in-store, keeping only
    those that are pending computation.
    """
    try:
        toc = in_store.toc
    except KeyError:
        LOG.warning("No subtargets found to be already computed in the store at %s",
                    in_store.path)
        LOG.warning("Will assume that all are pending a run.")
        return subtargets

    stored = in_store.toc.reindex(subtargets.index)
    pending = subtargets[stored.isna()]
    n_all = len(subtargets)
    n_pend = len(pending)
    LOG.info("Pending %s / %s subtargets in batch store %s", n_pend, n_all, in_store.path)
    return pending


def batch_stores_at_basedir(b, under_group, for_matrix_type):
    """..."""
    g = under_group; m = for_matrix_type
    index = 0
    while True:
        p = b / f"{index}.h5"
        yield matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)
        index +=1


def load_parallel_run_analysis(a, parallelized, output_in):
    """..."""
    analysis = a
    a = analysis.name
    compute_nodes, njobs = read_njobs(parallelized, for_quantity=a)
    n = njobs

    base, hdf_group = check_basedir(output_in, analysis, parallelized, mode='r',
                                    return_hdf_group=True)
    subtarget_batches = pd.read_hdf(Path(base) / BATCHED_SUBTARGETS, f"njobs-{n}")

    n_batches = subtarget_batches.value_counts()
    batches = {b: Path(base) / f"{b}.h5" for b in subtarget_batches.drop_duplicates().values}

    for b, path in batches.items():
        assert path.exists(), f"Missing batch {b} of parallel analysis job at {base}."

    g = f"{hdf_group}/{a}"
    m = analysis.output_type
    stores = {b: matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)
              for b, p in batches.items()}
    return stores
