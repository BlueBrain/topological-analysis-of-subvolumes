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
from ..io.write_results import (read as read_results)
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


def load_balance(toc, njobs, by=None):
    """..."""
    edge_counts =  toc.apply(lambda adj: adj.matrix.sum())
    computational_load = (np.cumsum(edge_counts.sort_values(ascending=True))
                          / edge_counts.sum())
    batches = ((njobs - 1) * computational_load).apply(int)
    return batches.loc[edge_counts.index].rename("batch")


GIGABYTE = 2 ** 30

def append_batch(toc, njobs):
    """..."""
    return pd.concat([toc, load_balance(toc, njobs, by="edge_count")], axis=1)


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


def apply_analysis(a, to_batch, among_neurons, using_store=None,
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


def analyze_table_of_contents(toc, using_neuron_properties, applying_analyses,
                              store_batch=None, with_batches_of_size=None, njobs=72):
    """..."""
    LOG.info("Analyze connectivity: %s", toc.shape[0])

    N = toc.shape[0]

    neurons = using_neuron_properties
    analyses = applying_analyses
    LOG.info("Analyze %s subtargets using  %s.", N, [a.name for a in analyses])

    batch_size = with_batches_of_size or int(N / (njobs-1)) + 1
    batched = append_batch(toc, njobs)

    n_analyses = len(analyses)
    n_batches = batched.batch.max() + 1

    def map_analyses(batch, label=None, bowl=None):
        """..."""
        LOG.info("ANALYZE batch %s / %s with %s targets and columns %s",
                 label, n_batches, batch.shape[0], batch.columns)

        nrows = batch.shape[0]
        def get_neurons(row):
            """..."""
            index = dict(zip(batch.index.names, row.name))
            return (neurons.loc[index["circuit"], index["subtarget"]]
                    .reset_index(drop=True))

        def analyze(analysis, at_index):
            """..."""
            memory_used_total = 0
            def analyze_row(r):
                """..."""
                nonlocal memory_used_total
                log_info = (f"({at_index} / {n_analyses}"
                            f" Batch {label} matrix {r.idx} / {batch.shape[0]}")
                result = analysis.apply(r.matrix, get_neurons(r), log_info)
                memory_used_row = sys.getsizeof(result) / GIGABYTE
                memory_used_total += memory_used_row

                LOG.info("\t\t\t MEMORY USAGE by data results from the analyses\n"
                         "\t\t\t for analysis %s batch %s matrix %s / %s: %s / %s",
                         analysis.name, label, r.idx, nrows,
                         memory_used_row, memory_used_total)

                return result

            return batch.assign(idx=range(batch.shape[0])).apply(analyze_row, axis=1)

        analyzed = {a: analyze(a, i) for i, a in enumerate(analyses)}
        LOG.info("MEMORY USAGE after running all analyses on batch %s: %s",
                 label, sum(sys.getsizeof(result) for result in analyzed.values()))

        LOG.info("Done batch %s / %s with %s targets, columns %s: analyzed to shape %s",
                 label, n_batches, batch.shape[0], batch.columns, len(analyzed))

        if not store_batch:
            bowl[label] = analyzed
            return analyzed

        store_measurement = store_batch(b=label)
        hdf_paths = {a.name: store_measurement(a, data) for a, data in analyzed.items()}
        LOG.info("Saved batch %s / %s with %s targets, columns %s, analyzed to shape %s"
                 " To an temporary HDF",
                 label, n_batches, batch.shape[0], batch.columns, len(analyzed))
        bowl[label] = hdf_paths
        return hdf_paths

    manager = Manager()
    bowl = manager.dict()
    processes = []

    for i, batch in batched.groupby("batch"):

        p = Process(target=map_analyses, args=(batch,),
                    kwargs={"label": "chunk-{}".format(i), "bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED")

    for p in processes:
        p.join()

    LOG.info("Obtained %s chunks.", len(bowl))
    result = {b: analyses_hdf_paths for b, analyses_hdf_paths in bowl.items()}
    LOG.info("DONE analyzing %s subtargets using  %s.", N, [a.name for a in analyses])

    return result


def parallely_analyze(quantity, subtargets, neuron_properties,
                      save=None, with_batches_of_size=None, njobs=-1, output=None,
                      log_info=None):
    """Run an analysis of quantity over all the subtargets in a table of contents.

    store_batch : a callable that provides a matrix storet for a given batch...
    """
    a = quantity
    toc = subtargets
    properties = neuron_properties

    N = toc.shape[0]
    LOG.info("Analyze connectivity %s", N)

    njobs = (1 if not njobs
             else (os.cpu_count() if njobs <= -1 else njobs))
    batched = append_batch(toc, njobs)
    n_batches = batched.batch.max() + 1

    manager = Manager()
    bowl = manager.dict()
    processes = []

    basedir, group = save if save else (Path.cwd(), "analysis")
    g = f"{group}/{a.name}"
    m = a.output_type

    def measure_batch(subtargets, *, batch, bowl=None):
        """..."""
        p = Path(basedir) / f"{batch}.h5"
        s = matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)
        of_subtargets = (batch, subtargets)

        result = apply_analysis(a, to_batch=of_subtargets, among_neurons=properties,
                                using_store=s, batch_index=batch, log_info=log_info)
        if bowl is not None:
            bowl[batch] = result
        return result

    for batch, of_subtargets in batched.groupby("batch"):
        LOG.info("Spawn process %s / %s", batch+1, n_batches)
        p = Process(target=measure_batch, args=(of_subtargets,),
                    kwargs={"batch": batch, "bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED")

    for p in processes:
        p.join()

    LOG.info("Parallely analyze %s: obtained %s chunks.", a.name, len(bowl))
    batch_stores = {batch: stores for batch, stores in bowl.items()}
    LOG.info("DONE analyzing %s in %s subtargets.", a.name, N)

    return batch_stores
