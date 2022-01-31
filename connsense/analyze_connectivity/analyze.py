"""Analyze connectivity of subtargets
"""
import sys
from multiprocessing import Process, Manager

import numpy as np
import pandas as pd

from analysis import  Analysis
from ..io.write_results import (read as read_results)
from ..io import logging

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


def load_balance(toc, by=None, njobs=72):
    """..."""
    edge_counts =  toc.apply(lambda adj: adj.matrix.sum())
    computational_load = (np.cumsum(edge_counts.sort_values(ascending=True))
                          / edge_counts.sum())
    batches = ((njobs - 1) * computational_load).apply(int)
    return batches.loc[edge_counts.index].rename("batch")


GIGABYTE = 2 ** 30


def analyze_table_of_contents(toc, using_neuron_properties, applying_analyses
                             ,store_batch=None, with_batches_of_size=None, njobs=72):
    """..."""
    LOG.info("Analyze connectivity: %s", toc.shape[0])

    N = toc.shape[0]

    neurons = using_neuron_properties
    analyses = applying_analyses
    LOG.info("Analyze %s subtargets using  %s.", N, [a.name for a in analyses])

    batch_size = with_batches_of_size or int(N / (njobs-1)) + 1

    batched = pd.concat([toc, load_balance(toc, by="edge_count", njobs=njobs)],
                        axis=1)

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
        #analyzed = pd.concat([analyze(a, i) for i, a in enumerate(analyses)],
                             #axis=0, keys=[a.name for a in analyses],
                             #names=["analysis"])

        LOG.info("Done batch %s / %s with %s targets, columns %s: analyzed to shape %s",
                 label, n_batches, batch.shape[0], batch.columns, len(analyzed))

        if not store_batch:
            bowl[label] = analyzed
            return analyzed

        hdf_paths = {analysis.name: store_batch(analysis, data,
                                                to_hdf_in_file=f"{label}.h5")
                     for analysis, data in analyzed.items()}
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

    #result = pd.concat([analyzed for _, analyzed in bowl.items()], axis=0)
    LOG.info("Obtained %s chunks.", len(bowl))
    result = {batch: analyses_hdf_paths
              for batch, analyses_hdf_paths in bowl.items() }
    LOG.info("DONE analyzing %s subtargets using  %s.", N, [a.name for a in analyses])

    return result
