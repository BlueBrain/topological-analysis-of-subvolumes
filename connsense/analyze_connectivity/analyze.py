"""Analyze connectivity of subtargets
"""
import sys
import os
from pprint import pformat
from multiprocessing import Process, Manager
import joblib
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis import  Analysis
from ..io.read_config import write as write_config
from ..io.write_results import read as read_results
from ..io.slurm import SlurmConfig
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


GIGABYTE = 2 ** 30

BATCHED_SUBTARGETS = ("batches.h5", "subtargets")


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
                     tapping=None, batch_size=None, log_info=None):
    """Apply an analysis to a subtarget in row of a batch.
    """
    analysis = a
    neurons = using_neuron_properties
    s = of_subtarget
    i = s.idx + 1
    l = index_entry["subtarget"]
    LOG.info("Apply analysis %s to subtarget %s (%s / %s): \n%s \n%s \n%s",
             analysis.name, l, i, batch_size or "", pformat(index_entry), s,
             log_info or "")

    result = analysis.apply(s.matrix, neurons, tapping, index_entry, log_info)

    return result


def apply_analysis(a, to_batch, among_neurons, using_store=None, tapping=None,
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
        index_entry = pd.Series(row.name, index=batch.index.names)
        subtarget = (index_entry["circuit"], index_entry["subtarget"])
        value = measure_quantity(a, of_subtarget=row, index_entry=index_entry,
                                 using_neuron_properties=subset(*subtarget),
                                 tapping=tapping, batch_size=batch.shape[0],
                                 log_info=f"Batch: {batch_index}")
        mem_row = getsizeof_measurement(value)
        mem_tot += mem_row
        LOG.info("MEMORY USAGE by subtarget %s (%s / %s): %sGB / total %sGB",
                 index_entry["subtarget"], row.idx + 1, batch.shape[0] or "batches",
                 mem_row, mem_tot)

        return using_store.write(value) if using_store else value

    saved = batch.assign(idx=range(batch.shape[0])).apply(to_subtarget, axis=1)
    update = using_store.prepare_toc(of_paths=saved)
    using_store.append_toc(update) if using_store else matrices
    return  using_store


def load_subtargets_batch(rundir, batched_subtargets_h5=None, group=None):
    """Load subtargets from a HDF at rundir.

    NOTE: This may need a relook -- I am guessing the default behavior.
    ~     Check against where the subtarges are written.
    """
    batched_subtargets_h5 = batched_subtargets_h5 or "batched_subtargets.h5"
    njobs = lambda : Path(rundir).name
    subtargets_group = group or njobs()
    batch = rundir / batched_subtargets_h5
    return pd.read_hdf(batch, subtargets_group)


def _remove_link(path):
    try:
        return path.unlink()
    except FileNotFoundError:
        pass
    return None



def cmd_sbatch(slurm_params, at_path):
    LOG.info("Prepare a Slurm command using sbatch params: \n%s", slurm_params)
    slurm_params["executable"] = "tap-analysis"
    slurm_config = SlurmConfig(slurm_params)
    return slurm_config.save(to_filepath=at_path/"tap-analysis.sbatch")


def configure_launch_multi(number, quantity, using_subtargets, control, at_workspace,
                           action=None, in_mode=None, slurm_config=None):
    """
    Arguments
    -----------
    number :: of compute nodes to run on
    quantity :: analysis to run
    using_subtargets :: TOC of subtargets to compute
    control :: A JSON config to use to load a control, Use None when ther eis not control.
    at_workspace :: directory to use for input and output
    """
    LOG.info("Configure a %s multinode launch to analyze %s of %s subtargets working  %s",
             number, quantity.name, len(using_subtargets), at_workspace)

    basedir, to_run_in = at_workspace

    master_launchscript = to_run_in / "launchscript.sh"

    def configure_chunk(c, subtargets):
        """...
        Here we have assumed something like:
        run_basedir / <mode> / <step> / <substep> / <njobs>

        For example if number of jobs is 40,
        run_basedir / prod / analyze-connectivity / betti-counts / njobs-40

        We need basedir to link to the executable `tap-analysis` (look at the project's setup.py)

        NOTE: We have implemented random controls using the same idea...
        TODO: Update the doc for conttrols...
        TODO: Abstract out how out the layout of the base-rundir.
        ~     Or provide a path of an executable to run single node analysis explicitly.
        """
        rundir = to_run_in / f"compute-node-{c}"
        rundir.mkdir(parents=False, exist_ok=True)

        # sbatch = rundir/"tap-analysis.sbatch"
        # _remove_link(sbatch)
        # sbatch.symlink_to(basedir/"tap-analysis.sbatch")

        # run_analysis = rundir.joinpath("tap-analysis")
        # _remove_link(run_analysis)
        # run_analysis.symlink_to(basedir/"tap-analysis")

        config = rundir.joinpath("config.json")
        _remove_link(config)
        config.symlink_to(basedir/"config.json")

        if control:
            control_config = rundir/"control.json"
            _remove_link(control_config)
            control_config.symlink_to(control)

        h5, hdf_group = BATCHED_SUBTARGETS
        path_h5 = rundir / h5

        if path_h5.exists():
            LOG.warning("OVERRIDE existing data for batched subtargets exists at %s\n"
                        "Consider providing flags,"
                        " for example allow override only in pipeline `mode=resume`", h5)

        LOG.info("Write %s subtargets to %s / %s: \n%s", subtargets.shape[0], path_h5, hdf_group,
                 pformat(subtargets))
        subtargets[["batch", "compute_node"]].to_hdf(path_h5, key=hdf_group, mode='w')

        with open(master_launchscript, 'a') as to_launch:
            script = cmd_sbatch(slurm_config or {}, at_path=rundir).name

            def write(aline):
                to_launch.write(aline + '\n')

            write(f"##################### LAUNCH analysis {quantity.name} for chunk {c}"
                  f" of {len(subtargets)} subtargets. ######################")

            write(f"pushd {rundir}")

            write(f"sbatch {script} --configure=config.json --mode={in_mode} --quantity={quantity.name} {action}")

            write("popd")

        return rundir

    batches = using_subtargets.batch.drop_duplicates().reset_index(drop=True).sort_values()
    compute_nodes = np.linspace(0, number-1.e-6, batches.max() + 1, dtype=int)

    chunked = using_subtargets.assign(compute_node=compute_nodes[using_subtargets.batch])
    return {c: configure_chunk(c, subtargets) for c, subtargets in chunked.groupby("compute_node")}


def get_index(subtargets):
    """..."""
    return subtargets.index.to_frame().reset_index(drop=True)


def get_algorithms(subtargets):
    """..."""
    return get_index(subtargets).algorithm.unique()


def parallely_analyze(quantity, subtargets, neuron_properties, action=None, in_mode=None,
                      controls=None, to_parallelize=None, to_tap=None, to_save=None,
                      log_info=None):
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
    ~                       i.e. where the tempurary HDF for each subtarget analysis will be written
    ~                       into an HDF store, for example `temp.h5` under the specified hdf-group.
    ~                       Example: (<tap-root>/"analysis", "analysis") where
    ~                               <tap-root> is specified in the config as `config[paths][root]`.
    ~                                For an analysis `A` then results for a given batch `B` of
    ~                                subtargets results should go to
    ~                                <tap-root>/"analysis"/<analysis-name>/<batch-index>/"tap.h5"
    ~                                under the HDF group
    ~                                "analysis"/<analysis-name>

    to_parallelize :: A mapping that describes how to parallelize an analysis.
    to_tap :: A popeline HDFStore that contains data for analysis that have been previously run.
    """
    a = quantity
    toc = subtargets

    N = toc.shape[0]

    LOG.info("Analyze connectivity for %s subtargets, saving to %s", N, to_save)
    LOG.info("Example subtarget adjacency TOC: \n %s", pformat(toc.head()))


    if controls is None:
        rundir, hdf_group = check_basedir(to_save, quantity, to_parallelize, action,
                                          controls=None, return_hdf_group=True)
        base = rundir.parent.parent.parent.parent
        LOG.info("Checked rundir %s without controls \n\t with base at %s", rundir, base)

        assert rundir.exists, f"check basedir did not create {b}"

        q = quantity.name
        compute_nodes, njobs = read_njobs(to_parallelize, for_quantity=q)
        n = njobs

        batched = append_batch(toc, using_basedir=rundir, njobs=n)

        to_sbatch = to_parallelize["analyses"].get(quantity.name, {}).get("sbatch", None)

        if compute_nodes > 1:
            multirun = configure_launch_multi(compute_nodes, quantity,
                                              using_subtargets=batched, control=None,
                                              at_workspace=(base, rundir),
                                              action=action, in_mode=in_mode,
                                              slurm_config=to_sbatch)
            LOG.info("Multinode run: \n %s", pformat(multirun))
            return multirun

        return dispatch_single_node(quantity, batched, neuron_properties, to_tap,
                                    to_save=(rundir, hdf_group), log_info=log_info)

    rundirs, hdf_group = check_basedir(to_save, quantity, to_parallelize, action,
                                       controls=controls.algorithms.index, return_hdf_group=True)

    def analyze_controlled(an, algorithm):
        """..."""
        to_run = rundirs[an]
        assert to_run.exists, f"check basedir for not created rundir {to_run}"

        base = to_run.parent.parent.parent.parent.parent
        LOG.info("Checked rundir %s for control algorithm %s \n\t with base %s",
                 rundirs, algorithm, base)

        q = quantity.name
        compute_nodes, njobs = read_njobs(to_parallelize, for_quantity=q)
        n = njobs

        batched = append_batch(toc, for_control=algorithm, using_basedir=to_run, njobs=n)

        to_sbatch = to_parallelize["analyses"].get(quantity.name, {}).get("sbatch", None)

        multirun = configure_launch_multi(compute_nodes, quantity,
                                          using_subtargets=batched, control=to_run/"control.json",
                                          at_workspace=(base, to_run),
                                          action=action, in_mode=in_mode,
                                          slurm_config=to_sbatch)
        LOG.info("Multinode run: \n %s", pformat(multirun))
        return multirun

    return {an: analyze_controlled(an, algorithm) for an, algorithm in controls.algorithms.items()}


def dispatch_single_node(to_compute, batched_subtargets, neuron_properties, to_tap, to_save,
                         log_info=None):
    """Dispatch computation to single node with multi-processing.
    """
    LOG.warning("Dispatch (to %s) computation on %s subtargets on a single node.",
                to_compute.name, len(batched_subtargets))
    analysis = to_compute; properties = neuron_properties

    manager = Manager()
    bowl = manager.dict()
    processes = []

    rundir, group = to_save

    def measure_batch(subtargets, *, index, bowl=None):
        """..."""
        p = rundir / f"{index}.h5"; g = f"{group}/{to_compute.name}"
        t = to_compute.output_type
        s = matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=t)
        of_subtargets = (index, filter_pending(subtargets, in_store=s))

        result = apply_analysis(to_compute, to_batch=of_subtargets, among_neurons=properties,
                                using_store=s, tapping=to_tap, batch_index=index,
                                log_info=log_info)

        if bowl is not None:
            bowl[index] = result

        return result

    n_batches = batched_subtargets.batch.max() + 1
    for batch, of_subtargets in batched_subtargets.groupby("batch"):
        LOG.info("Spawn process %s / %s", batch+1, n_batches)
        p = Process(target=measure_batch,
                    args=(of_subtargets,), kwargs={"index": batch, "bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED")

    for p in processes:
        p.join()

    LOG.info("Parallely analyze %s: obtained %s chunks.", analysis.name, len(bowl))
    batch_stores = {batch: stores for batch, stores in bowl.items()}
    LOG.info("DONE analyzing %s in %s subtargets.", analysis.name, len(batched_subtargets))

    return batch_stores


BASEDIR_MODE = {"run": 'w', "resume": 'a', "inspect": 'r'}


def check_basedir(to_save, quantity, to_parallelize=None, mode=None, controls=None,
                  return_hdf_group=False, strict=False):

    """The locaiton `to_save` in could have been used in a previous run.

    By default, the base directory will be prepared and returned.
    However, use `mode='r'` to get path to an existing basedir.

    TODO: Rename this method -- basedir is ambigous
    """
    mode = BASEDIR_MODE.get(mode, mode)

    LOG.info("Check basedir to save %s quantity %s in mode %s to paralellize %s",
             to_save, quantity.name, mode, to_parallelize)
    assert not mode or mode in ('w', 'r', 'a'), f"Illegal mode {mode}"
    mode = mode or 'r'

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
        LOG.info("To parallelize, check analysis basedir: \n %s", pformat(to_parallelize))
        config_already = path/"parallelize.json"
        if mode == 'w':
            if strict and config_already.exists():
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

    def get_jobs(at_path):
        q = quantity.name
        _, j = read_njobs(to_parallelize, for_quantity=q)
        njobs = at_path / f"njobs-{j}"
        if mode == 'w':
            njobs.mkdir(parents=False, exist_ok=True)
        else:
            LOG.warning("A rundir at %s not created because mode was not to write.", njobs)
        return njobs
        return (njobs, hdf_group) if return_hdf_group else njobs

    if controls is None:
        njobs = get_jobs(at_path=sq)
        return (njobs, hdf_group) if return_hdf_group else njobs

    path_controls = sq / "controls"
    if mode == 'w':
        path_controls.mkdir(parents=False, exist_ok=True)

        if to_parallelize and quantity in to_parallelize["analyses"]:
            cpath = path_controls / "parallelize.json"
            if mode == 'w' and not cpath.exists():
                write_config(to_parallelize[quantity], to_json=cpath)

    def get_control(c):
        p = path_controls / c
        if mode == "w":
            p.mkdir(parents=False, exist_ok=True)

            if to_parallelize and quantity in to_parallelize["analyses"]:
                cpath = p / "parallelize.json"
                if mode == 'w' and not cpath.exists():
                    write_config(to_parallelize[quantity], to_json=cpath)

        return get_jobs(at_path=p)

    controlled = {c: get_control(c) for c in controls}

    return (controlled, hdf_group) if return_hdf_group else controlled


def configure_algorithm(for_control, using_basedir):
    """..."""
    if not for_control: return

    from connsense.io.read_config import write
    control_json = Path(using_basedir.parent) / "control.json"
    description = deepcopy(for_control.description)
    description["name"] = for_control.name
    write(description, control_json)

    in_basedir = using_basedir/"control.json"
    _remove_link(in_basedir)
    in_basedir.symlink_to(control_json)
    return


def append_batch(toc, for_control=None, using_basedir=None, njobs=1):
    """Append a column of batches to a table of contents...

    If `using_basedir`, check for HDF dataframe containing a subset of adjacency matrices TOC
    from the input store with an additional column that gives the batch to run each row in.

    If the relevant pipeline step has not yet been run, batches will be computed for
    the argued `toc` and if `using_basedir` will be saved in a HDF dataframe.

    If for control by an algorithm, then the algorithm will be asked to write it's information too,
    so that it can be loaded by a downstream step that launches the parallel computations.
    """
    LOG.info("Append %s batches to %s subtargets in TOC: \n %s %s",
             njobs, len(toc), pformat(toc.head()),
             "not" if not using_basedir else f"using basedir{using_basedir}")

    def redo_batches():
        return load_balance_batches(toc, njobs, for_control, by="edge_count")

    if not using_basedir:
        return pd.concat([toc, redo_batches()], axis=1)

    subtargets_h5, and_hdf_group = BATCHED_SUBTARGETS
    path_h5 = Path(using_basedir)/subtargets_h5
    previously_run = path_h5.exists()
    if previously_run:
        previously = pd.read_hdf(path_h5, key=and_hdf_group)
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
    batches.to_hdf(path_h5, key=and_hdf_group, format="fixed", mode='w')
    configure_algorithm(for_control, using_basedir)

    return pd.concat([toc, batches], axis=1)


def get_size(index, for_control=None):
    """Get size of an adjancency object,
    which could be a sparse matrix or a lazy random wrapper around it.
    """
    def apply(adj):
        try:
            original = adj.original
        except AttributeError:
            original = adj

        try:
            matrix = original.matrix
        except AttributeError:
            matrix = original

        s = matrix.sum() + 1
        LOG.info("Control %s: Get size for subvolume %s / %s: %s",
                 for_control.name if for_control else "original",  index[adj], len(index), s)
        return s
    return apply


def load_balance_batches(toc, njobs, for_control=None,  by=None):
    """..."""
    index_adj = pd.Series(range(len(toc)), index=toc.values)
    edge_counts =  toc.apply(get_size(index_adj, for_control))
    computational_load = (np.cumsum(edge_counts.sort_values(ascending=True))
                          /edge_counts.sum())
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

    raise RuntimeError("Python should not have executed me")


def load_batched_subtargets(basedir, compute_nodes=None, as_paths=False,
                            COMPUTE_NODE="compute-node-{}", append_compute_node=False):
    """Load subtargets (i.e. a subset of adjacency matrices TOC from the input store,
    for which an analysis will be run.

    Arguments : INPROGRESS
    ----------
    to_analyze :: the analysis to run
    parallelized :: config to parallize with
    basedir :: path to the directory where the computation is being run
    COMPUTE_NODE :: string-pattern that names compute nodes...
    """
    LOG.info("Load a batch of subtargets from %s using %s compute nodes",
             basedir, compute_nodes)
    batched_subtargets_h5, and_hdf_group = BATCHED_SUBTARGETS

    def locate_compute_node(c):
        return Path(basedir) if c is None else Path(basedir) / COMPUTE_NODE.format(c)

    def get_compute_node(c):
        path_compute_node = locate_compute_node(c)
        path_h5 =  path_compute_node / batched_subtargets_h5
        if not path_h5.exists():
            raise RuntimeError(f"Not a valid TAP workspace: {basedir}."
                               " No batch of subtargets found at compute node "
                               f"{path_compute_node}.")
        def locate_batch(b):
            return path_compute_node / f"{b}.h5"

        LOG.info("Find batches in %s / %s", path_h5, and_hdf_group)
        batches = pd.read_hdf(path_h5, and_hdf_group)
        try:
            batches = batches.batch
        except AttributeError:
            LOG.info("No batches, must have been a single run without parallelization.")
        LOG.info("Batches in %s: \n %s", path_h5, pformat(batches))
        return batches.apply(locate_batch).rename("path") if as_paths else batches

    if not compute_nodes:
        return get_compute_node(None)

    col_compute_node = ({"keys": compute_nodes, "names": ["compute_node"]}
                        if append_compute_node else {})
    return pd.concat([get_compute_node(c) for c in compute_nodes], axis=0, **col_compute_node)


def load_parallel_run_analysis(a, parallelized, analyses_rundir):
    """Load data stores produced by a single-node parallel computation,
    each of which should have a produced a HDF store in a sub-directory
    where the analyses for a TAP pipeline are being run.
    """
    analysis = a
    a = analysis.name
    n_compute_nodes, njobs = read_njobs(parallelized, for_quantity=a)
    compute_nodes = range(n_compute_nodes) if n_compute_nodes > 1 else None
    n = njobs

    base, hdf_group = check_basedir(analyses_rundir, analysis, parallelized, mode='r',
                                    return_hdf_group=True)
    subtarget_batches = load_batched_subtargets(Path(base), compute_nodes, as_paths=True)

    LOG.info("Assemble stores for subtargets located at \n %s", pformat(subtarget_batches.values))

    n_batches = subtarget_batches.value_counts()
    LOG.info("Loading parallel run analysis %s, batch sizes: \n %s", a, pformat(n_batches))

    def load_stores(batches):
        for path in batches.values:
            if not path.exists():
                raise RuntimeError(f"Missing batch of parallel analysis job at {path}"
                                " To be a valid TAP workspace a previous run should have"
                                f" created data stores for each of the {len(n_batches)} batches")

        g = f"{hdf_group}/{a}"
        m = analysis.output_type
        LOG.info("Load data stores computed in a %s-parallel run for analysis %s",
                len(batches), g)
        stores = {b: matrices.get_store(to_hdf_at_path=p, under_group=g, for_matrix_type=m)
                  for b, p in batches.items()}
        return stores

    return load_stores(subtarget_batches.drop_duplicates())
    return load_stores({b: Path(b) for b in subtarget_batches.drop_duplicates().values})
