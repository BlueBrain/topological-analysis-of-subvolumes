"""Connectivity in subtargets."""
from collections.abc import Mapping
import h5py
import numpy
from scipy import sparse
import pandas
from tqdm import tqdm
from bluepy import Circuit

from connsense import plugins
from ..io.write_results import (read as read_results,
                                write_toc_plus_payload as write,
                                default_hdf)
from ..io.read_config import check_paths
from ..io import logging

from ..define_subtargets.config import SubtargetsConfig

STEP = "extract-connectivity"
LOG = logging.get_logger(STEP)

def connection_matrix_for_gids(sonata_fn, gids):
    idx = numpy.array(gids) - 1  # From gids to sonata "node" indices (base 0 instead of base 1)
    h5 = h5py.File(sonata_fn, "r")['edges/default']  # TODO: Instead of hard coding "default" that could be a config parameter
    N = len(gids)

    indices = []
    indptr = [0]
    for id_post in tqdm(idx):
        ids_pre = []
        ranges = h5['indices']['target_to_source']['node_id_to_ranges'][id_post, :]
        for block in h5['indices']['target_to_source']['range_to_edge_id'][ranges[0]:ranges[1], :]:
            ids_pre.append(h5['source_node_id'][block[0]:block[1]])
        if len(ids_pre) > 0:
            row_ids = numpy.nonzero(numpy.in1d(idx, numpy.hstack(ids_pre)))[0]
            indices.extend(row_ids)
        indptr.append(len(indices))
    mat = sparse.csc_matrix((numpy.ones(len(indices), dtype=bool), indices, indptr), shape=(N, N))
    return mat


def full_cmat(circ, conn_file, chunk=50000000):
    h5 = h5py.File(conn_file, "r")['edges/default']
    N = circ.cells.count()

    dset_sz = h5['source_node_id'].shape[0]
    A = numpy.zeros(dset_sz, dtype=int)
    B = numpy.zeros(dset_sz, dtype=int)
    splits = numpy.arange(0, dset_sz + chunk, chunk)
    for splt_fr, splt_to in tqdm(zip(splits[:-1], splits[1:]), total=len(splits) - 1):
        A[splt_fr:splt_to] = h5['source_node_id'][splt_fr:splt_to]
        B[splt_fr:splt_to] = h5['target_node_id'][splt_fr:splt_to]
    M = sparse.coo_matrix((numpy.ones_like(A, dtype=bool), (A, B)), shape=(N, N))
    return M.tocsr()


def random_cmat(circ, conn_file):
    """ For testing without running 'full_cmat'"""
    N = circ.cells.count()
    A = numpy.random.randint(0, N, 1000)
    B = numpy.random.randint(0, N, 1000)
    M = sparse.coo_matrix((numpy.ones_like(A, dtype=bool), (A, B)), shape=(N, N))
    return M.tocsr()


def find_connectome_files(circuit_dict):
    def lookup_sonata_files(circ_name, conn_lst):
        circ = circuit_dict[circ_name]
        return [circ.config["connectome"] if conn == "local"
                else circ.config["projections"][conn]
                for conn in conn_lst]

    return lookup_sonata_files


def run_extraction(circuits, subtargets, list_of_connectomes):
    if len(list_of_connectomes) == 0:
        LOG.warning("No connectomes defined. This step will do nothing!")
    circuits = dict([(k, Circuit(v)) for k, v in circuits.items()])

    connectome_names = [" + ".join(lst) for lst in list_of_connectomes]
    connectome_series = pandas.Series(list_of_connectomes, index=connectome_names)
    connectome_series.index.name = "connectome"

    # Cross product with list of connectomes
    connectomes = subtargets.index.to_frame().apply(lambda _: connectome_series, axis=1).stack()
    # Connectome to sonata filenames
    sonata_files = connectomes.index.to_frame()["circuit"].combine(connectomes, func=find_connectome_files(circuits))
    # Because pandas is utterly stupid
    subtargets = subtargets[sonata_files.index.droplevel("connectome")]
    subtargets.index = sonata_files.index

    # Extract connection matrices
    def extract(lst_connectome_files, gids):
        mat = sparse.csc_matrix((len(gids), len(gids)), dtype=bool)
        for conn_file in lst_connectome_files:
            mat = mat + connection_matrix_for_gids(conn_file, gids)
        return mat

    con_mats = sonata_files.combine(subtargets, func=extract)
    return con_mats


def run_extraction_from_full_matrix(circuits, subtargets, list_of_connectomes):
    """..."""
    LOG.info("Extract connectivity for connectomes: %s", list_of_connectomes)

    if len(list_of_connectomes) == 0:
        LOG.warning("No connectomes defined. This step will do nothing!")

    circuits = {k: c if isinstance(c, Circuit) else Circuit(c) for k, c in circuits.items()}
    connectome_files = find_connectome_files(circuits)

    connectome_names = [" + ".join(lst) for lst in list_of_connectomes]

    circ_lvl_idx = subtargets.index.names.index("circuit")
    idxx = [circ_lvl_idx] + numpy.setdiff1d(numpy.arange(len(subtargets.index.levels)), circ_lvl_idx).tolist()
    subtargets = subtargets.reorder_levels(idxx)

    res = []
    circ_names = list(subtargets.index.levels[0])
    for circ_name in circ_names:
        LOG.info("Connectivity for circuit {0}...".format(circ_name))
        conn_buffer = {}
        circ = circuits[circ_name]
        N = circ.cells.count()
        res_over_connectomes = []
        for lst_conn_ids in list_of_connectomes:
            LOG.info("\t...for the following connectomes: {0}".format(lst_conn_ids))
            lst_conn_files = connectome_files(circ_name, lst_conn_ids)
            M = sparse.csr_matrix((N, N), dtype=bool)
            for conn_file in lst_conn_files:
                if conn_file not in conn_buffer:
                    LOG.info("{0} not buffered... creating...".format(conn_file))
                    conn_buffer[conn_file] = full_cmat(circ, conn_file)  # random_cmat(circ, conn_file)
                M = M + conn_buffer[conn_file]
            subM = subtargets[circ_name].apply(lambda x: M[numpy.ix_(numpy.array(x) - 1, numpy.array(x) - 1)])
            res_over_connectomes.append(subM)
        res.append(pandas.concat(res_over_connectomes, keys=connectome_names, names=["connectome"]))
    res = pandas.concat(res, keys=circ_names, names=["circuit"])
    return res


def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, default_hdf(STEP))

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group)


def resolve_connectomes(in_argued):
    """..."""
    if isinstance(in_argued, str):
        return [[in_argued]]

    raise NotImplementedError(f"Argued type {type(in_argued)}."
                              " To do when ready to analyze local + mid-range.")

def extract_subtargets(in_config, connectome=None, output=None):
    """..."""
    LOG.warning("Extract conectivity of subtargets")

    input_paths, output_paths = check_paths(in_config, STEP)

    subtarget_cfg = SubtargetsConfig(in_config)

    path_subtargets = output_paths["steps"]["define-subtargets"]
    LOG.info("Read subtargets from %s", path_subtargets)
    subtargets = read_results(path_subtargets, for_step="extract-connectivity")
    LOG.info("Done reading subtargets %s", len(subtargets))

    parameters = in_config["parameters"]["extract-connectivity"]
    configured = parameters.get("connectomes", [])

    connectome = connectome or "local"
    assert connectome in configured, f"Argued connectome {connectome} must be among {configured}"

    if isinstance(configured, Mapping):
        computation = configured[connectome]
        LOG.info("Use a configured method to extract connectivity: %s", computation)
        _, method = plugins.import_module(computation["source"], computation["method"])
        extracted = method(subtarget_cfg.input_circuit, connectome, subtargets)
    else:
        LOG.info("Use connsense built in methods to extract connectivity.")
        assert isinstance(configured, list)
        extracted = run_extraction_from_full_matrix(subtarget_cfg.input_circuit, subtargets,
                                                    resolve_connectomes(in_argued=connectome))

    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    write(extracted, to_output, format="table")

    LOG.warning("DONE, exctracting %s subtarget connectivity", len(extracted))
    return to_output
