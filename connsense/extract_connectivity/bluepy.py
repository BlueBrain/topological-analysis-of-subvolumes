#!/usr/bin/env python3

"""Extract connectivity using `bluepy`.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from tqdm import tqdm

from bluepy import Cell, Synapse, Direction

from ..io import logging

STEP = "extract-connectivity"

INTRINSIC = ("local", "intra_SSCX_midrange_wm")

LOG = logging.get_logger(STEP)


def find_connectome(c, in_circuit):
    """...TODO: find a solution to removing `-extrinsic` explictly here.
    This is done to consider WM connectome as extrinsic...
    """
    return  in_circuit.connectome if c == "local" else in_circuit.projection(c.split("-extrinsic")[0])


def iter_afferent_afferent(to_subtarget, in_connectome_labeled, of_circuit, as_connections=False):
    """..."""
    edges_are_intrinsic = in_connectome_labeled in ("local", "long-range", "cortico-cortical")

    connectome = find_connectome(in_connectome_labeled, of_circuit)

    assert (isinstance(to_subtarget, (list, pd.Series)) or
            isinstance(to_subtarget, np.ndarray) and to_subtarget.ndim == 1),(
                f"Not handled subtargets of type, {type(to_subtarget)}")

    def get_afferent(gid):
        afferent = connectome.afferent_gids(gid)
        incoming = afferent[np.in1d(afferent, to_subtarget)] if edges_are_intrinsic else afferent
        if not as_connections:
            return incoming
        return pd.DataFrame({Synapse.PRE_GID: incoming, Synapse.POST_GID: gid})

    return (get_afferent(gid) for gid in to_subtarget)


def iter_connections(in_subtarget, connectome_labeled, of_circuit):
    """..."""

#    edges_are_intrinsic = connectome_labeled in INTRINSIC
#    connectome = find_connectome(connectome_labeled, of_circuit)
    labeled, edges_are_intrinsic = connectome_labeled
    connectome = find_connectome(labeled, of_circuit)

    try:
        gids = in_subtarget["gids"]
    except (KeyError, IndexError):
        gids = in_subtarget

    return connectome.iter_connections(gids if edges_are_intrinsic else None, gids, return_synapse_count=True)


def _get_connections_serial(in_subtarget, connectome_labeled, of_circuit):
    """..."""
    #iterc = iter_afferent_afferent(to_subtarget, in_connectome_labeled, of_circuit, as_connections=True)
    iterc = iter_connections(in_subtarget, connectome_labeled, of_circuit)
    return pd.DataFrame(list(iterc), columns=[Synapse.PRE_GID, Synapse.POST_GID, "nsyn"])


def iter_afferent(to_gids, in_subtarget, connectome_labeled, of_circuit):
    """..."""
    #edges_are_intrinsic = connectome_labeled in INTRINSIC
    #connectome = find_connectome(connectome_labeled, of_circuit)
    labeled, edges_are_intrinsic = connectome_labeled
    connectome = find_connectome(labeled, of_circuit)

    return connectome.iter_connections(pre=in_subtarget if edges_are_intrinsic else None, post=to_gids,
                                       return_synapse_count=True)


def get_connections(in_subtarget, connectome_labeled, of_circuit, n_batches=1):
    """Batch subtarget connections and run in parallel ...
    """
    if n_batches == 1:
        return _get_connections_serial(in_subtarget, connectome_labeled, of_circuit)

    def run_batch(gids, *, index=0, in_bowl=None):
        """..."""
        LOG.info("Get connections for %s gids batch %s among %s batches", len(gids), index, n_batches)
        def afferent(gid):
            """..."""
            iterc = iter_afferent([gid], in_subtarget,connectome_labeled, of_circuit)
            return pd.DataFrame(list(iterc), columns=[Synapse.PRE_GID, Synapse.POST_GID, "nsyn"])

        connections_batches = [batch for batch in (afferent(gid=g) for g in tqdm(gids))
                               if batch is not None and not batch.empty]
        if not connections_batches:
            LOG.warning("No connections in batch %s of %s gids: ", index, len(gids))
            return pd.DataFrame()

        connections = pd.concat(connections_batches)
        LOG.info("Done connections for %s gids batch %s among %s batches: \t%s connections",
                 len(gids), index or 1, n_batches, len(connections))
        if in_bowl is not None:
            in_bowl[index] = connections
        return connections

    from multiprocessing import Process, Manager
    manager = Manager()
    bowl = manager.dict()
    processes = []

    LOG.info("Extract edges for %s cells in %s batches", len(in_subtarget), n_batches)
    batched_gids = pd.DataFrame({"gid": in_subtarget,
                                "batch": np.linspace(0, n_batches - 1.e-9, len(in_subtarget), dtype=int)})
    for batch, gids in batched_gids.groupby("batch"):
        LOG.info("Spawn extraction of batch %s / %s", batch, n_batches)
        p = Process(target=run_batch, args=(gids["gid"].values,), kwargs={"index": batch, "in_bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("LAUNCHED computation of %s batches", n_batches)
    for p in processes:
        LOG.info("JOIN process %s", p)
        p.join()

    batched_connections = [x for x in bowl.values() if x is not None and not x.empty]
    return pd.concat(batched_connections).reset_index(drop=True) if batched_connections else pd.DataFrame()


def anonymize_gids(connections, gids, edges_are_intrinsic):
    """For connsense analyses.
    """
    source_gids = connections[Synapse.PRE_GID].to_numpy(int)
    target_gids = connections[Synapse.POST_GID].to_numpy(int)

    nodes_idx = pd.Series(gids.index.values, name="index", index=pd.Index(gids.to_numpy(), name="gid")).sort_index()
    N = len(gids)

    if edges_are_intrinsic:
        source_ids = nodes_idx[source_gids].values
        target_ids = nodes_idx[target_gids].values
    else:
        source_ids = 0
        target_ids = nodes_idx[target_gids].values + 1
        nodes_idx += 1

    edges = (pd.DataFrame({"source": source_ids, "target": target_ids}).join(connections)
             .set_index([Synapse.PRE_GID, Synapse.POST_GID]).rename(columns={"nsyn": 'weight'}))

    return (edges.sort_index(), nodes_idx.sort_index())


def get_adjacency(edges, number_nodes):
    """..."""
    N = number_nodes
    weighted = (edges.value_counts().rename("weight").reset_index() if "weight" not in edges
                else edges)
    rows = weighted["source"].to_numpy(int)
    cols = weighted["target"].to_numpy(int)
    data = weighted["weight"]
    return sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=int)


def as_adjmat(connections, gids, edges_are_intrinsic):
    """..."""
    edges, nodes = anonymize_gids(connections, gids, edges_are_intrinsic)
    N = len(nodes) if edges_are_intrinsic else len(nodes) + 1
    return get_adjacency(edges, number_nodes=N)


def extract_adj_0(circuit, edge_population, subtargets):
    """..."""
    LOG.info("Extract connectivity for circuit %s, subtarget %s", circuit, subtargets.index.values)

    connectome = edge_population["connectome"]
    source_population = edge_population["source_node_population"]
    target_population = edge_population["target_node_population"]
    edges_are_intrinsic = source_population == target_population
    c = circuit
    p = edge_population

    def extract_subtarget(gids):
        gids = pd.Series(gids, name="gid")
        connections = get_connections(to_subtarget=gids, in_edge_population=p, of_circuit=c)
        return as_adjmat(connections, gids, edge_population=p)

    connectivity = subtargets.apply(extract_subtarget)
    return pd.concat([connectivity], axis=0, keys=[connectome], names=["connectome"])


def extract_adj_batch(circuit, connectome, subtargets):
    """..."""
    LOG.info("Extract connectivity for circuit %s connectome %s, subtargets %s",
             circuit, connectome, subtargets.index.values)

    edges_are_intrinsic = connectome in INTRINSIC

    def extract_subtarget(gids):
        gids = pd.Series(gids, name="gid")
        connections = get_connections(to_subtarget=gids, in_connectome_labeled=connectome, of_circuit=circuit)
        return as_adjmat(connections, gids, edges_are_intrinsic)

    connectivity = subtargets.apply(extract_subtarget)
    return pd.concat([connectivity], axis=0, keys=[connectome], names=["connectome"])


def extract_adj_series(circuit, connectome, subtarget):
    """..."""
    assert len(subtarget) == 1, ("extract_adj will work with a single subtarget packaged in a single element Series."
                                 f" Provided list has {len(subtarget)}")
    LOG.info("Extract connectivity for circuit %s connectome %s, subtargets %s", circuit, connectome, subtarget)

    edges_are_intrinsic = connectome in INTRINSIC

    gids = pd.Series(subtarget.iloc[0], name="gid")
    connections = get_connections(subtarget, connectome_labeled=connectome, of_circuit=circuit)
    return as_adjmat(connections, gids, edges_are_intrinsic)


def extract_adj(circuit, connectome, subtarget, *, sources="intrinsic", n_parallel_batches=1):
    """..."""
    LOG.info("Extract connectivity for circuit %s connectome %s, subtargets \n%s", circuit, connectome, subtarget)
    intrinsic = sources == "intrinsic"
    gids = pd.Series(subtarget, name="gid")
    connections = get_connections(subtarget, connectome_labeled=(connectome, intrinsic), of_circuit=circuit,
                                  n_batches=n_parallel_batches)
    return as_adjmat(connections, gids, intrinsic)

def extract_edge_0(properties):
    """..."""
    synprops = [Synapse.PRE_GID, Synapse.POST_GID] + [Synapse[p.upper()] for p in properties]

    def extract(*, circuit, edge_population, subtargets):
        """..."""
        LOG.info("Extract properties for circuit %s, subtarget \n%s", circuit, subtargets.index.values)

        p = edge_population; c = circuit
        edges_are_intrinsic = p["source_node_population"] == p["target_node_population"]

        connectome = find_connectome(p["connectome"], in_circuit=c)

        def extract_subtarget(s):
            return connectome.pathway_synapses(s if edges_are_intrinsic else None, s, synprops)

        values = subtargets.apply(extract_subtarget)
        return pd.concat([values], axis=0, keys=[p["connectome"]], names=["connectome"])

    return extract


def extract_edge_batch(properties):
    """..."""
    synprops = [Synapse.PRE_GID, Synapse.POST_GID] + [Synapse[p.upper()] for p in properties]

    def extract(circuit, connectome, subtargets):
        """..."""
        LOG.info("Extract properties for circuit %s, subtarget \n%s", circuit, subtargets.index.values)

        edges_are_intrinsic = connectome in INTRINSIC

        circuit_connectome = find_connectome(connectome, circuit)

        def extract_subtarget(s):
            return (circuit_connectome.pathway_synapses(s if edges_are_intrinsic else None, s, synprops)
                    .reset_index().rename(columns={"index": "Synapse_ID"}))

        values = subtargets.apply(extract_subtarget)
        return (pd.concat([values], keys=[connectome], names=["connectome"])
                .reorder_levels(["circuit", "connectome", "subtarget"]))

    return extract


def extract_edge(properties, statistics=None):
    """..."""
    Synapse_CONN = [Synapse.PRE_GID, Synapse.POST_GID]
    synprops = Synapse_CONN + [Synapse[p.upper()] for p in properties]

    if statistics is True:
        statistics = ["mean", "std"]

    def extract(circuit, connectome, subtarget):
        """..."""
        LOG.info("Extract properties for circuit %s, subtarget \n%s", circuit, subtarget)

        edges_are_intrinsic = connectome in INTRINSIC

        circuit_connectome = find_connectome(connectome, circuit)

        gids = subtarget["gids"]
        value = circuit_connectome.pathway_synapses(gids if edges_are_intrinsic else None, gids, synprops)

        if not statistics:
            return value

        conns = value.reset_index(drop=True)
        summary = conns.groupby(Synapse_CONN).agg(statistics)
        return summary

    return extract


def extract_connectivity_0(circuits, edge_population, subtargets):
    """Extract connectivity from one or more  connectome
    """
    p = edge_population

    def subset(circuit_labeled):
        return subtargets.xs(circuit_labeled, level="circuit")

    def apply(extract):
        LOG.info("Apply %s to %s subtargets in %s circuits", extract.__name__, len(subtargets), len(circuits))
        result = pd.concat([extract(circuit=c, edge_population=p, subtargets=subset(l)) for l, c in circuits.items()],
                           axis=0, keys=[l for l,_ in circuits.items()], names=["circuit"])
        LOG.info("\t result of extraction: %s", len(result))
        return result

    try:
        properties = edge_population["properties"]
    except KeyError:
        edge_adjs = apply(extract_adj).rename("matrix")
        edge_props = pd.Series()
    else:
        edge_props = apply(extract_edge(properties)).rename("edge_properties")

        subtargets_connectome = (pd.concat([subtargets], keys=[p["connectome"]], names=["connectome"])
                                .reorder_levels(["circuit", "connectome", "subtarget"]))
        def matrix(row):
            connections = row.edge_properties[[Synapse.PRE_GID, Synapse.POST_GID]]
            nodes = pd.Series(row.gids, name="gid")
            return as_adjmat(connections, nodes, edge_population=p)

        edge_adjs = pd.concat([edge_props, subtargets_connectome], axis=1).apply(matrix, axis=1).rename("matrix")

    return {"adj": edge_adjs, "props": edge_props}


def extract_connectivity_batch(circuit, connectome, subtargets, properties=None):
    """..."""

    def apply(extract):
        """..."""
        LOG.info("Extract %s for %s subtargets in %s connectome %s", extract.__name__, len(subtargets)
                 , circuit, connectome)

        result = extract(circuit, connectome, subtargets)
        LOG.info("Extracted %s edges", len(result))
        return result

    if not properties:
        adjs = apply(extract_adj).rename("matrix")
        return {"adj": adjs}

    props = apply(extract_edge(properties)).rename("edge_properties")

    def matrix(row):
        """..."""
        connections = row.edge_properties[[Synapse.PRE_GID, Synapse.POST_GID]]
        nodes = pd.Series(row.gids, name="gid")
        return as_adjmat(connections, nodes, connectome)

    subtargets_connectome = (pd.concat([subtargets], keys=[connectome], names=["connectome"])
                             .reorder_levels(["circuit", "connectome", "subtarget"]))

    adjs = pd.concat([props, subtargets_connectome], axis=1).apply(matrix, axis=1).rename("matrix")

    return {"adj": adjs, "props": props}


def extract_connectivity(circuit, connectome, subtarget, properties=None, statistics=None):
    """..."""
    def apply(extract):
        """..."""
        LOG.info("Extract %s in circuit %s connectome %s subtarget %s", extract.__name__,
                 circuit, connectome, subtarget)

        result = extract(circuit, connectome, subtarget)
        LOG.info("Extracted %s edges", result.shape)
        return result

    if not properties:
        adjs = apply(extract_adj)
        return {"adj": adjs}

    props = apply(extract_edge(properties, statistics))


    Synapse_CONN = [Synapse.PRE_GID, Synapse.POST_GID]
    connections = props.index.to_frame().reset_index(drop=True) if statistics else props[Synapse_CONN]
    nodes = pd.Series(subtarget["gids"], name="gid")
    adj = as_adjmat(connections, nodes, connectome)

    return {"adj": adjs, "props": props}
