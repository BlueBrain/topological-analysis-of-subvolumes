#!/usr/bin/env python3

"""Extract connectivity using `bluepy`.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from bluepy import Cell, Synapse, Direction

from ..io import logging

STEP = "extract-connectivity"

LOG = logging.get_logger(STEP)


def find_connectome(c, in_circuit):
    """..."""
    return  in_circuit.connectome if c == "local" else in_circuit.projection(c)


def iter_afferent(to_subtarget, in_connectome_labeled, of_circuit, as_connections=False):
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


def get_connections(to_subtarget, in_connectome_labeled, of_circuit):
    """..."""
    iterc = iter_afferent(to_subtarget, in_connectome_labeled, of_circuit, as_connections=True)
    return pd.concat(list(iterc))


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

    edges = pd.DataFrame({"source": source_ids, "target": target_ids}, index=pd.MultiIndex.from_frame(connections))

    return (edges, nodes_idx)


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


def extract_adj(circuit, connectome, subtargets):
    """..."""
    LOG.info("Extract connectivity for circuit %s connectome %s, subtargets %s",
             circuit, connectome, subtargets.index.values)

    edges_are_intrinsic = connectome in ("local", "intra_SSCX_midrange_wm")

    def extract_subtarget(gids):
        gids = pd.Series(gids, name="gid")
        connections = get_connections(to_subtarget=gids, in_connectome_labeled=connectome, of_circuit=circuit)
        return as_adjmat(connections, gids, edges_are_intrinsic)

    connectivity = subtargets.apply(extract_subtarget)
    return pd.concat([connectivity], axis=0, keys=[connectome], names=["connectome"])


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


def extract_edge(properties):
    """..."""
    synprops = [Synapse.PRE_GID, Synapse.POST_GID] + [Synapse[p.upper()] for p in properties]

    def extract(circuit, connectome, subtargets):
        """..."""
        LOG.info("Extract properties for circuit %s, subtarget \n%s", circuit, subtargets.index.values)

        edges_are_intrinsic = connectome in ("local", "intra_SSCX_midrange_wm")

        circuit_connectome = find_connectome(connectome, circuit)

        def extract_subtarget(s):
            return (circuit_connectome.pathway_synapses(s if edges_are_intrinsic else None, s, synprops)
                    .reset_index().rename(columns={"index": "Synapse_ID"}))

        values = subtargets.apply(extract_subtarget)
        return (pd.concat([values], keys=[connectome], names=["connectome"])
                .reorder_levels(["circuit", "connectome", "subtarget"]))

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


def extract_connectivity(circuit, connectome, subtargets, properties=None):
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
