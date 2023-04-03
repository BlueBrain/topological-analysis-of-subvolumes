---
author: Vishal Sood
title: Connectivity across the `flatmap`.
---

Let us setup an interactive `Python` session where we can run the code
developed here.

``` jupyter
print("Welcome to EMACS Jupyter")
```

We will characterize the structure of activity across flatmap columns.
For this we will need to look into the `long-range` connectivity
*between* pairs of `flatmap-columns`.

# Setup

To get the notebook you will have to clone,

``` shell
git clone https://bbpgitlab.epfl.ch/conn/structural/topological-analysis-of-subvolumes.git
git checkout beta
```

To read the setup code, look in the [4](#Appendix). Here we use `noweb`
to include the code written there.

``` python
from importlib import reload
from collections.abc import Mapping
from collections import OrderedDict
from pprint import pprint, pformat
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

reload(matplotlib)
from matplotlib import pylab as plt
import seaborn as sbn

from IPython.display import display

from bluepy import Synapse, Cell, Circuit

GOLDEN = (1. + np.sqrt(5.))/2.
print("We will plot golden aspect ratios: ", GOLDEN)

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
SOODSPACE = PROJSPACE / "home/sood"
CONNSPACE = SOODSPACE / "topological-analysis-subvolumes/test/v2"
DEVSPACE  = CONNSPACE / "test" / "develop"

from connsense.develop import topotap as cnstap
tap = cnstap.HDFStore(CONNSPACE/"pipeline.yaml")
circuit = tap.get_circuit("Bio_M")
print("Available analyses: ")
pprint(tap.analyses)
circuit
```

# Long range connectivity between `flatmap-columns`

We want to summarize the *long-range* connectivity on top of
*local-connectivity* of `flatmap-columns`. We can develop a concept of a
`FlatmapColumn` as a `Python` class that can provide us with
*long-range* sources and targets of a group of node ids in another
`FlatmapColumn`,

``` python
CONNECTION_ID = ["source_node", "target_node"]
SUBTARGET_ID = ["subtarget_id", "circuit_id"]
NODE_ID = ["subtarget_id", "circuit_id", "node_id"]

def sparse_csr(connections):
    """..."""
    from scipy import sparse
    connections_counted = connections.value_counts().rename("count")
    return sparse.csr_matrix(rows=connections.source_nodes.values,
                             cols=connections.target_nodes.values)
```

``` python

def find_afferent(tap, flatmap_column, connectome):
    """..."""
    target_gids = tap.nodes.dataset.loc[flatmap_column]().gid.rename("target_gid")
    target_gids.index.rename("target_node", inplace=True)

    incoming = target_gids.apply(connectome.afferent_gids).rename("source_gids")
    subtargets = assign_subtargets(tap)
    sources = incoming.apply(subtargets.reindex)

    return (pd.concat(sources.values, keys=sources.index).fillna(-1).astype(np.int)
            .droplevel("gid").reset_index().set_index("target_node"))
```

For efferent,

``` python
def find_efferent(tap, flatmap_column, circuit, connectome):
    """..."""
    raise NotImplementedError
```

We may also want a filter of edges,

``` python
def filter_edges(tapestry, flatmap_column, circuit, connectome, direction, and_apply=None):
    """Filter afferent or efferent edges of a flatmap-column in a circuit's connectome."""
    assert direction in (Direction.AFFERENT, Direction.EFFERENT),\
        f"Invalid direction {direction}"

    affends = (find_afferent(tapestry, flatmap_column, circuit, connectome)
               .reset_index().groupby(NODE_ID).target_node.apply(list))

    def afferent(nodes):
        """Filter edges incoming from nodes."""
        source_nodes = index_subtarget(nodes)
        target_nodes = (source_nodes.apply(lambda n: tuple(n.values), axis=1)
                        .apply(lambda s: affends.loc[s]))
        return target_nodes if not and_apply else and_apply(target_nodes)

    def efferent(nodes):
        """Filter edges outgoing to nodes."""
        raise NotImplementedError("Efferent takes special care.")

    return afferent if direction == Direction.AFFERENT else efferent
```

We will need a subtarget assignment, a method that should be in tap.

``` python
def assign_subtargets(tap):
    """..."""
    def series(of_gids):
        return pd.Series(of_gids, name="gid",
                         index=pd.RangeIndex(0, len(of_gids), 1, name="node_id"))
    return (pd.concat([series(gs) for gs in tap.subtarget_gids], axis=0,
                      keys=tap.subtarget_gids.index)
            .reset_index().set_index("gid"))
```

## Simplices

A method to get them from `topology`,

``` python
def get_simplices(flatmap_column):
    subtarget_id, circuit_id = flatmap_column
    connectome_id = 0
    adj = tap.adjacency.dataset.loc[subtarget_id, circuit_id, connectome_id]()
    nodeps = tap.nodes.dataset.loc[subtarget_id, circuit_id]()
    return pd.concat([topology.list_simplices_by_dimension(adj, nodeps)],
                     keys=[(subtarget_id, circuit_id)], names=SUBTARGET_ID)


def index_subtarget(tap, flatmap_column, nodes=None):
    """..."""
    subtarget_id, circuit_id = flatmap_column

    if nodes is None or (isinstance(nodes, str) and nodes.lower() == "all"):
        nodes = tap.nodes.dataset.loc[subtarget_id, circuit_id].index.values

    return pd.DataFrame({"subtarget_id": subtarget_id, "circuit_id": circuit_id,
                         "node_id": nodes})
```

We can compute simplex lists in a the *local-connectome* of
`flatmap-columns`. We would like to know if there are `target-nodes` in
a given `flatmap-column` that are *post-synaptic* to all the nodes in a
`simplex`. We can call the number of simplices that `sink` at a
`target-node` as the `target-node`'s `sink-participation`. Analogously
we can define a `source-node`'s `source-participation` by computing the
number of `simplices` that `source` at the `source-node`.

``` python
def find_sinks(tap, flatmap_column, circuit, connectome, affends=None):
    """Find simplices that sink at each node in a flatmap-column."""

    if affends is None:
        affends = (find_afferent(tap, flatmap_column, circuit, connectome)
                   .reset_index().groupby(NODE_ID).target_node.apply(list))

    def of_source(flatmap_column, simplex_nodes):
        sdim = len(simplex_nodes)
        simplex = index_subtarget(tap, flatmap_column, simplex_nodes)
        simplex.index.rename("spos", inplace=True)
        simplex_pos = simplex.reset_index().set_index(NODE_ID)

        target_lists = (pd.concat([simplex_pos, affends.reindex(simplex_pos.index)], axis=1)
                        .set_index("spos").target_node).sort_index()
        targets = pd.concat([pd.Series(ns, name="target_node") for ns in target_lists],
                            keys=target_lists.index).droplevel(None)
        counts = targets.value_counts()
        return counts.index[counts == sdim].values

    of_source.afferent_edges = affends
    return of_source
```

How does a node in a *target* `flatmap-column` connect to `simplices` in
other `flatmap-columns`? How many *local-connnectome* simplices in a
given `flatmap-column` does a node connect to?

What about sources?

``` python
def find_sources(tap, flatmap_column, circuit, connectome, effends=None):
    """Find simplices that souce at each node in a flatmap-column."""

    if effends is None:
        effends = (find_efferent(tap, flatmap_column, circuit, connectome)
                   .reset_index().groupby(NODE_ID).target_node.apply(list))

    def of_source(flatmap_column, simplex_nodes):
        sdim = len(simplex_nodes)
        simplex = index_subtarget(tap, flatmap_column, simplex_nodes)
        simplex.index.rename("spos", inplace=True)
        simplex_pos = simplex.reset_index().set_index(NODE_ID)

        target_lists = (pd.concat([simplex_pos, affends.reindex(simplex_pos.index)], axis=1)
                        .set_index("spos").target_node).sort_index()
        targets = pd.concat([pd.Series(ns, name="target_node") for ns in target_lists],
                            keys=target_lists.index).droplevel(None)
        counts = targets.value_counts()
        return counts.index[counts == sdim].values

    of_source.afferent_edges = affends
    return of_source
```

We have not implemented `find_efferent`. We may not need it if we change
our approach.

Connectivity is between a group of source nodes and a group of target
nodes.

``` python
def is_subtarget(reference):
    """..."""
    ints = (int, np.uint8, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64)
    return (isinstance(reference, tuple) and len(reference) == 2
            and isinstance(reference[0], ints) and isinstance(reference[1], ints))


def _resolve_subtarget(tap, reference):
    """..."""
    if is_subtarget(reference):
        return reference

    s, _ = reference
    if not is_subtarget(reference=s):
        return None

    return s


def _resolve_nodes(tap, reference, indexed=True):
    """..."""
    if is_subtarget(reference):
        nodes = tap.nodes.dataset.loc[reference].index.values
        return index_subtarget(tap, reference, nodes) if indexed else nodes

    s, nodes = reference
    if not is_subtarget(reference=s):
        return None

    return index_subtarget(tap, s, nodes)


def find_edges(tap, sources=None, targets=None, *, connectome):
    """Find connectome edges from nodes among sources to nodes among targets."""
    source_nodes = _resolve_nodes(sources, indexed=True)
    target_nodes = _resolve_nodes(targets, indexed=False)

    afferent = (find_afferent(tap, _resolve_subtarget(targets), connectome)
                .reset_index().groupby(NODE_ID).target_node.apply(list))
```

# Incoming connections to a simplex

A simplex is a fully directional one represented as a vector of integer
node ids. We compute the simplices in `connsense-TAP` to be represented
as local `node-ids` which we can translate to the `global-id` (`gid`)
using the `subtarget`'s `node-properties`. Then we can look up the
`long-range` connetome's `afferent` gids, map them to the
`flatmap-columns`, and compute a scalar or vector `weight` for them.
Thus we will have a length `N` vector of `weights` for each `simplex`
(of a given dimension) in a given `flatmap-column`. Over all the columns
we have a matrix of weights that can be plotted as a `heatmap`. We can
visualize individual rows or columns over a `flatmap-grid`.

We can compute the weights based on filters. Let us develop these ideas
further in code.

``` python
def gather_inputs(circuit, subtarget, simplex, *, tap):
    """..."""
    gids = tap.
```

# Appendix
