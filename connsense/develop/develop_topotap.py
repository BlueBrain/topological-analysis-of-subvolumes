# %% [markdown]
"""# Test Develop a Circuit Factology
"""

# %% [code]
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
GOLDEN = (1. + np.sqrt(5.))/2.

from IPython.display import display

from bluepy import Synapse, Cell, Circuit

print("We will plot golden aspect ratios: ", GOLDEN)

from connsense.pipeline import pipeline
from connsense.develop import parallelization as devprl

from connsense.pipeline.store import store as tap_store
from connsense.develop import topotap as devtap

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
SOODSPACE = PROJSPACE / "home/sood"
CONNSPACE = SOODSPACE / "topological-analysis-subvolumes/test/v2"

PORTALSPACE = (SOODSPACE / "portal" / "develop" / "factology-v2" / "analyses/connsense"
               / "redefine-subtargets/create-index/morphology-mtypes")
EXPTLSPACE = PORTALSPACE / "experimental"

topaz = pipeline.TopologicalAnalysis(CONNSPACE/"pipeline.yaml", CONNSPACE/"runtime.yaml")
tap = tap_store.HDFStore(topaz._config)
circuit = tap.get_circuit("Bio_M")

topotap = devtap.HDFStore(CONNSPACE/"pipeline.yaml")
print("Available analyses: ")
pprint(topotap.analyses)

# %% [markdown]
"""Load a connsense-TAP to analyze topology of a circuit
"""
# %% [code]

from connsense.develop import topotap as topotap_store
reload(topotap_store)
topotap = topotap_store.HDFStore(CONNSPACE/"pipeline.yaml")
print("Available analyses: ")
pprint(topotap.analyses)

# %% [markdown]
"""## Subtargets in connsense-TAP
"""
# %% [code]

topotap.subtargets

# %% [markdown]
"""## Nodes in connsense-TAP
"""
# %% [code]

topotap.nodes.dataset

# %% [markdown]
"""Contents of nodes
"""
# %% [code]

topotap.nodes.dataset.iloc[0].get_value().info()

# %% [markdown]
"""Contents of nodes
"""
# %% [code]

topotap.nodes(subtarget="R19;C0", circuit="Bio_M").info()

# %% [markdown]
"""Nodes of a subtarget
"""
# %% [code]

topotap.nodes(subtarget="R19;C0").info()

# %% [markdown]
"""## Adjacency datasets
"""
# %% [code]
topotap.adjacency["local"].dataset

# %% [markdown]
""" Adjacency of a subtarget
"""
# %% [code]
topotap.adjacency["local"].dataset.iloc[0].get_value()

topotap.adjacency["local"](subtarget="R19;C0")

# %% [markdown]
"""## Analyses
"""
# %% [code]
pprint(topotap.analyses)

# %% [markdown]
"""Simplex counts
"""
# %% [code]
simplex_counts = topotap.analyses["connectivity"]["simplex-counts"]
simplex_counts.dataset

# %% [markdown]
"""Simplex counts
"""
# %% [code]
simplex_counts = topotap.analyses["connectivity"]["simplex-counts"]
simplex_counts("R19;C0")

# %% [markdown]
"""Load a connsense-TAP to analyze topology of a circuit
"""
# %% [code]

from connsense.develop import topotap as topotap_store
reload(topotap_store)
topotap = topotap_store.HDFStore(CONNSPACE/"pipeline.yaml")
print("Available analyses: ")
pprint(topotap.analyses)

# %% [markdown]
"""## Subtargets in connsense-TAP
"""
# %% [code]

topotap.subtargets

# %% [markdown]
"""## Nodes in connsense-TAP
"""
# %% [code]

topotap.nodes.dataset

# %% [markdown]
"""Contents of nodes
"""
# %% [code]

topotap.nodes.dataset.iloc[0].get_value().info()

# %% [markdown]
"""Contents of nodes
"""
# %% [code]

topotap.nodes(subtarget="R19;C0", circuit="Bio_M").info()

# %% [markdown]
"""Nodes of a subtarget
"""
# %% [code]

topotap.nodes(subtarget="R19;C0").info()

# %% [markdown]
"""## Adjacency datasets
"""
# %% [code]
topotap.adjacency["local"].dataset

# %% [markdown]
""" Adjacency of a subtarget
"""
# %% [code]
topotap.adjacency["local"].dataset.iloc[0].get_value()

# %% [markdown]
"""## Analyses
"""
# %% [code]
pprint(topotap.analyses)

# %% [markdown]
"""Simplex counts
"""
# %% [code]
simplex_counts = topotap.analyses["connectivity"]["simplex-counts"]
simplex_counts.dataset

# %% [markdown]
"""Simplex counts
"""
# %% [code]
simplex_counts = topotap.analyses["connectivity"]["simplex-counts"]
simplex_counts("R19;C0")

topotap.subtargets
