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
from connsense.pipeline.parallelization import parallelization as prl
from connsense.pipeline.store import store as tap_store

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
CONNSPACE = PROJSPACE / "home/sood" / "topological-analysis-subvolumes/test/v2"

topaz = pipeline.TopologicalAnalysis(CONNSPACE/"pipeline.yaml", CONNSPACE/"runtime.yaml")
tap = tap_store.HDFStore(topaz._config)
circuit = tap.get_circuit("Bio_M")
print("Available analyses: ")
pprint(tap.analyses)

import connsense.pipeline.pipeline
import connsense.pipeline.store.store

def reload_modules():
    """..."""
    reload(connsense.pipeline.pipeline)
    reload(connsense.pipeline.store.store)
