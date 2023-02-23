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

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
SOODSPACE = PROJSPACE / "home/sood"
CONNSPACE = SOODSPACE / "topological-analysis-subvolumes/test/v2"
DEVSPACE = CONNSPACE / "test" / "develop"

PORTALSPACE = (SOODSPACE / "portal" / "develop" / "factology-v2" / "analyses/connsense"
               / "redefine-subtargets/create-index/morphology-mtypes")
EXPTLSPACE = PORTALSPACE / "experimental"

from connsense.develop import parallelization as cnsprl, topotap as cnstap

tap = cnstap.HDFStore(CONNSPACE/"pipeline.yaml")
circuit = tap.get_circuit("Bio_M")
print("Available analyses: ")
pprint(tap.analyses)
