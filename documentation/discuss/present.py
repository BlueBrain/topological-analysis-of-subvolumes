

# #+RESULTS:
# [[file:./.ob-jupyter/1179885101204eb0ecc0024922e22784b9314ed0.png]]




# * Interface to TAP: How to work with the analyses data?

# We can use a ~class~ defined in ~connsense-TAP~ to interface with the data that has been extracted.

# ** Setup
# Let us setup an interactive ~Python~ session where we can run the code developed here.
# *** Introduction
# #+name: notebook-init

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

# We have run ~connsense-TAP~ for the SSCx dissemination variant /Bio-M/, extracting data that we will use to study the circuit's topology. Here are some workspaces that we use to /test-develop/ ~connsense-TAP~ for topology.
# *** Workspaces
# We have a ~connsense-TAP~ pipeline with circuit data extracted for the ~flatmap-columns~.
# #+name: notebook-workspaces

from connsense.pipeline import pipeline
from connsense.develop import parallelization as devprl

from connsense.pipeline.store import store as tap_store
from connsense.develop import topotap as devtap

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
CONNSPACE = PROJSPACE / "home/sood" / "topological-analysis-subvolumes/test/v2"

# While test-developing it will be good to have direct access to the ~connsense-TAP-store~ we will use. We will use a development version of the interface.
# *** ~connsense~ Modules
# #+name: notebook-connsense-tap

tap = devtap.HDFStore(CONNSPACE/"pipeline.yaml")
print("Configured Analyses: ")
pprint(tap.analyses)


# #+RESULTS:
# : We will plot golden aspect ratios:  1.618033988749895
# : Configured Analyses:
# : {'connectivity': {'model-params-dd2': <connsense.develop.topotap.TapDataset object at 0x7fff345e60d0>,
# :                   'simplex-counts': <connsense.develop.topotap.TapDataset object at 0x7fff345c7f70>}}


# We will use the deprecated ~connsense-TAP-HDFStore~ to load the circuit. We need the circuit for our discussion. ~connsense-TAP~ can be used without accessing the circuit itself.

otap = tap_store.HDFStore(tap._config)
circuit = otap.get_circuit("Bio_M")
