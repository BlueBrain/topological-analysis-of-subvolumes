# %% [markdown]
"""# Parallelization scheme for `connsense-TAP`

We develop a parallelization scheme for `connsense-TAP` computations.

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

# %% [markdown]
"""
We will need some ~connsense~ modules for our experiments
"""
# %% [code]
from connsense.pipeline import pipeline
from connsense.pipeline.parallelization import parallelization as prl
from connsense.pipeline.store import store as tap_store

# %% [markdown]
"""
We can set paths to load data, and to save the results of our experiments. Paths listed below are to artefacts associated with a SSCx-Dissemination circuit.
"""
# %% [code]
ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
CONNSPACE = PROJSPACE / "home/sood" / "topological-analysis-subvolumes/test/v2"
#CONNSPACE = (PROJSPACE / "home/sood" / "portal/develop/factology-v2/analyses/connsense"/
#             "redefine-subtargets/create-index/morphology-mtypes")

# %% [markdown]
"""
For our experiments, we will need a circuit, an object to run / investigate the pipeline, and another to load / investigate the computated data.
"""
topaz = pipeline.TopologicalAnalysis(CONNSPACE/"pipeline.yaml", CONNSPACE/"runtime.yaml")
tap = tap_store.HDFStore(topaz._config)
print("Available analyses: ")

# %% [markdown]
"""Load a connsense-TAP to analyze topology of a circuit
"""
# %% [code]

from connsense.develop import (topotap as topotap_store, parallelization as devprl)
reload(topotap_store)
topotap = topotap_store.HDFStore(CONNSPACE/"pipeline.yaml")
print("Available analyses: ")
pprint(topotap.analyses)

computation = "analyze-connectivity/simplex-counts"
print("Let test develop computation of")
pprint(prl.describe(computation))

params = prl.parameterize(*prl.describe(computation), topaz._config)
print("Using parameters")
pprint(params)

computation = "analyze-connectivity/simplex-counts"
params = devprl.parameterize(*devprl.describe(computation), topaz._config)

print("inputs for simplex-counts:\n")
pprint({k: v for k, v in params["input"].items() if k not in ("transformations", "slicing")})

print("\nwith transformations\n")
pprint(params["input"]["transformations"])

inputs = devprl.generate_inputs(computation, topaz._config)

display(inputs)

subtargets_per_control = inputs.groupby("control").size()
display(subtargets_per_control)

input_quantities_0 = inputs.iloc[0]()
for variable, value in input_quantities_0.items():
    print("input %s: \n%s"%(variable, type(value)))
