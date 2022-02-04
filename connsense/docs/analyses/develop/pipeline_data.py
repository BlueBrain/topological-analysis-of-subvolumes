#!/usr/bin/env python3



# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,.pct.py:percent,.lgt.py:light,.spx.py:sphinx,md,Rmd,.pandoc.md:pandoc
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.1.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


"""
# Data input / output in the pipeline.

We test develop analysis step of the pipeline while we present the `connsense` interface to
the pipeline's data.

"""
from importlib import reload

from pathlib import Path
from pprint import pformat
import yaml

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sbn

from connsense.io import logging, time as timing
from connsense import analyze_connectivity as anzconn
from connsense.pipeline.store import store as plumbing
from connsense.analyze_connectivity import analyze
from connsense.analyze_connectivity import matrices as matrix_output
from connsense.docs import utils as docutils

LOG = logging.get_logger(__name__)

"""
For the notebook interface of this script, set the following color scheme
for the plots.
"""
import matplotlib as mpl
import matplotlib as mpl
COLOR = "whitesmoke" #for a dark Jupyter theme
#COLOR = "slategray" #for a light Jupyter Theme
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['axes.facecolor'] = 'ffffff'
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

"""
**Workspace**

To run code while developing analyses, we will use a workspace that expects a certain
layout. For each analysis the development will be done in a subfolder, and associated
be associated with a data document written during the execution of this code document.
"""

reload(docutils)

WORKSPACE = (Path("/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript")
             / "topological-analysis-subvolumes" / "test")

def locate_test(for_analysis):
    """Path to the tests for a given analysis.
    """
    return WORKSPACE / for_analysis


DOC = docutils.Document("Pipeline analysis of simplex-lists.",
                        locate_test("simplices") / "document" / timing.stamp(now=True))

"""
**HDF store**

To test run analyses we need an HDF store. We have some available in the worksepace.
These are saved in a subfolder and configured in a YAML file:
"""

def locate_hdf_stores(label=None):
    """Locate an HDF store.
    """
    with open(WORKSPACE / "store" / "available.yaml", 'r') as avail:
        config = yaml.load(avail, Loader=yaml.FullLoader)

    root = Path(config["root"])
    hdfs = {hdf["label"]: root / hdf["file"] for hdf in config["hdfs"]}

    return hdfs[label] if label else hdfs


"""
To test develop analyses we will use a store that was created towards the end of 2021,
and contains adjacency matrices for all the subtargets when the flatspace is tiled
with $230\mu m$ hexagons.
"""

STORE = locate_hdf_stores()

LOG.info("HDFs available in %s workspace:", WORKSPACE)
for i, (label, path) in enumerate(STORE.items()):
    LOG.info("(%s) %s at %s: ", i, label, path)


"""
Let us a load the configuration for test developing simplex-lists.
"""
from connsense.io import read_config
at_path_config = locate_test(for_analysis="simplices") / "config.json"
config = read_config.read(at_path_config)

"""
We can investigate the pipeline store using the tools provided by `connsense`:
"""
from connsense import pipeline
topaz = pipeline.TopologicalAnalysis(at_path_config)
tap = topaz.data

"""
We can take a peek at what the TAP(/Topological Analaysis pipeline) instance `tap`
contains.
"""

LOG.info("Data groups in the pipeline store: \n %s", pformat(tap._groups))


"""
Since we are testing analysis of simplex lists, we should expect only simplex-lists
among configured analyses:
"""

LOG.info("Configured analysesL\n %s",
         pformat(config["parameters"]["analyze-connectivity"]))


"""

## Subtargets

For each circuit, the store containts data for (flatmap) subtargets. In this section
we discuss these subtargets, and develop methods to choose subtargets to run a pipeline with.

"""

DOC.append_section("Subtargets", at_level=1)

"""
To test develop, we would like to be able to choose and pass selected subtargets to
run the pipeline CLI. The selected subtargets could be added to a config and saved to
disc before invocation of the pipeline CLI, or we could pass the subtargets from the CLI
itself.

Let us proceed with the second option. However before we implement such a CLI,
let us list the subtargets by their size.

"""

def get_subtargets(tap, circuit="Bio_M", connectome="local"):
    """How many neurons and edges does each subtarget in circuit contain?

    Arguments
    ----------
    tap : A Topological Analysis Pipeline data store.
    circuit : Label for a circuit in the store.
    """
    def measure_subtarget(s):
        """..."""
        adj = tap.pour_adjacency(circuit=circuit, subtarget=s, connectome=connectome)
        return pd.Series({"nodes": adj.shape[0], "edges": np.sum(adj)})

    subtargets = tap.pour_subtargets(circuit=circuit)
    return (subtargets.apply(measure_subtarget)
            .sort_values(by="nodes", ascending=False)
            .join(subtargets)
            .set_index("subtarget"))



"""
What kind of subtargets are there in store to play with?
"""

subtargets = get_subtargets(tap)
LOG.info("Here are %s subtargets to choose from, the largest of these has %s nodes",
         len(subtargets), subtargets.nodes.max())
figure = plt.figure(figsize=(15, 12))
graphic = (figure, sbn.histplot(subtargets.nodes))
DOC.append_figure(graphic, "subgraphs")


"""
The distribution is not trivial. The smaller subtargets (to judge visually,
sizes less than 5000) can be merged with some criterion. Some sort of optimized tiling
of the flatspace with boxes of varying physical sizes, but about the same number of cells.

**TODO** : Tile the circuit flatspace with boxes of varying sizes keeping the number of cells
in each box about the same -- so minimize the variance of number of cells in each box.


## HDF outputs for analyses

HDF cannot be written in parallel. Due to this constraint, we will have to generate a temporary HDF
for each batch and combine them later.
Let us test-develop by following the steps of running the pipeline steps by hand:

"""

reload(anzconn)
neurons = anzconn.load_neurons(config["paths"])

LOG.info("Number of neuron properties to run pipeline: %s", neurons.shape[0])
neurons.head()


###############################################################################

###############################################################################
