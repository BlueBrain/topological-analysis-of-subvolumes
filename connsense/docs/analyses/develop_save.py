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
# Develop pipeline analyses.

"""

from pathlib import Path
from pprint import pformat
import yaml

import numpy as np
import pandas as pd


from connsense.io import logging


LOG = logging.get_logger(__name__)


"""
## Workspace

To run code while developing analyses, we will use a workspace that expects a certain
layout.

"""

WORKSPACE = (Path("/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript")
             / "topological-analysis-subvolumes" / "test")

"""
### HDF store

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
To test develop an individual analysis we will work within subfolders:
"""

def locate_test(for_analysis):
    """Path to the tests for a given analysis.
    """
    return WORKSPACE / for_analysis


"""
Let us a load the configuration for test developing simplex-lists.
"""


from connsense.io import read_config

config = read_config.read(locate_test(for_analysis="simplex-lists") / "config.json")
