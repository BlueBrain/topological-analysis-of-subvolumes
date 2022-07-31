#!/usr/bin/env python3

"""Extract data on node types.
"""

from connsense.io import read_config
from connsense.pipeline.pipeline import PARAMKEY
from connsense.pipeline.parallelization import run_multiprocess

STEP = "extract-node-types"


def setup(config, substep=None, in_mode=None, parallelize=None, output=None, **kwargs):
    """..."""
    modeltype = substep
    config = read_config.read(config)

    extractions = config["parameters"][STEP][PARAMKEY[STEP]]
    assert modeltype in extractions, f"NOT-CONFIGURED {modeltype}"
