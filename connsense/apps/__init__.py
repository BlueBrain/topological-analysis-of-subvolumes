"""Command line interfaces to run analysis jobs.
"""

from .import topological_analysis
from .import run_analysis

APPS = {"main": "tap",
        "define-subtargets": "tap-subtargets",
        "extract-voxels": "tap-atlas",
        "extract-node-types": "tap-models",
        "extract-node-populations": "tap-nodes",
        "extract-edge-populations": "tap-connectivity",
        "analyze-geometry": "tap-analysis",
        "analyze-composition": "tap-analysis",
        "analyze-connectivity": "tap-analysis"}
