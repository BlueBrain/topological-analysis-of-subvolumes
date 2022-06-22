#!/usr/bin/env python3

"""Extract subtargets' connectivity.
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from pprint import pformat
import json

import pandas as pd

from connsense import pipeline
from connsense.io import logging
from connsense.io.write_results import (read as read_results,
                                        read_toc_plus_payload,
                                        write as write_dataframe,
                                        write_toc_plus_payload,
                                        default_hdf)
from connsense.extract_connectivity.extract import extract_subtargets as extract_connectivity


STEP = "extract-connectivity"
LOG = logging.get_logger("Toplogical analysis of flatmapped subtargets.")


def get_parser():
    """..."""
    parser = ArgumentParser(description="Topological analysis of flatmapped subtargets.",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument("action",
                        help=("A pipeline (step) action to do."
                              " Following is a list of actions."
                              " The action may be expected to apply to all the pipeline steps,"
                              " unless otherwise indicated.\n"
                              "\t(1) run: to run..., initializing if not already done\n"
                              "\t(2) resume: resume from the current state\n"))

    parser.add_argument("-c", "--configure", required=True,
                        help=("Path to the (JSON) configuration that describes what to run.\n"
                              "The config should specify the input and output paths,"
                              "  and parameters  for each of the pipeline steps."))

    parser.add_argument("-x", "--connectome", required=True,
                        help=("Type of the connectome to extract, that should appear in the config."))

    parser.set_defaults(test=False)

    return parser


def main(argued=None):
    """..."""
    LOG.info("Extract TAP subtargets connectivity")

    if not argued:
        parser = get_parser()
        argued = parser.parse_args()

    at_path = Path(argued.configure)
    cfg = pipeline.TopologicalAnalysis.read_config(at_path)

    if argued.action == "run":
        output = extract_connectivity(cfg, argued.connectome)
        LOG.info("Extracted connectivity at %s", output)
        return output

    raise NotImplementedError(f"action {argued.action}")


if __name__ == "__main__":
    LOG.warning("Extract TAP subtargets connectivity.")

    parser = get_parser()
    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
