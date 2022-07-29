#!/usr/bin/env python3

"""Extract node types from the CLI.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from pprint import pformat

from connsense import pipeline
from connsense.io import logging
from connsense.extract_node_types.extract import extract_node_types


STEP = "extract-node-types"

LOG = logging.get_logger(STEP)


def get_parser():
    """..."""
    parser = ArgumentParser(description="Extract node types", formatter_class=RawTextHelpFormatter)

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

    parser.add_argument("-p", "--parallelize", required=False,
                        help=("Path to the (JSON) configuration that describes how to compute.\n"
                              "The entries in this configuration will provide parameters such as\n"
                              "paralellization parameters, whether to parallilize on a single node or multiple.\n"
                              "This configuraiton must be provided for parallilization of the tasks\n"
                              "Each pipeline step missing in the config will be run serially on a single node.\n"
                              "If the config itself is missing, all steps will be run serially."))

    parser.add_argument("-x", "--modeltype", required=True,
                        help=("A tag to name the class of models used to define a node-type"
                              " to extract, that should appear in the config."))

    parser.add_argument("-b", "--batch", required=False,
                        help=("Batch of input subtargets of a parallel process to run on a single compute nodes.\n"
                              "The batches will be run from a location read from the config."))

    parser.add_argument("-o", "--output", required=False,
                        help=("store the results at this path instead of the configured root."))

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
        output = extract_node_types(cfg, substep=argued.modeltype,
                                    for_batch=Path(argued.batch) if argued.batch else None, output=argued.output)
        LOG.info("Extracted node-type at %s", output)
        return output

    raise NotImplementedError(f"action {argued.action}")


if __name__ == "__main__":
    LOG.warning("Extract node-types.")

    parser = get_parser()
    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
