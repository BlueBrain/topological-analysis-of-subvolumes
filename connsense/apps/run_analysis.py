#!/usr/bin/env python3

"""ConnSense: An app to run a single analysis for the analyze-connectivity step.
This is will be use to generate multi-node launchscripts.

This is a work in progress. (Vishal Sood 20220220)
"""

from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

from connsense import pipeline
from connsense.io import logging
from connsense.io.write_results import (read as read_results,
                                        read_toc_plus_payload,
                                        write as write_dataframe,
                                        write_toc_plus_payload,
                                        default_hdf)
from connsense.apps import topological_analysis as topaz
from connsense.pipeline import workspace
from connsense import analyze_connectivity as anzconn
from connsense.analyze_connectivity import analyze

STEP = "analyze-connectivity"
LOG = logging.get_logger("Toplogical analysis of flatmapped subtargets.")


def resolve_analysis(argued, against_config):
    """The argued analysis (which can be only 1) must have been configured.
    """
    analyses = anzconn.get_analyses(against_config, as_dict=True)
    if argued.quantity not in analyses:
        raise RuntimeError(f"Analysis {argued.quantity} must have been configured"
                           f" in {[a.name for a in analyses.keys()]}")

    return analyses[argued.quantity]


def main(argued):
    """..."""
    LOG.info("Initialize the topological analysis pipeline.")

    at_path = Path(argued.configure)
    config = pipeline.TopologicalAnalysis.read_config(at_path)
    c = config
    paths = anzconn._check_paths(config["paths"])
    input_paths = paths["input"]
    output_paths = paths["output"]

    in_basedir = Path(argued.batches or Path.cwd())
    neurons = anzconn.load_neurons(input_paths)

    toc_adjs = anzconn.load_adjacencies(input_paths, from_batch=in_basedir)

    LOG.info("Resolve analysis to run")
    analysis = resolve_analysis(argued, against_config=c)

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))

    LOG.info("Compute analysis %s on %s subtargets, saving results to (%s, %s)",
             analysis.name, len(toc_adjs), in_basedir, hdf_group)

    result = analyze.dispatch_single_node(to_compute=analysis, batched_subtargets=toc_adjs,
                                          neuron_properties=neurons, action=argued.action,
                                          to_save=(in_basedir, hdf_group))
    LOG.warning("DONE running analysis %s for batch %s of %s subtargets saving (%s, %s).",
                analysis.name, argued.batches, len(toc_adjs), in_basedir, hdf_group)
    return result


if __name__ == "__main__":

    LOG.warning("Analyze circuit subtarget topology.")

    parser = ArgumentParser(description="Topological analysis of flatmapped subtargets.")

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

    parser.add_argument("-q", "--quantity", required=True,
                        help=("Name of the analysis to run, that should appear in the config."))

    parser.add_argument("-m", "--mode", required=False, default=None,
                        help=("Specify how the action should be performed. should be done\n"
                              "For example:\n"
                              "tap --configure=config.json --parallelize=parallel.json \\"
                              "    --mode=prod run\n"
                              "to run in production mode."))
    parser.add_argument("-b", "--batches", required=False, default=None,
                        help=("Location of a HDF containing a dataframe that "
                              "assigns batches to subtargets.\n"
                              "Default behavior will assume that the current working directory"
                              " is where the batches HDF are."))

    parser.set_defaults(test=False)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
