#!/usr/bin/env python3

"""ConnSense: An app to run a single analysis for the analyze-connectivity step.
This is will be use to generate multi-node launchscripts.

This is a work in progress. (Vishal Sood 20220220)
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
from connsense.io import read_config
from connsense.apps import topological_analysis as topaz
from connsense.pipeline import workspace
from connsense.pipeline.store import HDFStore as TAPStore
from connsense.randomize_connectivity.algorithm import SingleMethodAlgorithmFromSource
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


def read_control(in_file):
    """..."""
    if not in_file.exists(): return None
    with open(in_file, 'r') as f : description = json.load(f)

    LOG.info("Read controls with description: \n%s", pformat(description))

    name = description.pop("name")
    return SingleMethodAlgorithmFromSource(name, description)


def update_algorithm(in_toc, to_value):
    """...Change, from original, to the value specified.
    """
    return pd.concat([in_toc.droplevel("algorithm")], keys=[to_value], names=["algorithm"])


def apply_control_series(in_file, to_toc, using_batches):
    """...Controls applied `to_toc` which is a series...
    Replaced by the newer version that applies controls to rows of a dataframe
    so that information about the subvolume represented by a TOC entry can be passed to the control

    TODO: REMOVE
    """
    control = read_control(in_file)
    if not control:
        LOG.info("No controls to apply")
        return (to_toc, using_batches)

    LOG.info("Apply control %s to a %s entry TOC: \n%s", control.name,  len(to_toc), pformat(to_toc))
    of_controlled_values = to_toc.apply(lazy_random(control))
    controlled_toc = update_algorithm(in_toc=of_controlled_values, to_value=control.name)
    LOG.info("DONE applying controls: \n%s", pformat(controlled_toc))
    controlled_batches = update_algorithm(using_batches, to_value=control.name)
    return (controlled_toc, controlled_batches)


def lazy_random(control, among_neurons, pipeline_store=None):
    """Prepare a method to apply to a row containing a batch of matrix from a TOC.
    """
    from connsense.analyze_connectivity.randomize import LazyRandomMatrix

    def apply_to_batch(in_row):
        """This method will need the queried row to contain circuit and subtarget in its entires.
        This can be achieved by reseting the index on the Series that provides
        the values for `matrix, batch, compute_node`.
        """
        return LazyRandomMatrix(in_row.matrix, among_neurons.loc[in_row.circuit, in_row.subtarget],
                                using_shuffling=control, name=in_row.name, tapping=pipeline_store)

    return apply_to_batch


def apply_control(in_file, to_toc, among_neurons, using_batches, using_cache):
    """..."""
    to_batched = pd.concat([to_toc, using_batches], axis=1)

    control = read_control(in_file)
    if not control:
        LOG.info("No controls to apply")
        return to_batched

    LOG.info("Apply control %s to %s entry TOC: \n%s", control.name, len(to_toc), pformat(to_toc))
    of_controlled_values = (to_batched.reset_index().set_index(to_batched.index)
                            .apply(lazy_random(control, among_neurons, using_cache), axis=1)
                            .rename("matrix"))
    controlled_toc = update_algorithm(in_toc=of_controlled_values, to_value=control.name)
    controlled_batches = update_algorithm(using_batches, to_value=control.name)

    LOG.info("Done applying control %s", control.name)
    return pd.concat([controlled_toc, controlled_batches], axis=1)


def get_parser():
    """A parser to interpret CLI args...
    """
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

    return parser


def main(argued=None):
    """..."""
    LOG.info("Run a TAP analysis.")

    if not argued:
        parser = get_parser()
        argued = parser.parse_args()

    at_path = Path(argued.configure)
    config = pipeline.TopologicalAnalysis.read_config(at_path)
    c = config
    input_paths, output_paths = anzconn._check_paths(config)

    in_basedir = Path(argued.batches or Path.cwd())
    neurons = anzconn.load_neurons(input_paths)

    original, assigned = anzconn.load_adjacencies(input_paths, from_batch=in_basedir)
    pipeline_store = TAPStore(config)
    batched = apply_control(in_basedir/"control.json", to_toc=original, among_neurons=neurons,
                            using_batches=assigned, using_cache=pipeline_store)

    analysis = resolve_analysis(argued, against_config=c)
    LOG.info("Run analysis %s for a %s batched adjacencies: \n%s", analysis, len(batched),
             pformat(batched))

    _, hdf_group = output_paths["steps"].get(STEP, default_hdf(STEP))

    LOG.info("Compute analysis %s on %s subtargets, saving results to (%s, %s)",
             analysis.name, len(batched), in_basedir, hdf_group)

    result = analyze.dispatch_single_node(analysis, batched, neurons,
                                          to_tap=pipeline_store, to_save=(in_basedir, hdf_group))
    LOG.warning("DONE running analysis %s for batch %s of %s subtargets saving (%s, %s).",
                analysis.name, argued.batches, len(batched), in_basedir, hdf_group)
    return result


if __name__ == "__main__":

    LOG.warning("Analyze circuit subtarget topology.")

    parser = get_parser()
    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
