"""ConnSense: An app to run connectome utility pipeline.
"""
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

from connsense import pipeline
from connsense.io import logging
from connsense.pipeline import workspace

LOG = logging.get_logger("Toplogical analysis of flatmapped subtargets.")


def _read_steps(argued):
    """..."""
    return [s.strip().lower() for s in argued.step.split(';')] if argued.step else None


def _read_output_in_config(c, and_argued_to_be):
    """..."""
    a = and_argued_to_be
    if not a:
        return c["paths"]["output"]

    argued = Path(a)
    assert argued.suffix == "h5", f"Not a HDF h5: {argued}"

    if not argued.is_absolute():
        return c["paths"]["root"] / argued
    return argued


def is_to_init(action):
    """..."""
    return action.lower() in ("init", "initialize")


def lower(argument):
    """..."""
    return argument.lower() if argument else None


SUBSTEPS = {"define-subtargets": "grids",
            "extract-neurons": None,
            "evaluate-subtargets": "metrics",
            "extract-connectivity": "connectomes",
            "randomize-connectivity": "algorithms",
            "analyze-connectivity": "analyses"}


def parameterize_substeps(s, in_config):
    """..."""
    LOG.info("parameterize substeps for step %s in config \n %s", s,
             pformat(in_config["parameters"][s]))

    param = SUBSTEPS[s]
    if not param:
        return None

    return in_config["parameters"][s][param]


def check_step(as_argued, against_config):
    """Check the argued step and substep against what is possible with a config.
    The value of checking the argued pipeline step will be a tuple that can be used
    to initialize a workspace to run the pipeline.
    Each pipleine step's run will be done from a sub-folder of the step's folder.
    If the argued step does not have any substeps defined in the config,
    the work should be done in the step folder itself.
    Thus for such cases return (step, "_"), to distinguish from the case where
    the workspace folder of a step is to be initialized, but none of its substep in particular.
    """
    c = against_config
    s = lower(as_argued.step); ss = lower(as_argued.substep)

    if not s:
        assert not ss, f"Substep {ss} of step None maketh sense None"
        return (None, None)

    parameters = parameterize_substeps(s, in_config=c)

    if not parameters:
        assert not ss, f"Cannot argue a `substep` if step {s} takes no substep parameters."
        return (s, "_")

    if not ss:
        return (s, None)

    assert ss in parameters, ("f{substep} is not a substep of {step}."
                              f" Provide one of {pformat(parameters)}")

    return (s, ss)


def get_current(action, config, step, substep, with_parallelization=None):
    """..."""
    current_run = (workspace.initialize if is_to_init(action)
                   else workspace.current)
    return current_run(config, step, substep, with_parallelization)


def main(argued):
    """..."""
    LOG.info("Initialize the topological analysis pipeline.")
    at_path = Path(argued.configure)
    c = pipeline.TopologicalAnalysis.read_config(at_path)
    a = argued
    c["paths"]["output"] = _read_output_in_config(c, and_argued_to_be=a.output)

    p = pipeline.TopologicalAnalysis.read_parallelization(argued.parallelize)
    s, ss = check_step(argued, against_config=c)
    current_run = get_current(action=a.action, config=c, step=s, substep=ss,
                              with_parallelization=p)
    LOG.info("Workspace initialized at %s", current_run)

    if is_to_init(argued.action):
        return current_run

    topaz = pipeline.TopologicalAnalysis(config=c, parallelize=p, mode="run",
                                         workspace=current_run)

    LOG.info("Initialized a run of the TAP pipelein configration %s parallelization %s",
             argued.configure, argued.parallelize)

    LOG.info("Run the pipeline.")
    steps = _read_steps(argued)

    if not steps:
        raise ValueError("Provide one or more step to run, 'all` to run all the steps.")

    if steps == "all":
        raise NotImplementedError("An automated run of all steps."
                                  " Please run individual steps manually from the CLI")

    result = topaz.run(steps, sample=argued.sample, output=argued.output,
                       dry_run=argued.test)

    LOG.info("DONE running pipeline: %s", result)

    return result


# if __name__ == "__main__":ure

#     LOG.warning("Parse arguments.")
#     parser = ArgumentParser(description="Topological analysis of flatmapped subtargets")

#     parser.add_argument("steps",
#                         help=("Pipeline step to run. Use `all` to run the full pipeline.\n"
#                               "To initialize a workspace: topological_analysis.py init config.json"
#                               "To run a subset of steps, chain them as string using semicolon to join.\n"
#                               "For example 'define-subtargets;extract-neurons'. Spaces will be removed."))

#     parser.add_argument("config",
#                         help="Path to the configuration to run the pipeline.")

#     parser.add_argument("-j", "--njobs", type=int
#                        ,help="Number of jobs in parallel.")

#     parser.add_argument("--output",
#                         help="Path to the directory to output in.", default=None)

#     parser.add_argument("--sample",
#                         help="A float to sample subtargets with", default=None)

#     parser.add_argument("--dry-run", dest="test",  action="store_true",
#                         help=("Use this to test the pipeline's plumbing "
#                               "before running any juices through it."))
#     parser.set_defaults(test=False)

#     args = parser.parse_args()

#     LOG.warning(str(args))
#     main(args)

if __name__ == "__main__":

    LOG.warning("Analyze circuit subtarget topology.")

    parser = ArgumentParser(description="Topological analysis of flatmapped subtargets.")

    parser.add_argument("action",
                        help=("A pipeline (step) action to do."
                              " Following is a list of actions."
                              " The action may be expected to apply to all the pipeline steps,"
                              " unless otherwise indicated.\n"
                              "\t(1) init: to setup and intiialize.\n"
                              "\t(2) run: to run..., initializing if not already done\n"
                              "\t(3) resume: resume from the current state"))

    parser.add_argument("step", nargs='?', default=None,
                        help=("Pipeline step to run. Use `all` to run the full pipeline.\n"
                              "To run a subset of steps, chain them as string using semicolon to join.\n"
                              "For example 'define-subtargets;extract-neurons'. Spaces will be removed.\n"
                              "Argument `steps` may be skipped for initializing the pipeline"
                              "To initialize a workspace: topological_analysis.py init config.json "
                              "will create a directory to run the pipeline in ---"
                              " but without any subfolders for individual pipeline steps. "))

    parser.add_argument("substep", nargs='?', default=None,
                        help=("Some of the pipeline steps can be duvided into "
                              "individual indepedent sub-steps.\n"
                              "The simplest case is when a bunch of analyses must be run, "
                              "each for each subtarget independently of the others.\n"
                              "Provide a pipeline step's sub-step to run only that one.\n"
                              "For example,\n"
                              "tap init analyze-connectivity simplices \\\n"
                              "\t--configure=pipeline.json \\\n"
                              "\t--parallelize=parallel.json"))

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

    parser.add_argument("--output",
                        help="Path to the directory to output in.", default=None)

    parser.add_argument("--sample",
                        help="A float to sample subtargets with", default=None)

    parser.add_argument("--dry-run", dest="test",  action="store_true",
                        help=("Use this to test the pipeline's plumbing "
                              "before running any juices through it."))
    parser.set_defaults(test=False)

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
