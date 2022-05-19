"""ConnSense: An app to run connectome utility pipeline.
"""
from collections import OrderedDict
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from pprint import pformat

from connsense import pipeline
from connsense.io import logging, read_config
from connsense.pipeline import workspace

LOG = logging.get_logger("Toplogical analysis of flatmapped subtargets.")


def _read_steps(argued):
    """Read the pipeline steps from arguments.

    NOTE: Individual steps of the pipeline are quite complex, and during the development
    so far (20220218) we have tested mosty for `analyze-connevtivity` substeps.
    We will come back here once we are confident about all of our analyses  ---
    and logging so that we can be understand the state of the pipeline.
    """
    return [s.strip().lower() for s in argued.step.split(';')] if argued.step else None


def _read_step(argued):
    """Read the step provided at the CLI invocation.
    """
    return argued.step


def _read_substeps(argued):
    """Read the pipeline substeps from arguments.

    NOTE: Individual substeps of the pipeline are quite complex, and during the development
    so far (20220218) we have tested mosty for `analyze-connevtivity` substeps.
    We will come back here once we are confident about all of our analyses  ---
    and logging so that we can be understand the state of the pipeline.
    """
    return [s.strip().lower() for s in argued.step.split(';')] if argued.substep else None


def _read_substep(argued):
    """Read the single substep provided at the CLI invocation.
    """
    return argued.substep


def _read_output_in_config(c, and_argued_to_be):
    """...
    "output": {"store": output_hdf, "steps": output_steps}}
    """
    a = and_argued_to_be
    output = c["paths"]["output"]
    if not a:
        return output

    try:
        argued = Path(a)
    except TypeError:
        raise TypeError("The argued output must be path to a hdf.h5 store")

    assert argued.suffix == ".h5", f"Not a HDF h5: {argued} suffix {argued.suffix}"

    if not argued.is_absolute():
        argued = c["paths"]["root"] / argued

    output["store"] = argued
    return output


def is_to_init(action):
    """..."""
    return action.lower() in ("init", "initialize")


def lower(argument):
    """..."""
    return argument.lower() if argument else None


SUBSTEPS = OrderedDict([("define-subtargets", "grids"),
                        ("extract-neurons", None),
                        ("evaluate-subtargets", "metrics"),
                        ("extract-connectivity", "connectomes"),
                        ("randomize-connectivity", "controls"),
                        ("analyze-connectivity", "analyses")])


def parameterize_substeps(s, in_config):
    """..."""
    parameters = in_config["parameters"]
    sections = list(parameters.keys())
    LOG.info("Parameterize %s among steps %s in config ", s, pformat(sections))

    param = SUBSTEPS[s]
    if not param:
        return None
    step = parameters[s]
    return step[param]


def _parameterize_substeps_0(s, in_config):
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

    assert ss in parameters, (f"substep {ss} is not a substep of step {s}."
                              f" Provide one of {pformat(list(parameters.keys()))}")

    return (s, ss)


def check_mode(argued):
    """What mode should the pipeline action be performed in?
    """
    if argued.mode and argued.mode not in ("test", "develop", "prod"):
        raise ValueError("Illegal action %s mode %s", argued.action, argued.mode)
    return argued.mode


def get_current(action, mode, config, step, substep, controls=None, with_parallelization=None):
    """..."""
    current_run = workspace.initialize if is_to_init(action) else workspace.current
    return current_run(config, step, substep, controls, mode, with_parallelization)


def main(argued=None):
    """..."""
    if not argued:
        parser = get_parser()
        argued = parser.parse_args()

    LOG.info("Initialize the topological analysis pipeline: \n%s", pformat(argued))

    at_path = Path(argued.configure)
    c = pipeline.TopologicalAnalysis.read_config(at_path)
    a = argued
    c["paths"]["output"] = _read_output_in_config(c, and_argued_to_be=a.output)

    p = pipeline.TopologicalAnalysis.read_parallelization(argued.parallelize)
    s, ss = check_step(argued, against_config=c)
    m = check_mode(argued)
    current_run = get_current(action=a.action, mode=m, config=c,
                              step=s, substep=ss, controls=a.controls,
                              with_parallelization=p)
    LOG.info("Workspace initialized at %s", current_run)

    if is_to_init(argued.action):
        return current_run

    w = current_run
    topaz = pipeline.TopologicalAnalysis(config=c, parallelize=p, mode="run", workspace=w)

    LOG.info("Initialized a run of the TAP pipelein configration %s parallelization %s",
             argued.configure, argued.parallelize)

    LOG.info("Run the pipeline.")

    if not s:
        raise ValueError("Provide a step to run: ")

    a = argued.action; b = argued.batch; c = argued.controls
    p = argued.sample; o = argued.output; t = argued.test
    result = topaz.run(step=s, substep=ss, action=a, in_mode=m, controls=c, batch=b,
                       sample=p, output=o, dry_run=t)


    LOG.info("DONE running pipeline")

    return result


def get_parser():
    """Get a parser for the CLI arguments.
    """
    parser = ArgumentParser(description=read_description(),
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument("action",
                        help=("A pipeline action to run. Following TAP actions have been defined.\n"
                              "\t(1) init: to setup and intiialize.\n"
                              "\t(2) run: to run..., initializing if not already done\n"
                              "\t(3) resume: resume from the current state\n"
                              "\t(4) collect: collect the results into a single store.\n"
                              "The action may be expected to apply to all the pipeline steps unless\n"
                              "otherwise indicated."))


    parser.add_argument("step", nargs='?', default=None,
                        help=("Pipeline step to act on. "
                              "Only one step may be run at a time, which is parameterized by `step`.\n"
                              "Skip this argument when initializing a pipeline run:\n"
                              "python tap --configure=config.json --parallelize=parallel.json init\n"
                              "which will setup a workspace to run the entire pipeline."))

    parser.add_argument("substep", nargs='?', default=None,
                        help=("Some of the pipeline steps can be duvided into "
                              "individual indepedent sub-steps.\n"
                              "The simplest case is when a bunch of analyses must be run, "
                              "each for each subtarget independently of the others.\n"
                              "Provide a pipeline step's sub-step to run only that one.\n"
                              "For example,\n"
                              "python tap <action> analyze-connectivity simplices\n"
                              "to run an action that analyzes simplices"))

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

    parser.add_argument("-m", "--mode", required=False, default=None,
                        help=("Specify how the action should be performed. should be done\n"
                              "For example:\n"
                              "tap --configure=config.json --parallelize=parallel.json --mode=prod run\n"
                              "to run in production mode."))

    parser.add_argument("-b", "--batch", required=False,
                        help=("Path to a dir that contains a `batches.h5` file that maps subtargets to int\n"
                              "The dataframe should contain the input to run the pipeline stage computation\n"
                              "For example, analyses can be run on a selection of subtarget adjacency matrices\n"
                              "that can be saved in this data for a later run.\n"
                              "This option will be useful to programmatically produce shell scripts to launch\n"
                              "multiple single-node computations.\n"
                              "This will allow use to configure a multi-node parallel computation for analyses\n"
                              "that require more than a single compute node."))

    parser.add_argument("-x", "--controls", required=False, default=None,
                        help=("Apply a random control to a subvolume, and compute an analysis.\n"
                              "The random control itself should be configured in the config loaded from the \n"
                              "reference provided as the argument, for example  `--configure=<config.json>`.\n"
                              "The random controls should be specified in the config for each analyses,\n"
                              "and a future version will be able to produce an execution script / config\n"
                              "from this information. For now it makes goods practice for the purpose of\n"
                              "documentaiton."))

    parser.add_argument("--output",
                        help="Path to the directory to output in.", default=None)

    parser.add_argument("--sample",
                        help="A float to sample subtargets with", default=None)

    parser.add_argument("--dry-run", dest="test",  action="store_true",
                        help=("Use this to test the pipeline's plumbing before running any juices through it."))

    parser.set_defaults(test=False)

    return parser


def read_description():
    """..."""
    path = Path.cwd() / "project.org"
    description = ""
    if not path.exists():
        LOG.info("No description at %s", path)
        description += "Topological analysis of circuit subvolumes."

    else:
        LOG.info("Read description from %s", path)
        with open(path, 'r') as f:
            description += "\n\n" + f.read()

    return description


if __name__ == "__main__":

    LOG.warning("Analyze circuit subtarget topology.")

    parser = get_parser()

    args = parser.parse_args()

    LOG.warning(str(args))
    main(args)
