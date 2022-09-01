"""The topological analysis pipeline, starting with a configuration.
"""
from importlib import import_module
from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod
from collections.abc import Mapping
from collections import OrderedDict, namedtuple
from pathlib import Path
from pprint import pformat
from lazy import lazy
import json

from ..io.write_results import (read_toc_plus_payload, read_node_properties,
                                read_subtargets)
from .step import Step
from .store import HDFStore
from ..io import logging
from .import ConfigurationError, NotConfiguredError, PARAMKEY, COMPKEYS, workspace


LOG = logging.get_logger("pipeline.")

PipelineState = namedtuple("PipelineState", ["complete", "running", "queue"],
                           defaults=[None, None, None])


class TopologicalAnalysis:
    """..."""
    from connsense import (define_subtargets,
                           extract_nodes,
                           extract_node_types,
                           evaluate_subtargets,
                           extract_connectivity,
                           sample_connectivity,
                           randomize_connectivity,
                           analyze_node_types,
                           analyze_composition,
                           analyze_connectivity,
                           analyze_physiology)

    __steps__ = OrderedDict([("define-subtargets", Step(define_subtargets)),
                             ("extract-node-types", Step(extract_node_types)),
                             ("extract-node-populations", Step(extract_nodes)),
                             ("evaluate-subtargets", Step(evaluate_subtargets)),
                             ("extract-edge-populations", Step(extract_connectivity)),
                             ("sample-edge-populations", Step(sample_connectivity)),
                             ("randomize-connectivity", Step(randomize_connectivity)),
                             ("analyze-node-types", Step(analyze_node_types)),
                             ("analyze-composition", Step(analyze_composition)),
                             ("analyze-connectivity", Step(analyze_connectivity)),
                             ("analyze-physiology", Step(analyze_physiology))])

    def decompose(self, computation_type, of_quantity):
        """Decompose a computation into it's components."""
        from connsense.pipeline.parallelization import parallelization as prl
        parameters = prl.parameterize(computation_type, of_quantity, self._config)

        return {var: val for var, val in parameters.items() if var not in COMPKEYS}

    @classmethod
    def sequence_of_steps(cls):
        """..."""
        return cls.__steps__.items()

    @classmethod
    def subset(complete, configured_steps):
        """configured : list of steps."""

        if configured_steps is None:
            return complete.sequence_of_steps()

        return OrderedDict([(step, to_take) for step, to_take in complete.sequence_of_steps
                            if step in configured_steps])

    @classmethod
    def read_config(cls, c, raw=False, return_location=True):
        """..."""
        from connsense.io import read_config

        try:
            path = Path(c)
        except TypeError:
            assert isinstance(c, Mapping)
            return (None, c) if return_location else c

        config = read_config.read(path, raw=raw)
        return (path, config) if return_location else config

    @classmethod
    def read_parallelization_0(cls, config):
        """..."""
        if not config:
            return None

        try:
            path = Path(config)
        except TypeError:
            assert isinstance(config, Mapping)
            return config

        if path.suffix.lower() in (".yaml", "yml"):
            with open(path, "r") as fid:
                return yaml.load(fid, Loader=yaml.FullLoader)

        if path.suffix.lower() == ".json":
            with open(path, "r") as fid:
                return json.load(fid)
        raise ValueError(f"Unknown file format {path.suffix}")

    @classmethod
    def read_parallelization(cls, config, of_pipeline=None, return_path=False):
        from .parallelization.parallelization import read_runtime_config as read_runtime
        return read_runtime(config, of_pipeline, return_path)

    @classmethod
    def read_steps(cls, config):
        """config : Mapping<key: value>."""
        try:
            configured = config["steps"]
        except KeyError:
            configured = list(cls.__steps__.keys())
        return configured

    def __init__(self, config, parallelize=None, mode="inspect", workspace=None):
        """Read the pipeline steps to run from the config.
        """
        assert mode in ("inspect", "run"), mode

        c = config
        p = parallelize

        self._path_config, self._config = self.read_config(c, return_location=True)

        self._path_parallelize, self._parallelize = (
            self.read_parallelization(p, of_pipeline=self._config, return_path=True) if parallelize
            else (None, None))

        self._data = HDFStore(self._config)

        self._workspace = workspace
        self._mode = mode

        self.configured_steps =  self.read_steps(self._config)
        self.running = None
        self.state = PipelineState(complete=OrderedDict(),
                                   running=None,
                                   queue=self.configured_steps)

    @property
    def workspace(self):
        """..."""
        return self._workspace

    def set_workspace(self, to_dirpath):
        """..."""
        self._workspace = to_dirpath
        return self

    @property
    def data(self):
        """..."""
        return self._data

    def get_h5group(self, step):
        """..."""
        return self._data_groups.get(step)

    def initialize(self, step=None, substep=None, subgraphs=None, controls=None, mode=None):
        """..."""
        current = workspace.initialize((self._config, self._path_config), step, substep, subgraphs, controls, mode,
                                        (self._parallelize, self._path_parallelize))
        LOG.info("Workspace initialized at %s", current)

        if step == "index":
            self._data.create_index()
        return current

    def setup(self, step, substep=None, subgraphs=None, controls=None, **kwargs):
        """Setup the pipeline, one step and if defined one substep at a time.
        We can chain all the steps together later.
        """
        LOG.warning("SETUP pipeline action for step %s %s ", step, substep)

        in_mode = kwargs.get("in_mode", None)

        if self._mode == "inspect":
            if not in_mode == "inspect":
                raise RuntimeError("Cannot run a read-only pipeline."
                                   " You can use read-only mode to inspect the data"
                                   " that has already been computed.")
            if action and action.lower() != "inspect":
                raise RuntimeError(f"Cannot run {action} for a pipeline in mode inspect\n"
                                   "In mode inspect, there is no action to do,\n"
                                   " so use action=None or action='inspect'")

        result = self.__steps__[step].setup(self._config, substep=substep, subgraphs=subgraphs, controls=controls,
                                            parallelize=self._parallelize, tap=self.data, **kwargs)

        LOG.warning("DONE setup for pipeline step %s %s", step, substep)
        LOG.info("RESULT %s %s: %s", step, substep, result)
        return result


    def run(self, step, substep=None, in_mode=None, subgraphs=None, controls=None, inputs=None, **kwargs):
        """Run the pipeline, one (computation_type, of_quantity) at a time.
        """
        LOG.warning("RUN pipeline for step %s %s ", step, substep)

        return self.__steps__[step].run(computation='/'.join([step, substep] if substep else [step]),
                                        in_config=self._config, using_runtime=self._parallelize,
                                        on_compute_node=inputs.parent, inputs=inputs)


    def collect(self, step, substep=None, in_mode=None, subgraphs=None, controls=None):
        """Collect the batched results generated in a single step.
        """
        computation = '/'.join([step, substep] if substep else [step])
        LOG.info("Gather computation %s %s --> %s", step, substep, computation)
        runner = self.__steps__[step]
        try:
            gather = runner.collect
        except AttributeError as aerror:
            raise NotImplementedError(f"A method to collect in {step}") from aerror
        else:
            LOG.info("Use to gather results: %s", gather)

        return gather(computation, in_config=self._config, using_runtime=self._parallelize,
                      for_control=controls, making_subgraphs=subgraphs)
