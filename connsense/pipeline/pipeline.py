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


LOG = logging.get_logger("pipeline.")

PipelineState = namedtuple("PipelineState", ["complete", "running", "queue"],
                           defaults=[None, None, None])


class TopologicalAnalysis:
    """..."""
    from connsense import define_subtargets
    from connsense import extract_neurons
    from connsense import extract_connectivity
    from connsense import randomize_connectivity
    from connsense import analyze_connectivity

    __steps__ = OrderedDict([("define-subtargets"      , Step(define_subtargets)),
                             ("extract-neurons"        , Step(extract_neurons)),
                             ("extract-connectivity"   , Step(extract_connectivity)),
                             ("randomize-connectivity" , Step(randomize_connectivity)),
                             ("analyze-connectivity"   , Step(analyze_connectivity))])

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
    def read_config(cls, c, raw=False):
        """..."""
        from connsense.io import read_config

        try:
            path = Path(c)
        except TypeError:
            assert isinstance(c, Mapping)
            return c

        return read_config.read(path, raw=raw)

    @classmethod
    def read_parallelization(cls, config):
        """..."""
        if not config:
            return None

        try:
            path = Path(config)
        except TypeError:
            assert isinstance(config, Mapping)
            return config

        with open(path, 'r') as fptr:
            parallelization = json.load(fptr)
        return parallelization

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

        self._config = self.read_config(c)
        self._parallelize = self.read_parallelization(p) if parallelize else None

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

    @property
    def data(self):
        """..."""
        return self._data

    def dispatch(self, step, substep, action, controls, in_mode, **kwargs):
        """..."""
        LOG.info("Pipeline dispatch %s %s %s", step, substep, action)
        result = (self.__steps__[step]
                  .run(self._config, action=action, substep=substep, in_mode=in_mode,
                       parallelize=self._parallelize, tap=self.data, controls=controls,
                       **kwargs))

        return result

    def get_h5group(self, step):
        """..."""
        return self._data_groups.get(step)

    def collect(self, step, substep, in_mode, controls, **kwargs):
        """Collect the batched results generated in a single step.
        """
        runner = self.__steps__[step]
        try:
            gather = runner.collect
        except AttributeError as aerror:
            raise NotImplementedError(f"A method to collect in {step}") from aerror
        else:
            LOG.info("Use to gather results: %s", gather)

        return gather(self._config, in_mode=in_mode, parallelize=self._parallelize, substep=substep,
                      controls=controls, **kwargs)

    # def run_queue(self, steps=None, substeps=None, action=None, in_mode=None,
    #               *args, **kwargs):
    #     """Run the pipeline.
    #     This used to be the `run` method for analyze-connectivity ---
    #     This method will not work --- we will come back when we are ready to
    #     run a pipeline queue --- for now use the `run` method
    #     """
    #     if self._mode == "inspect":
    #         if not in_mode == "inspect":
    #             raise RuntimeError("Cannot run a read-only pipeline."
    #                                " You can use read-only mode to inspect the data"
    #                                " that has already been computed.")
    #         if action and action.lower() != "inspect":
    #             raise RuntimeError(f"Cannot run {action} for a pipeline in mode inspect\n"
    #                                "In mode inspect, there is no action to do,\n"
    #                                " so use action=None or action='inspect'")

    #     if action.lower() in ("collect", "merge"):
    #         return self.collect(steps, in_mode, *args, **kwargs)

    #     if steps:
    #         s = self.state
    #         self.state = PipelineState(complete=s.complete, running=s.running, queue=steps)
    #     LOG.warning("Dispatch from %s queue: %s", len(self.state.queue), self.state.queue)

    #     while self.state.queue:
    #         step = self.state.queue.pop(0)

    #         LOG.warning("Dispatch pipeline step %s", step)

    #         self.running = step
    #         result = self.dispatch(step, action, in_mode, *args, **kwargs)
    #         self.running = None
    #         self.state.complete[step] = result

    #         LOG.warning("DONE pipeline step %s: %s", step, result)

    #     LOG.warning("DONE running %s steps: ", len(self.state.complete))

    #     return self.state

    def run(self, step, substep=None, action=None, in_mode=None, controls=None, **kwargs):
        """Run the pipeline, one step and if defined one substep at a time.
        We can chain all the steps together later.
        """
        LOG.warning("RUN pipeline action %s for step %s %s ", action, step, substep)

        if self._mode == "inspect":
            if not in_mode == "inspect":
                raise RuntimeError("Cannot run a read-only pipeline."
                                   " You can use read-only mode to inspect the data"
                                   " that has already been computed.")
            if action and action.lower() != "inspect":
                raise RuntimeError(f"Cannot run {action} for a pipeline in mode inspect\n"
                                   "In mode inspect, there is no action to do,\n"
                                   " so use action=None or action='inspect'")

        if action.lower() in ("collect", "merge"):
            return self.collect(step, substep, in_mode, controls, **kwargs)

        result = self.dispatch(step, substep, action, controls, in_mode, **kwargs)

        LOG.warning("DONE run action %s for pipeline step %s %s", action, step, substep)
        LOG.info("RESULT %s %s: %s", step, substep, result)
        return result
