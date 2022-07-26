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


PARAMKEY = {"define-subtargets": "definitions",
            "extract-voxels": "annotations",
            "extract-node-types": "models",
            "extract-node-populations": "populations",
            "extract-edge-types": "models",
            "extract-edge-populations": "populations",
            "randomize-connectivity": "algorithms",
            "analyze-geometry": "analyses",
            "analyze-composition": "analyses",
            "analyze-connectivity": "analyses"}


class TopologicalAnalysis:
    """..."""
    from connsense import define_subtargets
    from connsense import extract_nodes
    from connsense import evaluate_subtargets
    from connsense import extract_connectivity
    from connsense import randomize_connectivity
    from connsense import analyze_connectivity

    __steps__ = OrderedDict([("define-subtargets", Step(define_subtargets)),
                             ("extract-node-populations", Step(extract_nodes)),
                             ("evaluate-subtargets", Step(evaluate_subtargets)),
                             ("extract-edge-populations", Step(extract_connectivity)),
                             ("randomize_connectivity", Step(randomize_connectivity)),
                             ("analyze_connectivity", Step(analyze_connectivity))])

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
    def read_parallelization(cls, config, of_pipeline=None):
        from .parallelization import read_config as read_runtime
        return read_runtime(config, of_pipeline)

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
        from .parallelization import read_config as read_runtime
        assert mode in ("inspect", "run"), mode

        c = config
        p = parallelize

        self._config = self.read_config(c)
        self._parallelize = self.read_parallelization(p, of_pipeline=self._config) if parallelize else None

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

    def dispatch(self, step, substep, action, subgraphs, controls, in_mode, **kwargs):
        """..."""
        LOG.info("Pipeline dispatch %s %s %s", step, substep, action)
        result = (self.__steps__[step]
                  .run(self._config, action=action, substep=substep,
                       subgraphs=subgraphs, controls=controls,
                       in_mode=in_mode, parallelize=self._parallelize,
                       tap=self.data, **kwargs))

        return result

    def get_h5group(self, step):
        """..."""
        return self._data_groups.get(step)

    def collect(self, step, substep, in_mode, subgraphs, controls, **kwargs):
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
                      subgraphs=subgraphs, controls=controls, **kwargs)

    def run(self, step, substep=None, action=None, in_mode=None, subgraphs=None, controls=None,
            **kwargs):
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
            return self.collect(step, substep, in_mode, subgraphs, controls, **kwargs)

        result = self.dispatch(step, substep, action, subgraphs, controls, in_mode, **kwargs)

        LOG.warning("DONE run action %s for pipeline step %s %s", action, step, substep)
        LOG.info("RESULT %s %s: %s", step, substep, result)
        return result
