"""A single step in a pipeline."""
from types import ModuleType
from importlib import import_module
from pathlib import Path

from ..plugins import load_module_from_path as load_module
from ..io import logging
from .runnable import Runnable

LOG = logging.get_logger("pipeline step.")


def prepare_runner(obj):
    """Resolve object `obj` to be able to run a pipeline step.
    TODO: Refactor, clean up, and add some inspection expected of the run method.
    """
    if callable(obj):
        runner = obj
    elif isinstance(obj, ModuleType):
        runner = obj
    else:
        try:
            with_name = str(obj)
            module = import_module(with_name)
        except ModuleNotFoundError:
            try:
                from_path = Path(obj)
                module = load_module(from_path)
                assert module
            except TypeError:
                runner = None
            else:
                runner = module
        else:
            runner = module

        if not runner:
            raise ValueError(f"Couldn't find a runner in obj={obj}")

        try:
            run = module.run
        except AttributeError:
            raise TypeError("To define a step, a module needs to be runnable."
                            f"Found {module} with no `run` method.")
        else:
            #TODO: inspect run
            pass

        runner = module

    return runner


class Step(Runnable):
    """An individual step in the pipeline,
    should implement a run method.
    """
    def __init__(self, obj):
        """A pipeline step."""
        self._runner = prepare_runner(obj)

    def _setup_dev_version(self, computation, in_config, using_runtime, **kwargs):
        """..."""
        from connsense.develop.parallelization import setup_multinode, setup_compute_node
        LOG.warning("Run %s using code under development: %s", computation, setup_multinode)
        return setup_multinode(setup_compute_node, computation, in_config, using_runtime, in_mode="develop")

    def setup(self, computation, in_config, using_runtime=None, **kwargs):
        """..."""
        if using_runtime:
            from connsense.pipeline.parallelization.parallelization import run_multinode, setup_compute_node
            return run_multinode(setup_compute_node, computation, in_config, using_runtime)

        _, substep = computation.split('/')
        return self._runner.setup(in_config, substep, **kwargs)

    def collect(self, computation, in_config, using_runtime, in_mode, **kwargs):
        """Allow collection of results produced by parallel runs.
        To collect results both the scientific config, and the parallelization config
        will be required.

        Expect attribute errors if this instance's runner  does not implement
        a collect method.
        """
        if in_mode == "develop":
            from connsense.develop.parallelization import setup_multinode, collect_results
            return setup_multinode(collect_results, computation, in_config, using_runtime, **kwargs)

        from .parallelization.parallelization import run_multinode, collect_multinode
        return run_multinode(collect_multinode, computation, in_config, using_runtime, **kwargs)

    def _run_dev_version(self, computation, in_config, using_runtime, on_compute_node, slicing, **kwargs):
        """..."""
        from ..develop.parallelization import run_multiprocess
        LOG.warning("Run %s using code under development: %s", computation, run_multiprocess)
        return run_multiprocess(computation, in_config, using_runtime, on_compute_node, slicing=slicing)

    def run(self, computation, in_config, using_runtime, on_compute_node, inputs, **kwargs):
        """..."""
        from .parallelization.parallelization import run_multiprocess
        return run_multiprocess(computation, in_config, using_runtime, on_compute_node, inputs)

    def check_state(self, pipeline):
        """TODO: Check where a pipeline is along the sequence of steps that define it."""
        return True
