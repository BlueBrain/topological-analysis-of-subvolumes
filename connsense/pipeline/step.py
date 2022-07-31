"""A single step in a pipeline."""
from types import ModuleType
from importlib import import_module
from pathlib import Path

from ..plugins import load_module_from_path as load_module
from .runnable import Runnable


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

    def setup(self, config, **kwargs):
        """..."""
        return self._runner.setup(config, **kwargs)

    def collect(self, config, **kwargs):
        """Allow collection of results produced by parallel runs.
        To collect results both the scientific config, and the parallelization config
        will be required.

        Expect attribute errors if this instance's runner  does not implement
        a collect method.
        """
        return self._runner.collect(config, **kwargs)

    def run(computation, in_config, using_runtime, on_compute_node, inputs):
        """..."""
        from .parallelization import run_multiprocess
        return run_multprocess(computation, in_config, using_runtime, on_compute_node, inputs)

    def check_state(self, pipeline):
        """TODO: Check where a pipeline is along the sequence of steps that define it."""
        return True
