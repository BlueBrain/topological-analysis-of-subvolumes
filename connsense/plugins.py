"""General utilities."""

from types import ModuleType
from collections.abc import Mapping
import importlib
from pprint import pformat
from pathlib import Path

from connsense.io import logging

LOG = logging.get_logger("Write Results")


class ImportFailure(ModuleNotFoundError):
    pass


def import_module(from_path, with_method=None):
    """..."""
    LOG.info("Import module from path %s, with method %s", from_path, with_method)

    if isinstance(from_path, Mapping):
        try:
            source = from_path["source"]
        except KeyError as kerr:
            LOG.warning("No source in path: \n%s", pformat(from_path))
            raise ImportFailure from kerr
        try:
            method = from_path["method"]
        except KeyError as kerr:
            LOG.warning("No method in path: \n%s", pformat(from_path))
            raise ImportFailure from kerr
        return import_module(source, method)

    if not from_path:
        assert not with_method, "Cannot find a method without a path."
        LOG.warning("Cannot import a module from a null path")
        return (None, None)
    try:
        module = import_module_with_name(from_path)
    except ModuleNotFoundError:
        path = Path(from_path)

        assert path.exists(), f"No such path {path}"

        assert path.suffix == ".py", f"Not a python file {path}!"

        spec = importlib.util.spec_from_file_location(path.stem, path)

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

    if with_method:
        if not hasattr(module, with_method):
            LOG.warning("No method %s among module: \n%s", with_method, pformat(dir(module)))
            raise TypeError(f"No method to {with_method}")
        return (module, getattr(module, with_method))

    return module


def import_module_with_name(n):
    """Must be in the environment."""
    if isinstance(n, ModuleType): return n
    assert isinstance(n, str)
    return importlib.import_module(n)


def load_module_from_path(p):
    """Load a module from a path.
    """
    path = Path(p)

    if not path.exists:
        raise FileNotFoundError(p.as_posix())

    if  path.suffix != ".py":
        raise ValueError(f"Not a python file {path}!")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) #

    return module


def get_module(from_object, with_function=None):
    """Get module from an object.
    Read the code to see what object can be resolved to a module.
    If `with_method`, look for the method in the module.
    """
    def iterate(functions):
        """..."""
        if isinstance(functions, str):
            return [functions]
        try:
            items = iter(functions)
        except TypeError:
            items = [functions]

        return items

    def check(module, has_functions=None):
        """..."""
        if not has_functions:
            return module

        def get_method(function):
            """..."""
            try:
                method = getattr(module, function)
            except AttributeError:
                raise TypeError(f" {module} is missing required method {function}.")
            return method

        if isinstance(has_functions, str):
            methods = get_method(has_functions)

        methods = {f: get_method(f) for f in iterate(has_functions)}

        return (module, methods)

    try:
        module = import_module_with_name(str(from_object))
    except ModuleNotFoundError:
        module = load_module_from_path(p=from_object)
        if not module:
            raise ModuleNotFoundError(f"that was specified by {from_object}")

    return check(module)


def collect_plugins_of_type(T, in_config):
    """..."""
    return {T(name, description) for name, description in items()}
