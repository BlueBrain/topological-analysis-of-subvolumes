from collections.abc import Mapping, Iterable
import os
from pathlib import Path
import json
import yaml


def adjust_root(in_a_dict_paths):
    """..."""
    if not in_a_dict_paths:
        return {}

    try:
        root = in_a_dict_paths["root"]
    except KeyError:
        root = None
    else:
        root = Path(root)

    def absolute(filepath):
        if os.path.isabs(filepath):
            return filepath
        return (root / filepath).as_posix()

    try:
        files_dict = in_a_dict_paths["files"]
    except KeyError:
        return {}
    else:
        files = files_dict.items()

    def nest(filename):
        """..."""
        try:
            subfiles = filename.items
        except AttributeError:
            return absolute(filename)
        return {label: absolute(name) for label, name in subfiles}

    return {label: nest(specified) for label, specified in files}


def locate_relpaths(cfg):
    """Convert relative paths to absolute paths.
    """

    path_circuit = adjust_root(cfg["paths"]["circuit"])
    path_flatmap = adjust_root(cfg["paths"].get("flatmap", None))
    paths = {"format": "absolute", "circuit": path_circuit, "flatmap": path_flatmap}

    def append_groups(to_hdf):
        """..."""
        steps = {step: (to_hdf, group) for step, group in pipeline["steps"].items()}
        return steps

    pipeline = cfg["paths"]["pipeline"]
    basedir = Path(pipeline["root"])

    input_hdf = basedir / pipeline["input"]["store"]
    input_steps = append_groups(input_hdf)

    output_hdf = basedir/pipeline["output"]["store"] if "output" in pipeline else None
    output_steps = append_groups(output_hdf) if output_hdf else input_steps

    pipeline_paths = {"root": pipeline["root"],
                      "input": {"store": input_hdf, "steps": input_steps},
                      "output": {"store": output_hdf, "steps": output_steps}}
    paths.update(pipeline_paths)

    parameters = cfg["parameters"]

    return {"steps": pipeline["steps"], "paths": paths, "parameters": parameters}


def read(fn, raw=False):
    """Read JSON format config, converting a relative path specification to absolute.
    """
    try:
        path = Path(fn)
    except TypeError as terror:
        raise TypeError(f"Unexpected type of config file reference{type(fn)}.") from terror

    def read_raw():
        if path.suffix.lower() in (".yaml", "yml"):
            with open(path, "r") as fid:
                return yaml.load(fid, Loader=yaml.FullLoader)
        if path.suffix.lower() == ".json":
            with open(path, "r") as fid:
                return json.load(fid)

    cfg = read_raw()

    assert "paths" in cfg,\
        "Configuration file must specify 'paths' to input/output files!"

    assert "parameters" in cfg,\
        "Configuration file must specify 'parameters' for pipeline steps!"

    if raw:
        return cfg

    format_paths = cfg["paths"].get("format", "relative")
    if format_paths == "absolute":
        return cfg

    return locate_relpaths(cfg)

def serialize_json(paths):
    """..."""
    if isinstance(paths, Mapping):
        return {serialize_json(key): serialize_json(val) for key, val in paths.items()}

    if isinstance(paths, str):
        return paths

    if isinstance(paths, Path):
        return paths.as_posix()

    if isinstance(paths, Iterable):
        return [serialize_json(p) for p in paths]

    if isinstance(paths, float):
        return str(paths)

    return paths


def write(config, to_json, and_yaml=None):
    """..."""
    with open(to_json, 'w') as to_file:
        to_file.write(json.dumps(serialize_json(config)))

    if and_yaml:
        with open(and_yaml, 'w') as to_file:
            yaml.dump({key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
                      to_file, allow_unicode=True)
    return to_json


def check_paths(in_config, step):
    """
    in_config :: A Mapping read from a JSON config file, by the `read` method above.
    """
    p = in_config["paths"]
    input = p["input"]; output = p["output"]

    if "circuit" not in p:
        raise RuntimeError("No circuits defined in config!")

    if "define-subtargets" not in input["steps"]:
        raise RuntimeError("No method to define subtargets.")

    if "extract-node-populations" not in input["steps"]:
        raise RuntimeError("No method to extract nodes.!")

    if "extract-edge-populations" not in input["steps"]:
        raise RuntimeError("No connection matrices in config!")

    if step not in output["steps"]:
        raise RuntimeError(f"No {step} in config output!")

    return (input, output)
