#!/usr/bin/env python3

"""
Setup a run of the Topological Analysis (of subvolumes) Pipeline (TAP)
"""
from  pathlib import Path
from pprint import pformat
import shutil

import pandas as pd
import numpy as np

from ..io import time
from ..io import logging, read_config

STEP = "setup-pipeline"
LOG = logging.get_logger(STEP)


def is_analysis(step):
    """How is each analyses step configured?"""
    return step.startswith("analyze-")


def get_rundir(config, step=None, substep=None, subgraphs=None, controls=None, mode=None, with_base=False,
               **kwargs):
    """..."""
    def apply_controls(rundir):
        """Check of a control method has been argued, and if so
        create a specific folder.

        NOTES:
        20221008: Controls will be run as part of the complete computation of original + controls + subsets
        together. Thus there will be no need to create a sub-workspace for controls or subgraphs.
        However we leave the code here. The controlling and subgraphing procedures should not interfere with the
        main computation's setup.
        """
        if not controls:
            return rundir

        try:
            base, subdir = rundir
        except TypeError:
            controlled = rundir / "controls"
            controlled.mkdir(exist_ok=True, parents=False)
            argued = controlled / controls
            argued.mkdir(exist_ok=True, parents=False)
            return argued

        controlled = subdir / "controls"
        controlled.mkdir(exist_ok=True, parents=False)
        argued = controlled / controls
        argued.mkdir(exist_ok=True, parents=False)
        return (base, argued)

    assert not mode or mode in ("test", "develop", "prod"), str(mode)

    pipeline = config["paths"]

    basedir = Path(pipeline["root"])
    assert basedir.exists() and basedir.is_dir(), f"Pipeline root folder {basedir} must exist."

    rundir = basedir / "run"
    rundir.mkdir(parents=False, exist_ok=True)

    modir = rundir / mode if mode else rundir
    modir.mkdir(parents=False, exist_ok=True)

    def setup(subdir, at_dirpath):
        """..."""
        subdirpath = at_dirpath / subdir
        subdirpath.mkdir(parents=False, exist_ok=True)
        return subdirpath

    if step:
        stepdir = setup(step, at_dirpath=modir)
        if substep and substep != '_':
            substep_at = substep.split('/')
            parent = stepdir
            for substep in substep_at:
                substepdir = setup(substep, at_dirpath=parent)
                parent = substepdir
            if subgraphs:
                subgraphsdir = setup(subgraphs, at_dirpath=parent)
                result = apply_controls((rundir, subgraphsdir) if with_base else subgraphsdir)
            else:
                result = apply_controls((rundir, parent) if with_base else parent)
        else:
            result = apply_controls((rundir, stepdir) if with_base else modir)
    else:
        assert not substep, f"Substep {substep} of step None maketh sense None"
        result = apply_controls((rundir, modir) if with_base else modir)

    LOG.info("get rundir for mode %s, step %s, substep %s, subgraphs %s, controls %s: %s",
             mode, step, substep, subgraphs, controls, result)
    return result


def check_configs(c, and_to_parallelize, at_location, mode=None, must_exist=False, create=False, strict=False):
    """Check if a config file exists at a location, and create it if it does not.
    Either a config must exist at a location, or it must not.
    """
    p = and_to_parallelize
    check_location = at_location if not mode else at_location / mode

    pc = {"json": check_location / "config.json", "yaml": check_location / "pipeline.yaml"}
    pc_exists = pc["json"].exists() or pc["yaml"].exists()

    pp = ({"json": check_location / "parallel.json", "yaml": check_location / "runtime.yaml"}
          if and_to_parallelize else None)
    pp_exists = pp and (pp["json"].exists() or pp["yaml"].exists())

    if strict and must_exist:
        if not pc_exists:
            raise FileNotFoundError(f"Location {at_location} must have a config but does not."
                                    "\n\tInitialize the parent folders first. Start at the base.\n"
                                    "`tap --config=<JSON> --parallelize=<JSON>  init`\n"
                                    "before setting up run modes or steps and substeps.")
        if pp and not pp_exists:
            raise FileNotFoundError(f"Location {at_location} must have a parallization config"
                                    " but does not.\n"
                                    "You may need to initialize the pipeline workspace.\n"
                                    "\n\tInitialize the parent folders first. Start at the base.\n"
                                    "`tap --config=<JSON> --parallelize=<JSON>  init`\n"
                                    "before setting up run modes or steps and substeps.")

        return (pc, pp)

    l = at_location
    if strict and pc.exists() and pp and pp.exists():
        raise FileExistsError(f"Location {l} seems to already have run configs")

    if not create:
        return (pc_exists, pp_exists)

    check_config = pc_exists  or read_config.write(c, to_json=pc["json"], and_yaml=pc["yaml"])
    check_parall = (pp_exists or read_config.write(p, to_json=pp["json"], and_yaml=pp["yaml"])
                    if pp else None)
    return (check_config, check_parall)


def timestamp(dir):
    """..."""
    today, now = time.stamp(now=True, format=lambda x, y: (x, y))

    day = dir / today
    day.mkdir(parents=False, exist_ok=True)

    at_time = day / now
    at_time.mkdir(parents=False, exist_ok=False)
    return at_time


def symlink(configs, at_dirpath, to_base_configs):
    """..."""
    pipeline_exists, runtime_exists = configs
    base_pipeline , base_runtime = to_base_configs

    return ((at_dirpath / base_pipeline.name).symlink_to(base_pipeline) if not pipeline_exists else None,
            (at_dirpath / base_runtime.name).symlink_to(base_runtime) if base_runtime and not runtime_exists else None)


def initialize(config, step=None, substep=None, subgraphs=None, controls=None,
               mode=None, parallelize=None, strict=False):
    """Set up a run of the pipeline.
    """
    try:
        c, path_config = config
    except TypeError:
        c = config
        path_config  = None
    try:
        p, path_parallelize = parallelize
    except TypeError:
        p = parallelize
        path_parallelize = None

    s = step; ss = substep; m = mode;
    LOG.info("Initialize workspace for config with keys: \n %s", pformat(list(c.keys())))

    if p:
        LOG.info("with parallelization keys \n %s", pformat(list(p.keys())))
    else:
        LOG.info("witout parallelization.")

    to_run, stage = get_rundir(c, s, ss, subgraphs, controls, mode, with_base=True)

    if not s:
        assert not ss, f"Substep {ss} of step None maketh sense None"

    def _check_configs_must_exist(x, and_to_create):
        check_configs(c, and_to_parallelize=p, at_location=to_run, must_exist=x, create=and_to_create)

    if_just_base = not mode and not step
    #check_configs(c, p, at_location=to_run, must_exist=not if_just_base, create=if_just_base)
    base_configs = check_configs(c, p, at_location=to_run, must_exist=not if_just_base, create=False)
    symlink(base_configs, at_dirpath=to_run, to_base_configs=(path_config, path_parallelize))

    if mode:
        rundir_configs = check_configs(c, p, at_location=to_run, mode=m, must_exist=not if_just_base, create=False)
        symlink(rundir_configs, at_dirpath=to_run/mode, to_base_configs=(path_config, path_parallelize))

    if mode or step:
        stage_configs = check_configs(c, p, at_location=stage, must_exist=True, create=False)
        symlink(stage_configs, at_dirpath=stage, to_base_configs=(path_config, path_parallelize))

    return stage


def cleanup(config, **kwargs):
    """Clean up the workspace.
    """
    raise NotImplementedError


def current(config, step, substep, subgraphs, controls, mode, to_parallelize):
    """..."""
    try:
        c, path_config  = config
    except TypeError:
        c = config

    run, stage = get_rundir(c, step, substep, subgraphs, controls, mode, with_base=True)

    if not stage.exists():
        stage = initialize(config, step, substep, subgraphs, controls, mode, to_parallelize)

    return stage


def locate_base(in_rundir, for_step, create=False):
    """..."""
    base = Path(in_rundir) / for_step

    if create:
        base.mkdir(parents=False, exist_ok=True)

    if not base.exists():
        raise FileNotFoundError(f"A workspace folder {base} must be created\n"
                                "Use `tap --config=<location> init <step> <substep>` to initialize"
                                "a folder to run a substep of a pipeline step.\n"
                                "For example \n"
                                "`tap --config=<location> init analyze-connectivity simplices`\n"
                                "will create a directory to compute simplices in.")
    return base

def find_base(rundir, max_expected_depth=6):
    """...
    Computations are staged hierarchically under a root dir.
    This method will find where the root is from a directory under it.
    """
    if rundir.name == "run":
        return rundir

    if max_expected_depth == 0:
        return None

    return find_base(rundir.parent, max_expected_depth-1)
