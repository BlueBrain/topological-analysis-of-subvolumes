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


def get_rundir(config, step=None, substep=None, mode=None,
               with_base=False):
    """..
    ."""
    assert not mode or mode in ("test", "develop", "prod"), str(mode)

    pipeline = config["paths"]

    basedir = Path(pipeline["root"])
    assert basedir.exists() and basedir.is_dir(),\
        f"Pipeline root folder {basedir} must exist."

    rundir = basedir / "run"
    rundir.mkdir(parents=False, exist_ok=True)

    modir = rundir / mode if mode else rundir
    modir.mkdir(parents=False, exist_ok=True)

    if not step:
        assert not substep, f"Substep {f} of step None maketh sense None"
        return (rundir, modir) if with_base else modir

    stepdir = modir / step
    stepdir.mkdir(parents=False, exist_ok=True)

    if not substep or substep == "_":
        return (rundir, stepdir) if with_base else stepdir

    substepdir = stepdir / substep
    substepdir.mkdir(parents=False, exist_ok=True)

    return (rundir, substepdir) if with_base else substepdir


def check_configs(c, and_to_parallelize, at_location, must_exist=False, create=False):
    """Check if a config file exists at a location, and create it if it does not.
    Either a config must exist at a location, or it must not.
    """
    j = at_location / "config.json"
    p = (at_location / "parallelize.json" if and_to_parallelize else None)

    if must_exist:
        if not j.exists():
            raise FileNotFoundError(f"Location {at_location} must have a config but does not."
                                    "\n\tInitialize the parent folders first.")
        if p and not p.exists():
            raise FileNotFoundError(f"Location {at_location} must have a parallization config"
                                    " but does not.\n"
                                    "You may need to initialize the pipeline workspace.")

        return (j, p)

    l = at_location
    if j.exists() and p and p.exists():
        raise FileExistsError(f"Location {l} seems to already have run configs")

    check_config = (read_config.write(c, to_json=j)
                    if not j.exists() and create else True)
    check_parallel = (read_config.write(and_to_parallelize, to_json=p)
                      if p and not p.exists() and create else True)
    return (check_config, check_parallel)


def timestamp(dir):
    """..."""
    today, now = time.stamp(now=True, format=lambda x, y: (x, y))

    day = dir / today
    day.mkdir(parents=False, exist_ok=True)

    at_time = day / now
    at_time.mkdir(parents=False, exist_ok=False)
    return at_time


def initialize(config, step=None, substep=None, mode=None, parallelize=None):
    """Set up a run of the pipeline.
    """
    c = config; s = step; ss = substep; p = parallelize
    LOG.info("Initialize workspace for config \n %s", pformat(c))
    if p:
        LOG.info("with parallelization \n %s", pformat(p))
    else:
        LOG.info("witout parallelization.")

    run, stage = get_rundir(c, s, ss, mode, with_base=True)

    if not s:
        assert not ss, f"Substep {ss} of step None maketh sense None"

    def _check_configs_must_exist(x, and_to_create):
        check_configs(c, and_to_parallelize=p, at_location=run,
                      must_exist=x, create=and_to_create)

    if_just_base = not mode and not step

    check_configs(c, and_to_parallelize=p, at_location=run,
                  must_exist=not if_just_base, create=if_just_base)

    return stage


def cleanup(config, **kwargs):
    """Clean up the workspace.
    """
    raise NotImplementedError


def current(config, step, substep, mode, to_parallelize):
    """..."""
    run = get_rundir(config, step, substep, mode)
    cwd = run / "current"

    if not cwd.exists():
        cwd = initialize(config, step, substep, mode, to_parallelize)

    return cwd


def locate_base(in_rundir, for_step):
    """..."""
    base = Path(in_rundir) / for_step
    if not base.exists():
        raise FileNotFoundError(f"A workspace folder {base} must be created\n"
                                "Use `tap --config=<location> init <step> <substep>` to initialize"
                                "a folder to run a substep of a pipeline step.\n"
                                "For example \n"
                                "`tap --config=<location> init analyze-connectivity simplices`\n"
                                "will create a directory to compute simplices in.")
    return base
