#!/usr/bin/env python3

"""
Setup a run of the Topological Analysis (of subvolumes) Pipeline (TAP)
"""
from  pathlib import Path
from pprint import pformat
import shutil

import pandas as pd
import numpy as np

from ..io.time import stamp as timestamp
from ..io import logging, read_config

STEP = "setup-pipeline"
LOG = logging.get_logger(STEP)


def get_rundir(config, step=None, substep=None):
    """..
    ."""
    pipeline = config["paths"]

    basedir = Path(pipeline["root"])
    assert basedir.exists() and basedir.is_dir(),\
        f"Pipeline root folder {basedir} must exist."

    rundir = basedir / "run"
    rundir.mkdir(parents=False, exist_ok=True)

    if not step:
        assert not substep, f"Substep {f} of step None maketh sense None"
        return rundir

    stepdir = rundir / step
    stepdir.mkdir(parents=False, exist_ok=True)

    if not substep or substep == "_":
        return stepdir

    substepdir = stepdir / substep
    substepdir.mkdir(parents=False, exist_ok=True)

    return substepdir


def check_config(c, at_location, must_exist=False, create=False):
    """Check if a config file exists at a location, and create it if it does not.
    Either a config must exist at a location, or it must not.
    """
    j = at_location / "config.json"

    if must_exist:
        if not j.exists():
            raise FileNotFoundError(f"Location {at_location} must have a config but does not."
                                    "\n\tInitialize the parent folders first.")
        return j

    if j.exists():
        raise FileExistsError(f"Location {at_location} seems to already have a config.")

    return (read_config.write(c, to_json=j) if create else True)


def initialize(config, step=None, substep=None, **kwargs):
    """Set up a run of the pipeline.
    """
    c = config; s = step; ss = substep
    LOG.info("Initialize workspace for config %s", pformat(c))
    run = get_rundir(c, s, ss)

    if not s:
        assert not ss, f"Substep {ss} of step None maketh sense None"
        _=check_config(c, at_location=run, must_exist=False, create=True)
        return run

    if not substep:
        _=check_config(c, at_location=run.parent, must_exist=True)
        return run

    _=check_config(c, at_location=run.parent.parent, must_exist=True)

    today, now = timestamp(now=True, format=lambda x, y: (x, y))

    day = run / today
    day.mkdir(parents=False, exist_ok=True)

    time = day / now
    time.mkdir(parents=False, exist_ok=False)

    with open(time / "INITIALIZED", 'w') as _:
        LOG.info("Initialized a TAP working space at: %s", time)

    LOG.warning("An sbatch script should be prepared and deposited in the workspace.")
    LOG.error("NotImplementedError(topological_pipeline.sbatch)")

    current_run = run.joinpath("current")
    if current_run.exists():
        current_run.unlink()
    current_run.symlink_to(time)
    return current_run


def cleanup(config, **kwargs):
    """Clean up the workspace.
    """
    raise NotImplementedError


def current(config, step=None, substep=None, **kwargs):
    """..."""
    run = get_rundir(config, step, substep)
    cwd = run / "current"

    if not cwd.exists():
        cwd = initialize(config, step, substep, **kwargs)

    return cwd


def locate_base(in_current_run, for_step):
    """..."""
    base = Path(in_current_run) / for_step
    if not base.exists():
        raise FileNotFoundError(f"A workspace folder {base} must be created\n"
                                "Use `tap --config=<location> init <step> <substep>` to initialize"
                                "a folder to run a substep of a pipeline step.\n"
                                "For example \n"
                                "`tap --config=<location> init analyze-connectivity simplices`\n"
                                "will create a directory to compute simplices in.")
    return base
