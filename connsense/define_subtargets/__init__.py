"""Targets defined using the circuit's flatmap.

NOTE: As of today (20220207), this file is under a refactor towards a uniform configuration
~     each pipeline step.
"""
from pathlib import Path

import pandas as pd
import numpy as np

from bluepy import Cell

from flatmap_utility import subtargets as flatmap_subtargets
from flatmap_utility.tessellate import TriTille

from .config import SubtargetsConfig
from ..pipeline import workspace
from ..io.write_results import write
from ..io import read_config, logging
from ..io.read_config import check_paths

XYZ = [Cell.X, Cell.Y, Cell.Z]

STEP = "define-subtargets"
LOG = logging.get_logger(STEP)


def get_cell_ids(circuit, target=None, sample=None):
    """..."""
    gids = pd.Series(circuit.cells.ids(target), name="gid")

    if isinstance(sample, int):
        return gids.sample(n=sample)

    if isinstance(sample, float):
        return gids.sample(frac=sample)

    assert not sample, sample

    return gids


def get_cell_positions(circuit, target=None, sample=None):
    """..."""
    positions = circuit.cells.get(target, properties=XYZ)
    positions.index.name = "gid"
    if isinstance(sample, int):
        return positions.sample(n=sample)

    if isinstance(sample, float):
        return positions.sample(frac=sample)

    assert not sample, sample

    return positions


def cached(circuit, method):
    """..."""
    try:
        value = getattr(circuit, method.__qualname__)
    except AttributeError:
        value = method(circuit)
        setattr(circuit, method.__qualname__, value)
    return value


def flatmap_positions(circuit):
    """..."""
    flatmap = circuit.atlas.load_data("flatmap")
    orientations = circuit.atlas.load_data("orientation")
    return (flattened
            .supersampled_neuron_locations(circuit, flatmap, orientations)
            .rename(columns={"flat x": "x", "flat y": "y"}))


def get_flatmap(circuit, target=None, sample=None, subpixel=True, dropna=True):
    """..."""
    LOG.info("GET flatmap for target %s sample %s%s",
             target, sample, ", with subsample resolution." if subpixel else ".")

    if not subpixel:
        flatmap = circuit.atlas.load_data("flatmap")
        positions = get_cell_positions(circuit, target, sample)

        fpos =  pd.DataFrame(flatmap.lookup(positions.values),
                             columns=["x", "y"], index=positions.index)

        LOG.info("DONE getting flatmap")
        return fpos[np.logical_and(fpos.x >= 0, fpos.y >= 0)]

    flat_xy = cached(circuit, flatmap_positions)
    if target is not None or sample is not None:
        gids = get_cell_ids(circuit, target, sample)
        in_target = flat_xy.reindex(gids)
    else:
        in_target = flat_xy

    assert in_target.index.name == "gid", in_target.index.name
    LOG.info("DONE getting flatmap")
    return in_target.dropna() if dropna else in_target


def flatmap_hexbin(circuit, radius=10, gridsize=120, sample=None):
    """Bin circuit cell's flatmap coordinates in hexagonal grid.
    """
    positions = get_cell_positions(circuit, sample=sample)
    flatmap = get_flatmap(circuit, positions)

    tritiling = TriTille(radius)

    bins = tritiling.bin_hexagonally(positions, use_columns_row_indexing=True)
    return bins


def name_subtarget(hexbin):
    """Name the subtargets using their column and row index."""
    return f"R{hexbin.row};C{hexbin.col}"


    """...
    Input
    ----------
    circuit : BluePyCircuit
    flatmap : flatmap NRRD, omit to use the one on the circuit's atlas
    radius : of the balls that would pack the resulting hexgrid
    size : approximate number of neurons per subtarget.
    target : a BluePy target cell type to define subtargets in, omit to use all cells.
    sample : an int > 1 or float < 1 to use a random sample of the entire target.
    naming_scheme : A call-back or a dict to use to name the subtargets from their location

    Output
    ----------
    pd.Series
    Indexed by : flat_x, flat_y : the x, y-coordinates of the center of the subtarget in the flatmap.
    ~            subtarget_name : a pretty name for the subtarget.
    Values : lists of gids
    """


def generate_hexgrid(radius, label, circuit, flatmap):
    """TODO: subset to the configured base target.
    """
    LOG.info("GENERATE subtargets for circuit %s", label)

    subtargets = flatmap_subtargets.generate(circuit, flatmap, radius)
    if label:
        subtargets = subtargets.assign(circuit=label)
    subtargets = subtargets.rename(columns={"grid_x": "flat_x", "grid_y": "flat_y"})
    LOG.info("DONE %s subtargets for circuit %s", subtargets.shape[0], label)
    return subtargets


def generate_central_columns(from_descriptions, circuit, atlas):
    """Generate the ...."""

def generate_group(g, label, circuit, flatmap, parameters):
    """Generate a group of subtargets.
    """
    if g != "hexgrid":
        raise NotImplementedError(f"Subtargets defined as {g}")

    return generate_hexgrid(parameters.target_radius, label, circuit, flatmap)



def define(config, sample=None, fmt=None):
    """Define configured subtargets.
    Arguments
    ----------
    config : A `SubtargetConfig` or path to a JSON file...
    sample : A float less than 1 or an int to sample from configured cells.
    fmt : Format long  or wide of the output dataframe.
    """
    if not isinstance(config, SubtargetsConfig):
        config = SubtargetsConfig(config)

    fmt = fmt or "wide"
    assert fmt in ("wide", "long")

    LOG.info("Compute sub-targets with:")
    LOG.info("\tinput circuits %s: ", config.input_circuit.keys())
    LOG.info("\tinput flatmaps %s: ", config.input_flatmap.keys())
    LOG.info("\twith flatmap hexagons of radius %s:", config.target_radius)
    LOG.info("\toutput in format %s goes to %s", format, config.output)

    def generate(label, circuit, flatmap):
        """TODO: subset to the configured base target.
        """
        args = (label, circuit, flatmap)
        defined = config.definitions.items()
        subtargets = pd.concat([generate_grid(g, *args, using_parameters=p) for g, p in defined],
                               axis=0, keys=[grid for grid,_ in defined], names=["grid"])
        subtargets = {g: generate_grid(g, *args, using_parameters=p) for g, p in defined}
        return subtargets

    subtargets = pd.concat([generate(*args) for args in config.argue()])

    if fmt == "long":
        return subtargets

    def enlist(group):
        """..."""
        gids = pd.Series([group["gid"].values], index=["gids"])
        return group[["flat_x", "flat_y"]].mean(axis=0).append(gids)

    columns = ["circuit", "gid", "flat_x", "flat_y"]
    index_vars = ["circuit", "grid", "subtarget", "flat_x", "flat_y"]
    subtargets_gids = subtargets[columns].groupby(index_vars).apply(enlist).gids
    LOG.info("DONE %s subtargets for all circuits.", subtargets_gids.shape[0])
    return subtargets_gids


def read(config):
    """..."""
    try:
        config = read_config.read(config)
    except TypeError:
        assert isinstance(config, Mapping)
    return config


def run(config, action, substep=None, in_mode=None, parallelize=None,
        output=None, batch=None, sample=None, tap=None, **kwargs):
    """Run the definition(s) of subtargets.
    We will implement multiple definitions, just like analyze-connectivity...
    The definition can be passed as the substep.

    substep :: The definition to run, default behavior is to define the hexgrid,
    ~          already available here.
    parallelize :: The parallelization config (or path to JSON),
    ~              that should list a specification for the required definition
    batch :: A batch of sub-computations --- TODO how can a definition be subcomputed?
    ~        For analye-connecivity we have parallelized subtargets.
    ~        Here we are defining them?
    ~        We could parallelize the hex-grid pixels.
    ~        Maybe we are already doing that!
    ~        Let us find out by refactoring.
    sample :: Sample the hex-grid.
    tap :: The TAP-HDFStore
    kwargs :: keyword arguments that will be dropped --- but may make sense to another step
    ~        These are passed by the pipeline run interface...
    """
    LOG.warning("Define a %s inside the circuit %s", definition, input_paths)

    subtargets_config = SubtargetsConfig(config)

    in_rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)
    to_define_subtargets_in = workspace.locate_base(in_rundir, for_step=STEP)


    raise NotImplementedError

def run_0(config, in_mode=None, parallelize=None, *args,
        output=None, sample=None, dry_run=None, **kwargs):
    """Run generation of subtargets based on a TAP config.
    TODO Deprecate this
    Post beta release with analyze connectivity downstream working,
    we have come back to define subtargets (with an urgency).
    The run method will be made to confirm with that implemented for
    analyze-connectivity run

    TODO
    ----------
    Use a `rundir` as the directory to run the definition of subtargets.
    """
    LOG.warning("Get subtargets for config %s", config)

    if parallelize and STEP in parallelize and parallelize[STEP]:
        LOG.error("NotImplemented yet, parallilization of %s", STEP)
        raise NotImplementedError(f"Parallilization of {STEP}")

    if sample:
        LOG.info("Sample %s from cells", sample)
        sample = float(sample)
    else:
        sample = None

    config = SubtargetsConfig(config)

    hdf_path, hdf_group = config.output

    if output:
        try:
            path = Path(output)
        except TypeError as terror:
            LOG.info("Could not trace a path from %s,\n\t because  %s", output, terror)
            try:
                hdf_path, hdf_group = output
            except TypeError:
                raise ValueError("output should be a tuple (hdf_path, hdf_group)"
                                 "Found %s", output)
            except ValueError:
                raise ValueError("output should be a tuple (hdf_path, hdf_group)"
                                 "Found %s", output)
        else:
            hdf_path = path

    LOG.info("Output in %s\n\t, group %s", hdf_path, hdf_group)

    LOG.info("DISPATCH the definition of subtargets.")
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        subtargets = define(config, sample=sample, fmt="wide")
        LOG.info("Done defining %s subtargets.", subtargets.shape)

    LOG.info("Write result to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        output = write(subtargets, to_path=(hdf_path, hdf_group))
        LOG.info("Done writing results to %s", output)

    LOG.warning("DONE, defining subtargets.")
    return f"result saved at {output}"
