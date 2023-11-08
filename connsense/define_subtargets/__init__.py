"""Targets defined using the circuit's flatmap.
ad_n


NOTE: As of today (20220207), this file is under a refactor towards a uniform configuration
~     each pipeline step.
"""
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import numpy as np

from bluepy import Cell

from flatmap_utility import subtargets as flatmap_subtargets
from flatmap_utility.tessellate import TriTille

from ..import plugins
from .config import SubtargetsConfig
from ..pipeline import workspace
from ..io.write_results import write, default_hdf
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


def generate_hexgrid_0(radius, label, circuit, flatmap):
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



def define_0(config, sample=None, fmt=None):
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
        subtargets = pd.concat([generate_hexgrid_0(g, *args, using_parameters=p) for g, p in defined],
                               axis=0, keys=[grid for grid,_ in defined], names=["grid"])
        subtargets = {g: generate_hexgrid_0(g, *args, using_parameters=p) for g, p in defined}
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


def parameterize_hexgrid(subtargets_config):
    """..."""
    return {key: value for key, value in subtargets_config.definitions["hexgrid"]
            if key != "COMMENT"}


def generate_hexgrid(subtargets_config, fmt):
    """..."""
    parameters = parameterize_hexgrid(subtargets_config)

    def generate(label, circuit, flatmap):
        """..."""
        subtargets = (flatmap_subtargets.generate(circuit, flatmap, **parameters)
                      .assign(circuit=label)
                      .rename(columns={"grid_x": "flat_x", "grid_y": "flat_y"}))
        LOG.info("Defined %s hexgrid-subtargets for circuit %s", len(subtargets), label)
        return subtargets

    configured = subtargets_config.argue()
    subtargets = pd.concat([generate(label=l, circuit=c, flatmap=f) for l, c, f in configured])

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


def parameterize_predefined(subtargets_config):
    """..."""
    return {key: value for key, value in subtargets_config.definitions["predefined"]
            if key != "COMMENT"}

def get_predefined_groups(subtargets_config):
    """..."""
    configured = subtargets_config.definitions["predefined"]["groups"]
    return {key: value for key, value in configured.items() if key != "COMMENT"}

def generate_predefined(subtargets_config, fmt):
    """..."""
    assert fmt == "wide"
    circuits = subtargets_config.input_circuit.items()
    subtargets_group = get_predefined_groups(subtargets_config)
    LOG.info("Generate for %s circuits predefined subtarget groups: \n%s",
             len(circuits), subtargets_group.keys())

    def generate_group(g, subtargets):
        """..."""
        subtargets = pd.Index(subtargets, name="subtarget")

        def in_circuit(c, labeled):
            """..."""
            cells = c.cells
            return pd.Series([cells.ids(s) for s in subtargets], name="gids", index=subtargets)

        return pd.concat([in_circuit(c, labeled=l) for l, c in circuits], axis=0,
                         keys=[l for l, _ in circuits], names=["circuit"])

    return pd.concat([generate_group(g, subtargets=ss) for g, ss in subtargets_group.items()])


def define_0(subtarget_type, using_config, fmt=None):
    """..."""
    fmt = fmt or "wide"
    assert fmt in ("wide", "long"), f"Unknown format {fmt}"

    subtargets_config = SubtargetsConfig(using_config)
    try:
        subtarget_def = subtargets_config.definitions[subtarget_type]
    except KeyError as kerr:
        raise ValueError(f"Unknown subtarget type {subtarget_type}") from kerr

    return subtarget_def.generate(subtargets_config.input_circuit)


def read(config):
    """..."""
    try:
        path = Path(config)
    except TypeError:
        assert isinstance(config, Mapping)
        return config
    return  read_config.read(path)


def output_specified_in(configured_paths, and_argued_to_be):
    """..."""
    steps = configured_paths["steps"]
    to_hdf_at_path, under_group = steps.get(STEP, default_hdf(STEP))

    if and_argued_to_be:
        to_hdf_at_path = and_argued_to_be

    return (to_hdf_at_path, under_group)


def input_circuits(definition, in_config, tap):
    """..."""
    configured = in_config["definitions"][definition]["input"]["circuit"]
    return pd.Series(configured, index=tap.subset_index("circuit", configured), name="circuit")


def extract_subtargets(definition, in_config, tap):
    """..."""
    defparams = in_config["definitions"][definition]
    members = in_config.get("members", defparams.get("members", None))
    _, load = plugins.import_module(defparams["loader"])

    def index_subtarget(values):
        """..."""
        return pd.Series(values, index=pd.RangeIndex(0, len(values), 1, name="subtarget_id"),
                         name="subtarget")

    def from_circuit(c, subtargets=None):
        """..."""
        if subtargets is not None:
            return subtargets.apply(lambda s: load(circuit=c, subtarget=s)).rename("gids")
        return load(circuit=c)

    circuits = input_circuits(definition, in_config, tap).apply(tap.get_circuit)
    kwargs = defparams.get("kwargs", {})

    if members and isinstance(members, list):
        members = index_subtarget(members)
        info = None
        subtargets = (pd.concat([members.apply(lambda s: load(c, s, **kwargs)).rename("gids")
                                for c in circuits], axis=0,
                                keys=circuits.index.values, names=[circuits.index.name])
                      .reorder_levels(["subtarget_id", "circuit_id"]))
        return (members, info, subtargets)


    if "conntility_loader_cfg" in kwargs:
        assert len(circuits) == 1,\
            "unique cell-based subtargets can be defined only for a single circuit analysis"
        circuit = circuits.iloc[0]
        members, info, subtargets = load(circuit, **kwargs)
        subtargets = (pd.concat([subtargets], keys=circuits.index.values,
                                names=[circuits.index.name])
                      .reorder_levels(["subtarget_id", "circuit_id"]))
        return (members, info, subtargets)

    assert "annotation" in "kwargs" and "info" in kwargs, "Missing subtarget info"

    from .flatmap import read_subtargets, load_nrrd
    subtargets_with_info = read_subtargets(kwargs["info"])
    members = subtargets_with_info["subtarget"]
    info = subtargets_with_info.drop(columns="subtarget")
    subtargets = (pd.concat([(load(circuit=c, path=kwargs["annotation"])
                              .reindex(members.index, fill_value=[]))
                                for c in circuits], axis=0,
                            keys=circuits.index.values, names=circuits.index.names)
                    .reorder_levels(["subtarget_id", "circuit_id"]))
    return (members, info, subtargets)


    raise NotImplementedError(f"NOT-YET when members of type {type(members)}")


def run(config, substep=None, in_mode=None, output=None, **kwargs):
    """..."""
    from ..pipeline.store.store import HDFStore as TapStore
    config = read(config)
    input_paths, output_paths = check_paths(config, STEP)

    parameters = config["parameters"][STEP]
    definition = substep

    LOG.warning("Define %s subtargets inside the circuit %s", substep,
                parameters["definitions"][substep]["input"]["circuit"])

    in_rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)
    to_define_subtargets_in = workspace.locate_base(in_rundir, for_step=STEP)
    LOG.warning("\tworking in %s", to_define_subtargets_in)

    subtargets, subtarget_info, gidses = extract_subtargets(definition, in_config=parameters,
                                                            tap=TapStore(config))

    connsense_h5, define_subtargets = output_specified_in(output_paths, and_argued_to_be=output)
    under_group = define_subtargets + "/" + definition

    subtargets.to_hdf(connsense_h5, key=under_group+"/name")
    if subtarget_info is not None:
        subtarget_info.to_hdf(connsense_h5, key=under_group+"/info")
    gidses.to_hdf(connsense_h5, key=under_group+"/data")

    return (connsense_h5, under_group)

    # subtargets = pd.Series(members, name="subtarget", index=pd.Index(range(len(members)), name="subtarget_id"))

    # sbtcfg = SubtargetsConfig(config)
    # circuit_label = definition["input"]["circuit"]
    # circuit = sbtcfg.input_circuit[circuit_label]

    # _, load = plugins.import_module(definition["loader"])

    # subtargets_gids = pd.concat([subtargets.apply(lambda s: load(circuit, s)).rename("gids")], axis=0,
    #                             keys=[circuit_label], names=["circuit"])

    # LOG.info("Defined %s %s-subtargets.", len(subtargets), substep)
    # to_output = output_specified_in(output_paths, and_argued_to_be=output)
    # connsense_h5, group = to_output

    # write(subtargets, to_path=(connsense_h5, group))
    # write(subtargets, to_path=(out_h5, hdf_group+"/index"), format="fixed")
    # output = write(subtargets_gids, to_path=(out_h5, hdf_group+"/"+substep), format="fixed")
    # LOG.info("DONE: define-subtargets %s %s", substep, output)
    # return output


def run_2(config, substep=None, in_mode=None, output=None, **kwargs):
    """..."""
    config = read(config)
    input_paths, output_paths = check_paths(config, STEP)
    LOG.warning("Define %s subtargets inside the circuit %s", substep, input_paths)

    in_rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)
    to_define_subtargets_in = workspace.locate_base(in_rundir, for_step=STEP)
    LOG.warning("\tworking in %s", to_define_subtargets_in)

    parameters = config["parameters"][STEP]
    members = parameters["members"]
    definition = parameters["definitions"][substep]

    subtargets = pd.Series(members, name="subtarget", index=pd.Index(range(len(members)), name="subtarget_id"))

    sbtcfg = SubtargetsConfig(config)
    circuit_label = definition["input"]["circuit"]
    circuit = sbtcfg.input_circuit[circuit_label]

    _, load = plugins.import_module(definition["loader"])

    subtargets_gids = pd.concat([subtargets.apply(lambda s: load(circuit, s)).rename("gids")], axis=0,
                                keys=[circuit_label], names=["circuit"])

    LOG.info("Defined %s %s-subtargets.", len(subtargets), substep)
    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    out_h5, hdf_group = to_output
    write(subtargets, to_path=(out_h5, hdf_group+"/index"), format="fixed")
    output = write(subtargets_gids, to_path=(out_h5, hdf_group+"/"+substep), format="fixed")
    LOG.info("DONE: define-subtargets %s %s", substep, output)
    return output


def run_1(config, substep=None, in_mode=None, parallelize=None,
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
    config = read(config)
    input_paths, output_paths = check_paths(config, STEP)
    LOG.warning("Define %s subtargets inside the circuit %s", substep, input_paths)

    in_rundir = workspace.get_rundir(config, mode=in_mode, **kwargs)
    to_define_subtargets_in = workspace.locate_base(in_rundir, for_step=STEP)
    LOG.warning("\tworking in %s", to_define_subtargets_in)

    subtargets = define(subtarget_type=substep, using_config=config, fmt="wide")

    LOG.info("Defined %s %s-subtargets.", len(subtargets), substep)
    to_output = output_specified_in(output_paths, and_argued_to_be=output)
    LOG.info("...write them to %s", to_output)
    output = write(subtargets, to_path=to_output, format="fixed")
    LOG.warning("DONE: define_subtargets %s %s", substep, output)
    return output


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
        subtargets = define_0(config, sample=sample, fmt="wide")
        LOG.info("Done defining %s subtargets.", subtargets.shape)

    LOG.info("Write result to %s", output)
    if dry_run:
        LOG.info("TEST pipeline plumbing.")
    else:
        output = write(subtargets, to_path=(hdf_path, hdf_group))
        LOG.info("Done writing results to %s", output)

    LOG.warning("DONE, defining subtargets.")
    return f"result saved at {output}"
