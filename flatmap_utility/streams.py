#!/usr/bin/env python3
# %%
# %matplotlib inline

# %%
# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,.pct.py:percent,.lgt.py:light,.spx.py:sphinx,md,Rmd,.pandoc.md:pandoc
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.1.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
We have modeled corticocortical inputs into the SSCx as fibers entering from the white-matter
and proceeding towards the pia, along the local voxel orientations.
We have defined a `flatspace` for the circuit to map these *white-matter projections* from one
SSCx-subregion to another, as well as to locate the entry points of thalamocortical projections.
Here we introduce a `flatmap_utility` to handle such `flatspace` projections.

Flatmapping involves the mapping of each voxel (and then each circuit-space point)
to a pixel in the flat-space.

A cylinder in the flat-space will become a cone of uneven diameter in the circuit-space.

Here we develop tools to visualize and analyze such flatmapped columns `fmap-columns`.
"""
# %%

from pathlib import Path

from lazy import lazy

import numpy as np
import pandas as pd

from voxcell.voxel_data import OrientationField
from bluepy import Cell

from connsense.io import logging

from flatmap_utility import flatmap_utility as fmutils
from flatmap_utility.subtargets import cache_evaluation_of, XYZ


LOG = logging.get_logger(__name__)

# %% [markdown]
"""
We will be mapping between the flatspace and real coordinates of the circuit.
It will be useful to estalish a terminology, so that it is easy to use and follow the code.
"""

def define_term(value, description):
    """Define a term.

    TODO: Consider making a class, like HD did for DMT.
    """
    LOG.info("Define a connsense term %s: \n\t%s", value, description)
    return value


CRCTSPACE = define_term("circuit_space",
                        "A tag that names the circuit space coordinates in all dataframes passed around.")
FMAPSPACE = define_term("flat_space",
                        "A tag that names the flat space coordinates in all dataframes passed around.")
FMAP_X = define_term("flat_x",
                     "A tag that names the flat space coordinate x within FMAPSPACE columns")

FMAP_Y = define_term("flat_y",
                     "A tag that names the flat space coordinate x within FMAPSPACE columns")

DEPTH = define_term("depth",
                    "A tag that names the flat space coordinate x within FMAPSPACE columns")

# %%

FXY = ["flat_x", "flat_y"]

# %%[markdown]
"""
A stream of information flows into the cortex at the white-matter as an axonal projection from
the thalamus. Assignment of post-synaptic gids for a single projection is done in the flat-space,
and not in the circuit-space.

Each individual fiber coming in from the thalamus is modeled as a flat-space circular based column.
Whether a cortical neuron recieves input from such a fiber should then be a function of the location
and shapes of the fiber's local axonal arbor, and the location and shapes of the neuron's dendrites.
Whatever the exact dependence, the *connection-probability*, fiber to neuron, should be a function
of the relative locations of the fiber and the neuron's soma.

The flat-space, where the connectivity assignments are made (I hope so -- TODO find out projectionizer),
may turn out to be very curved in the (physical) circuit space.

An individual streamline's trace can be computed by following local orientations: i.e defining some
ordering, or statistical summary to choose the next voxel until you are at the top.
This seems complicated to do.

However, we can use the flatmap to compute the traces of thalamic inputs.
Consider a straight up flat circular column in the flat-space.
We can consider that this *flat-column* represents the catchment area of an input thalamic
fiber that runs up from the white-matter towards pia, entering at the center of the column
at it's intersection with the white-matter.

Here we model a thalamic input fiber, and it's association with a subset of nearby cells,
as `class InputStream`
"""
# %%

def get_bins(xs, resolution, padding=0.1):
    """..."""
    xmin = np.min(xs) - padding
    xmax = np.max(xs) + padding

    try:
        bins = resolution["bins"]
    except KeyError:
        try:
            delta = resolution["delta"]
        except KeyError:
            try:
                nbins = resolution["nbins"]
            except KeyError:
                raise TypeError("resolution must provide one of `nbins, delta, bins`")
            else:
                bins = np.linspace(xmin, xmax, nbins + 1)
        else:
            assert "nbins" not in resolution,\
                "Resolution can be specified by only one of `bins, delta, nbins`."
            bins = np.arange(xmin, xmax, delta)
    else:
        assert "delta" not in resolution and "nbins" not in resolution, \
            "Resolution can be specified by only one of `bins, delta, nbins`."

    return bins[np.searchsorted(bins, xs) - 1]


def get_radial(positions, origin, orientation=None):
    """
    Get distance of positions from an `origin`.
    If `orientation` is provided, compute distance from an axis that starts at
    `origin` and runs along `orientation`.
    """
    positions_relative = positions - origin
    distance_origin = np.linalg.norm(positions_relative, axis=1)
    if orientation is None:
        return distance_origin

    cos_theta = np.dot(positions_relative, orientation) / distance_origin
    thetas = pd.Series(np.arccos(cos_theta), name="theta", index=positions.index)
    sin_theta = np.sqrt(1. - cos_theta ** 2)
    distances = (pd.Series(sin_theta * distance_origin, name="distance", index=positions.index)
                 .fillna(0.))
    return pd.concat([thetas, distances], axis=1)

def distribute_radially(circuit_space, flat_space=None, statistics=None):
    """..."""
    position = circuit_space.position.mean().to_numpy()

    statistics = statistics or "median"
    if isinstance(statistics, str):
        statistics = [statistics]

    _mo = circuit_space.orientation.mean()
    orientation = (_mo / np.linalg.norm(_mo)).to_numpy()

    radial = get_distance(circuit_space.position, position, orientation).agg(statistics)

    positions_orientations_shape = ([("position", "x"), ("position", "y"), ("position", "z"),
                                    ("orientation", "x"), ("orientation", "y"), ("orientation", "z")]
                                    + [("shape", f"radial_{statistic}") for statistic in statistics])
    circuit_space_dist =  pd.Series(np.concatenate([position, orientation, radial.values], axis=0),
                                    index=pd.MultiIndex.from_tuples(positions_orientations_shape))

    if not flat_space:
        return circuit_space_dist


class FlatMap:
    """A utility to handle and read data from a circut atlas' flatmap.NRRD.
    """
    def __init__(self, atlas):
        """..."""
        self._atlas = atlas

    @lazy
    def voxel_data(self):
        """..."""
        return self._atlas.load_data("flatmap")

    @lazy
    def orientations(self):
        """..."""
        return OrientationField.load_nrrd(Path(self._atlas.dirpath) / "orientation.nrrd")

    @staticmethod
    def frame_indices(in_voxel_data):
        """..."""
        fx = in_voxel_data.raw[:, :, :, 0]; fy = in_voxel_data.raw[:, :, :, 1]
        i, j, k = np.where(np.logical_and(fx > -1, fy > -1))
        return pd.DataFrame(np.array([i, j, k]).transpose(), columns=list("ijk"))

    @lazy
    def voxel_indices(self):
        """///"""
        return self.frame_indices(self.voxel_data)

    def orient(self, circuit_space_positions):
        """..."""
        return pd.DataFrame(self.orientations.lookup(circuit_space_positions.values)[:, :, 1],
                            columns=XYZ)

    @lazy
    def voxel_circuit_space(self):
        """..."""
        xyz = dict(i=Cell.X, j=Cell.Y, k=Cell.Z)
        positions = self.voxel_data.indices_to_positions(self.voxel_indices).rename(columns=xyz)
        orientations = self.orient(positions)
        return pd.concat([positions, orientations], keys=["position", "orientation"], axis=1)

    def fmap(self, circuit_space_positions):
        """..."""
        return (fmutils.supersampled_neuron_locations(circuit_space_positions, self.voxel_data,
                                                      self._atlas.load_data("orientation"),
                                                      include_dept=True)
                .rename(columns={"flat x": FMAP_X, "flat y": FMAP_Y}))

    @lazy
    def voxels_flat_space(self):
        """..."""
        positions = cache_evaluation_of(lambda _: self.fmap(self.voxels_circuit_space.position),
                                        on_circuit=self.atlas, as_attribute="fmap")
        return pd.concat([positions], axis=1, keys=["position"])


class FlatSpaceColumn:
    """A straight column in the circuit's flatmap space (flatspace for short)
    Positions in the circuit are mapped to a flat space, however each position can be assigned
    a depth to form a three dimensional flatspace!
    """

    def __init__(self, center, radius, circuit, *, atlas=None, flatmap=None, orientations=None,
                 number_cells=None, number_connections=None):
        """..."""
        self._center = np.array(center)
        self._radius = radius

        self._circuit = circuit
        self._atlas = atlas
        self._flatmap = flatmap
        self._orientations = orientations

        self._number_cells = number_cells
        self._number_connections = number_connections

    @lazy
    def name(self):
        """..."""
        return f"C{tuple(self._center)}"

    @lazy
    def center(self):
        """...We need a reverse transformation of coordinates."""
        return pd.Series({FMAPSPACE: self._center, CRCTSPACE: np.nan})

    @lazy
    def circuit(self):
        """..."""
        return self._circuit

    @lazy
    def atlas(self):
        """..."""
        return self._circuit.atlas if self._atlas is None else self._atlas

    @lazy
    def flatmap(self):
        """..."""
        return self.atlas.load_data("flatmap") if self._flatmap is None else self._flatmap

    @lazy
    def orientations(self):
        if self._orientations is None:
            return OrientationField.load_nrrd(Path(self.atlas.dirpath) / "orientation.nrrd")
        return self._orientations

    @lazy
    def voxels_indices(self):
        """..."""
        flatmap_x = self.flatmap.raw[:, :, :, 0]; flatmap_y = self.flatmap.raw[:, :, :, 1]
        i, j, k = np.where(np.logical_and(flatmap_x > -1, flatmap_y > -1))
        return pd.DataFrame(np.array([i, j, k]).transpose(), columns=list("ijk"))

    @lazy
    def voxels_circuit_space(self):
        """..."""
        xyz = {"i": Cell.X, "j": Cell.Y, "k": Cell.Z}
        positions = self.flatmap.indices_to_positions(self.voxels_indices).rename(columns=xyz)
        orientations = self.orient(positions)
        return pd.concat([positions, orientations], keys=["position", "orientation"], axis=1)

    def fmap(self, peas):
        """Flatmap the positions argued.
        """
        return (fmutils.supersampled_neuron_locations(peas,
                                                     self.flatmap,
                                                     self.atlas.load_data("orientation"),
                                                     include_depth=True)
                .rename(columns={"flat x": "flat_x", "flat y": "flat_y"}))

    @lazy
    def voxels_flat_space(self):
        """..."""

        positions = cache_evaluation_of(lambda _: self.fmap(self.voxels_circuit_space.position),
                                        on_circuit=self.atlas, as_attribute="fmap")
        return pd.concat([positions], axis=1, keys=["position"])

    @lazy
    def voxels(self):
        """..."""
        circuit_space = self.voxels_circuit_space
        flat_space = self.voxels_flat_space

        in_catchment = flat_space.index[self.get_mask(flat_space)]
        voxels = pd.concat([circuit_space.reindex(in_catchment), flat_space.reindex(in_catchment)],
                           axis=1, keys=["circuit_space", "flat_space"])

        return voxels.sort_values(by=("flat_space", "position", "depth"))

    def get_mask(self, positions_in_flat_space):
        """..."""
        radial = np.linalg.norm(positions_in_flat_space.position[FXY] - self._center, axis=1)
        return radial < self._radius

    def filter_column(self, circuit_space_positions):
        """..."""
        flat_space = fmutils.supersampled_neuron_locations(circuit_space_positions,
                                                           self.flatmap, self.orientations,
                                                           include_depth=True)
        mask = self.get_mask(flat_space)
        return circuit_space_positions[mask]

    def orient(self, circuit_space_positions):
        """..."""
        return pd.DataFrame(self.orientations.lookup(circuit_space_positions.values)[:,:,1],
                            columns=XYZ)

    @lazy
    def cells(self):
        """Neurons in this `FlatSpaceColumn`'s catchment.
        """
        xyzs = self.circuit.cells.get(properties=XYZ)
        circuit_space = pd.concat([xyzs, self.orient(xyzs)], keys=["position", "orientation"], axis=1)

        def fmap(using_circuit):
            return (fmutils.supersampled_neuron_locations(using_circuit, self.flatmap,
                                                          self.atlas.load_data("orientation"),
                                                          include_depth=True)
                    .rename(columns={"flat x": "flat_x", "flat y": "flat_y"}))

        flat_space = pd.concat([cache_evaluation_of(fmap, self.circuit, "fmap")], axis=1,
                               keys=["position"])
        in_catchment = flat_space.index[self.get_mask(flat_space)]

        cells = pd.concat([circuit_space.reindex(in_catchment), flat_space.reindex(in_catchment)],
                          axis=1, keys=["circuit_space", "flat_space"])

        return cells.sort_values(by=("flat_space", "position", "depth"))


    @lazy
    def number_cells(self):
        """..."""
        if self._number_cells:
            return self._number_cells
        return self.cells.shape[0]

    @staticmethod
    def values_space(s, in_position_data):
        """..."""
        try:
            return in_position_data[s]
        except KeyError:
            pass
        return None

    def position_relatively(self, xyzs):
        """...Position flatspace positions relative to the center of this `FlatSpaceColumn`.

        The simpler case would be that `xyzs` is just the flatspace positions, in which case we
        just subract the center of this `FlatSpaceColumn`.
        It could also have to columns, one for circuit-space, another flat-space positions.
        In this case we would have to subtract the circuit-space center of this `FlatSpaceColumn`.
        """
        spaces = {FMAPSPACE: self.values_space(FMAPSPACE, in_position_data=xyzs),
                  CRCTSPACE: self.values_space(CRCTSPACE, in_position_data=xyzs)}

        relative_spaces = {space: value - self.center[space] for space, value in spaces.items()
                           if value is not None}

        return xyzs.assign(**relative_spaces)

    def flatmap_positions(self, xyzs):
        """..."""
        return xyzs[[FMAP_X, FMAP_Y]]

    def measure_radial(self, distance_of):
        """...Measure radial distance in flatmap space from the flat center of this `FlatSpaceColumn`
        """
        relative = self.position_relatively(xyzs=distance_of)[FMAPSPACE]
        return np.linalg.norm(relative, axis=1)

    class DoesNotSeemToBeFlatMapData(AttributeError): pass

    def mask(self, dataframe, *, positions_within_radial):
        """...Positions must be in flatspace.
        --------------------------------------------------------------------------------------------
        Arguments
        --------------------------------------------------------------------------------------------
        dataframe :: of positions, or contains positions in a multi-indexed column

        positions_with_radius :: Float #that must be provided as a keyword argument.
        --------------------------------------------------------------------------------------------
        TODO: We can add further filters ...
        """
        def _mask(positions):
            """Mask the positions sub-dataframe in the input.
            """
            radial = self.measure_radial(distance_of=positions)
            return radial < positions_within_radial

        try:
            positions = dataframe.position
        except AttributeError:
            return _mask(positions=dataframe)
        return _mask(positions)

    def distribute_radially(self, circuit_space, statistics=None):
        """..."""
        position = circuit_space.position.mean().to_numpy()

        statistics = statistics or "median"
        if isinstance(statistics, str):
            statistics = [statistics]

        _mo = circuit_space.orientation.mean()
        orientation = (_mo / np.linalg.norm(_mo)).to_numpy()

        radial = get_distance(circuit_space.position, position, orientation).agg(statistics)

        positions_orientations_shape = ([("position", "x"), ("position", "y"), ("position", "z"),
                                        ("orientation", "x"), ("orientation", "y"), ("orientation", "z")]
                                        + [("shape", f"radial_{statistic}") for statistic in statistics]
                                        + [("flat", "x"), ("flat", "y")]
                                        + [("nodes", "number")])

        values = np.concatenate([position, orientation, radial.values, self._center, [self.number_cells]],
                                axis=0)

        return pd.Series(values, index=pd.MultiIndex.from_tuples(positions_orientations_shape))

    @staticmethod
    def move_index_value(of_statistics, to_column_level_of_dataframe):
        """Pandas aggregation operations that we use give dataframes with each statistic
        in the index. We want to move the statistic to a new level of the columns,
        thus makeing the dataframe wide from long in statistic.
        """
        def value(statistic):
            return to_column_level_of_dataframe.loc[[statistic]].reset_index(drop=True)

        return pd.concat([value(s) for s in of_statistics], axis=1, keys=of_statistics)

    @staticmethod
    def summarize_circuit_space(column, statistics):
        """..."""
        stats = column.circuit_space.position.agg(statistics)
        positions = FlatSpaceColumn.move_index_value(statistics, to_column_level_of_dataframe=stats)

        stats = column.circuit_space.orientation.agg(statistics)
        orientations = FlatSpaceColumn.move_index_value(statistics, to_column_level_of_dataframe=stats)

        return pd.concat([positions, orientations], axis=1, keys=["position", "orientation"])

    @staticmethod
    def summarize_circuit_space(column, statistics):
        """..."""
        def summarize_quantity(q):
            """..."""
            quantity = column.circuit_space[q].agg(statistics)
            if q == "orientation":
                norm = quantity.apply(lambda statistic: np.linalg.norm(statistic), axis=1)
                quantity = quantity.div(norm, axis=0)
            return FlatSpaceColumn.move_index_value(statistics, to_column_level_of_dataframe=quantity)

        positions = summarize_quantity("position")
        orientations = summarize_quantity("orientation")

        def get_overlap(by_statistic):
            """What is the overlap of the column's summary orientation (by argued statistic),
            with each orientation in the column.
            This overlap can serve as a proxy for the pairwise overlaps...
            """
            orientation = orientations[by_statistic].values[0]
            return pd.DataFrame([(column.circuit_space.orientation
                                  .apply(lambda r: np.dot(r.values, orientation), axis=1)
                                  .agg(["mean", "std"]))])

        overlaps = pd.concat([get_overlap(s) for s in statistics], axis=1, keys=statistics)

        return pd.concat([positions, orientations, overlaps],
                         axis=1, keys=["position", "orientation", "overlap"])


    @staticmethod
    def summarize_flat_space(column, statistics):
        """..."""
        stats = column.flat_space.position.agg(statistics)
        positions = FlatSpaceColumn.move_index_value(statistics, stats)

        return pd.concat([positions], axis=1, keys=["position"])

    @staticmethod
    def summarize_radial(column, statistics, in_space=None):
        """..."""
        in_space = in_space or CRCTSPACE
        assert in_space in (FMAPSPACE, CRCTSPACE), f"Unknown space {in_space}"

        if in_space == CRCTSPACE:
            positions = column.circuit_space.position
            center = positions.mean().to_numpy()

            orientations = column.circuit_space.orientation
            mean_orient = orientations.mean().to_numpy()
            axis = mean_orient / np.linalg.norm(mean_orient)

        else:
            positions = column.flat_space.position
            center = positions.mean().to_numpy()
            axis = np.array([0., 1., 0.])

        radial_positions = get_radial(positions, center, axis)
        stats = radial_positions.agg(statistics)
        radial = FlatSpaceColumn.move_index_value(statistics, stats)

        return pd.concat([radial], axis=1, keys=["radial"])

    @staticmethod
    def summarize_shape(column, statistics):
        """..."""
        circuit_space = FlatSpaceColumn.summarize_circuit_space(column, statistics)
        radial_circuit_space = FlatSpaceColumn.summarize_radial(column, statistics, CRCTSPACE)
        circuit_space_and_radial = pd.concat([circuit_space, radial_circuit_space], axis=1)

        fmap_space = FlatSpaceColumn.summarize_flat_space(column, statistics)
        radial_fmap_space = FlatSpaceColumn.summarize_radial(column, statistics, FMAPSPACE)
        fmap_space_and_radial = pd.concat([fmap_space, radial_fmap_space], axis=1)

        return pd.concat([circuit_space_and_radial, fmap_space_and_radial], axis=1,
                         keys=[CRCTSPACE, FMAPSPACE])

    def channel(self, of_positions, radius=None, depth_bins=None, statistics="median",
                keep_index=None):
        """Dig a channel as a dataframe of circuit-space and flatmap-space positions...
        that are within this `FlatSpaceColumn`.
        If a radius is provided, then a smaller (circular) column with that radius will be
        used to filter. This smaller column will be centered at this `FlatSpaceColumn`'s center,
        and the argued radius value should be smaller than it's radius
        If no radius is argued all the positions in this `FlatSpaceColumn` will be used.

        Argued depth bins will be used to bin the positions and find means.
        A column called `shape` will be returned with mean positions  and orientations for each bin.
        Each bin's shape will be the mean radial distance from the axis that is parallel
        to the mean orientation and pass through the mean position of the bin

        """
        assert of_positions in ("voxels", "cells"), f"Invalid positions {of_positions}"

        peas = self.voxels if of_positions=="voxels" else self.cells

        if peas.empty:
            return pd.DataFrame()

        mask = lambda: self.mask(peas[FMAPSPACE], positions_within_radial=radius)
        inchannel = peas[mask()] if radius is not None else peas

        depth_values = inchannel[FMAPSPACE].position.depth.values
        depth_bins = get_bins(depth_values, resolution={"nbins": depth_bins or 20})

        depth_index = ["depth_bin"] + (keep_index or ["gid"])
        depth_groups = (inchannel.reset_index().assign(depth_bin=depth_bins).set_index(depth_index)
                        .groupby("depth_bin"))

        def shape(at_depth):
            return self.summarize_shape(at_depth, statistics=statistics)

        shapes_at_depth = depth_groups.apply(shape).droplevel(None)
        return shapes_at_depth

    @staticmethod
    def lingress(channel_shape):
        """Fit a linear model to the channel shape.
        """
        import statsmodels.api as sm
        from statsmodels.formula import api as StatModel
        from patsy import ModelDesc

        description = ModelDesc.from_formula("radial ~ depth")
        input_data = {"radial": channel_shape.values, "depth": channel_shape.index.values}
        model = StatModel.ols(description, input_data)
        fit = model.fit()
        params = pd.concat([fit.params], axis=0, keys=["params"])
        pvalues = pd.concat([fit.pvalues], axis=0, keys=["pvalues"])
        error = pd.concat([pd.Series({"rsquared": fit.rsquared, "rsquared_adj": fit.rsquared_adj})],
                        axis=0, keys=["fit"])
        return params.append(pvalues).append(error)

    @staticmethod
    def lingress_conicity(channels, statistics=None, drop_ends=None):
        """Fit linear models to channel """

        def lingress_subtarget(s, channel):
            """..."""
            if drop_ends is not None:
                if drop_ends == 0:
                    channel = channel.iloc[1:]
                elif drop_ends == -1:
                    channel = channel.iloc[:-1]
                elif drop_ends == True or drop_ends == [0, -1]:
                    channel = channel.iloc[1:-1]
                else:
                    raise ValueError(f"Unhandled drop ends for {drop_ends}")

            if channel.empty:
                return

            radial = channel.circuit_space.radial

            def lingress_stat(istic):
                """..."""
                by_subtarget = pd.Index([s], name="subtarget")
                lr = FlatSpaceColumn.lingress(radial[istic].distance.droplevel("subtarget"))
                lr.columns = pd.MultiIndex.from_tuples([(f"conicity_{l0}", l1)
                                                        for l0, l1 in lr.index.values])

                return pd.DataFrame([lr], index=by_subtarget)

            return pd.concat([lingress_stat(istic) for istic in statistics], axis=1, keys=statistics)

        subtargets = channels.groupby("subtarget")
        return pd.concat([lingress_subtarget(s, channel) for s, channel in subtargets])


def dig_channels(in_subtargets, with_radius, in_circuit, using=None, depth_bins=20, statistics=None):
    """..."""
    statistics = statistics or ["min", "mean", "std", "median", "mad", "max"]

    def get_channel(for_subtarget):
        """..."""
        at_fmap_xy = np.array((for_subtarget.position[FMAP_X], for_subtarget.position[FMAP_Y]))
        fmap_column = FlatSpaceColumn(at_fmap_xy, with_radius, in_circuit,
                                      number_cells=for_subtarget.number.nodes,
                                      number_connections=for_subtarget.number.edges)
        return fmap_column.channel(of_positions=(using or "voxels"), radius=None,
                                   depth_bins=depth_bins, statistics=statistics)
                                   
    progress = [f"{int(p)}%"  for p in 100 * np.linspace(0, 1, len(in_subtargets))]
    channels_as_dataframes = in_subtargets.assign(progress=progress).apply(get_channel, axis=1)
    return pd.concat([c for c in channels_as_dataframes], keys=in_subtargets.index)


def compute_conicity(channels, statistics, drop_ends):
    """..."""
    lingress = FlatSpaceColumn.lingress_conicity(channels, statistics, drop_ends)

    column_depth = (channels.flat_space.position["median"].depth.groupby("subtarget")
                    .agg(lambda ds: ds.max() - ds.min()).rename("column_depth")
                    .loc[lingress.index])

    LOG.info("%s column depths ", len(column_depth))

    conicity_ratio_empirical = (channels.circuit_space.radial["mean"].distance.groupby("subtarget")
                                .agg(lambda sr: sr[-1] / sr[0]).rename("ratio_empirical")
                                .loc[lingress.index])

    def summarize(s):
        LOG.info("%s entries in %s lingress ", len(lingress[s]), s)

        conicity_top = lingress[s].params.Intercept.rename("top")
        conicity_slope = lingress[s].params.depth.rename("slope")
        conicity_rsquared = lingress[s].fit.rsquared.rename("rsquared")

        conicity_ratio_model = ((conicity_top.values + conicity_slope.values * column_depth)
                                / conicity_top.values).rename("ratio_model")

        return pd.concat([conicity_top, conicity_slope, conicity_rsquared,
                          conicity_ratio_model, conicity_ratio_empirical,
                          column_depth], axis=1)

    return pd.concat([summarize(s) for s in statistics], axis=1, keys=statistics)

def analyze_conicity(in_subtargets, with_radius, in_circuit, using=None,
                     depth_bins=20, statistics=None, to_summarize=None,
                     drop_ends=None):
    """Measure subtargets conicity and characterize it.
    TODO: What are these parameters?
    ~ Define a class that parameterizes an analysis of the flatmap-column conicity,
    ~ and then may be runs it too.

    in_subtargets: DataFrame containing flatspace position and node, edge numbers...

    """
    channels = dig_channels(in_subtargets, with_radius, in_circuit, using, depth_bins, statistics)

    conicity = compute_conicity(channels, statistics=to_summarize, drop_ends=drop_ends)

def compute_shapes(of_subtargets, with_radius, in_circuit, using=None, depth_bins=20,
                   statistics=None, to_summarize=None):
    """..."""
    channels = dig_channels(of_subtargets.position, with_radius, in_circuit, using, depth_bins, statistics)

    if not to_summarize:
        return channels

    conicity = FlatSpaceColumn.lingress_conicity(channels, statistics=to_summarize)

    return conicity.join(of_subtargets.number)
