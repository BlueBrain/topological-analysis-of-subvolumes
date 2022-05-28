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
Thalamic inputs into the cortex are modeled as fibers entering from the white-matter
and proceeding towars the pia, along a local orientation.
We have defined orientation for each voxel in the atlas.

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

from flatmap_utility import flatmap_utility as fmutils
from flatmap_utility.subtargets import cache_evaluation_of, XYZ

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
                bins = np.linspace(xmin, xmax, nbins)
        else:
            assert "nbins" not in resolution,\
                "Resolution can be specified by only one of `bins, delta, nbins`."
            bins = np.arange(xmin, xmax, delta)
    else:
        assert "delta" not in resolution and "nbins" not in resolution, \
            "Resolution can be specified by only one of `bins, delta, nbins`."
    return bins


def get_distance(positions, origin, orientation=None):
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
    sin_theta = np.sqrt(1. - cos_theta ** 2)
    return pd.Series(sin_theta * distance_origin, name="distance", index=positions.index).fillna(0.)


def distribute_radially(circuit_space):
    """..."""
    mean_position = circuit_space.position.mean()
    mean_orientation = circuit_space.orientation.mean()
    norm = np.linalg.norm(mean_orientation)
    return get_distance(circuit_space.position, mean_position, mean_orientation/norm)


class InputStream:
    """A thalamic input fiber passes trough the cortex, whitematter --> pia or pia --> whitematter,
    along local orientation.
    But there are other ways to think of a stream-line.
    """

    def __init__(self, circuit, *, atlas=None, flatmap=None, orientations=None,
                 center=None, radius=None):
        """...
        """
        self._center = np.array([2000., 2000.]) if center is None else center
        self._radius = 230.0 if radius is None else radius

        self._circuit = circuit
        self._atlas = atlas
        self._flatmap = flatmap
        self._orientations = orientations

    @lazy
    def name(self):
        """..."""
        return f"C{tuple(self._center)}"

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

    @lazy
    def voxels_flat_space(self):
        """..."""

        def fmap(_):
            return (fmutils.supersampled_neuron_locations(self.voxels_circuit_space.position,
                                                          self.flatmap,
                                                          self.atlas.load_data("orientation"),
                                                          include_depth=True)
                    .rename(columns={"flat x": "flat_x", "flat y": "flat_y"}))

        positions = cache_evaluation_of(fmap, self.atlas, "fmap")
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

    def filter_catchment(self, circuit_space_positions):
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
        """Neurons in this `InputStream`'s catchment.
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

    def catchment(self, resolution={"nbins": 100}, using="voxels"):
        """..."""
        assert using in ("cells", "voxels"), f"Cannot define catchement using {using}"

        data = self.cells if using=="cells" else self.voxels
        depth_bins = get_bins(data.flat_space.position.depth, resolution)

        depth_bins = np.searchsorted(depth_bins, data.flat_space.position.depth) - 1

        return data.reset_index().assign(depth_bin=depth_bins).set_index(["depth_bin", "gid"])

    def trace(self, resolution={"nbins": 100}, using="voxels"):
        """..."""
        catchment = self.catchment(resolution, using).circuit_space

        mean_positions = catchment.position.groupby("depth_bin").mean()
        mean_orientations = catchment.orientation.groupby("depth_bin").mean()
        norm = np.linalg.norm(mean_orientations, axis=1)

        return pd.concat([mean_positions, mean_orientations.apply(lambda c: c/norm, axis=0 )], axis=1,
                         keys=["position", "orientation"])


    def channel(self, resolution={"nbins": 100}, using="voxels"):
        """..."""
        catchment = self.catchment(resolution, using).circuit_space
        if catchment.empty:
            return None
        radial = pd.concat([distribute_radially(grp) for _, grp in catchment.groupby("depth_bin")])
        return catchment.assign(radial = radial.values)
