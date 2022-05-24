"""Configure subtargets circuit cells defined using the flatmap
of circuit atlas.
"""
from collections.abc import Mapping
from pathlib import Path
from lazy import lazy

import numpy as np
import pandas as pd

from bluepy import Circuit
from bluepy.exceptions import BluePyError
from voxcell.voxel_data import VoxelData

from ..io import read_config, logging
LOG = logging.get_logger("define-subtargets")


class SubtargetsConfig:
    """Define and load subtargets in a circuit's flatmap."""

    @staticmethod
    def read_json(from_object, reader):
        """
        Read JSON from an object, using a reader.
        Notice that `from_object` may already be a dict.
        This makes `SubtargetsConfig` definition makes more flezible,
        allowing us to initialize with a dict that may have been read by
        an config interpreter upstream in the stream.
        """
        if isinstance(from_object, Mapping):
            return from_object

        try:
            path = Path(from_object)
        except TypeError:
            return from_object

        reader = reader or read_config
        config = reader.read(path)
        return config

    def __init__(self, config, label=None, reader=None):
        """
        config : Mapping or path to a JSON file that contains one.
        label  : Label for the subtargets (sub)-section in the config.
        """

        config = self.read_json(config, reader)
        assert isinstance(config, Mapping), type(config)

        self._config = config
        self._label = label or "define-subtargets"

    @staticmethod
    def load_circuit(with_maybe_config):
        """..."""
        try:
            circuit = Circuit(with_maybe_config)
        except BluePyError:
            circuit = with_maybe_config
        return circuit

    @lazy
    def input_circuit(self):
        """..."""
        paths = self._config["paths"]

        input_circuit = paths["circuit"]
        try:
            configs = input_circuit.items
        except AttributeError:
            config = input_circuit
            return {"_": self.load_circuit(config)}
        return {label: self.load_circuit(config) for label, config in configs()}

    @lazy
    def input_atlas(self):
        """..."""
        try:
            circuits = self.input_circuit.items
        except AttributeError:
            return self.input_circuit.atlas
        return {label: circuit.atlas for label, circuit in circuits()}

    @lazy
    def default_flatmap(self):
        """..."""
        try:
            atlases = self.input_atlas.items
        except AttributeError:
            return self.input_atlas.load_data("flatmap")
        return {circuit: atlas.load_data("flatmap") for circuit, atlas in atlases()}

    @lazy
    def input_flatmap(self):
        """Resolve atlas flatmap for each input circuit."""
        paths = self._config["paths"]

        try:
            flatmap = paths["flatmap"]
        except KeyError:
            return self.default_flatmap

        try:
            flatmap_nrrd = Path(flatmap)
        except TypeError:
            pass
        else:
            flatmap = VoxelData.load_nrrd(flatmap_nrrd)
            return {c: flatmap for c in self.circuits.keys()}

        assert isinstance(flatmap, Mapping)

        def resolve_between(circuit, flatmap_nrrd):
            """..."""
            if flatmap_nrrd:
                return VoxelData.load_nrrd(flatmap_nrrd)
            return circuit.atlas.load_data("flatmap")

        return {c: resolve_between(circuit, flatmap.get(c, None))
                for c, circuit in self.input_circuit.items()}

    @lazy
    def parameters(self):
        """..."""
        return self._config.get("parameters", {}).get(self._label, {})

    def define_subtarget(self, group, using_spec):
        """Define a group of subtarget.
        Our main example is that of a grid of depth-wise columnar subtargets of
        a neocortical circuit , with the grid covering a 2D flatmap.
        In general the label `grid` may reference any definition that generates
        circuit subtargets, and is parameterized `using_spec`. and need not strictly
        cover the flatmap / circuit.

        A use-case separate from the cortical flapmap hexgrid, might be a set of
        columns defined for the Hippocampus CA1 circuit.
        """
        if group == "hexgrid":
            return Hexgrid(self, using_spec)

        if group == "central_columns":
            return CentralColumns(self, using_spec)

        raise NotImplementedError(f"Definition {group} of subtargets.")

    @lazy
    def definitions(self):
        """Groups of subtargets defined in the config.
        """
        configured = self.parameters.get("definitions", {}).items()
        return {s: self.define_subtarget(group=s, using_spec=c) for s, c in configured}

    @lazy
    def target(self):
        """Base target to subset --- the default `None` will indicate that
        the whole circuit is the base target.
        """
        return self.parameters.get("base_target", None)

    def argue(self):
        """..."""
        for label, circuit in self.input_circuit.items():
            yield (label, circuit, self.input_flatmap[label])

    @lazy
    def output(self):
        """..."""
        return self._config["paths"]["output"]["steps"][self._label]


class Hexgrid:
    """Subtargets that form a grid of hexagons in the flatmap.
    """
    def __init__(self, config, parameters):
        """config: a SubtargetsConfig.
        """
        self._config = config
        self._parameters = parameters

    @lazy
    def mean_target_size(self):
        """..."""
        try:
            value = self.parameters["mean_target_size"]
        except KeyError:
            return None

        assert self.target_radius is None,\
            "Cannot set both radius and mean target size, only one."
        return value

    @lazy
    def target_radius(self):
        """For example, if using hexagons,
        length of the side of the hexagon to tile with.
        """
        try:
            value = self.parameters["radius"]
        except KeyError:
            return None

        assert self.mean_target_size is None,\
            "Cannot set both radius and mean target size, only one."

        return value

    @lazy
    def tolerance(self):
        """Relative tolerance, a non-zero positive number less than 1 that determines
        the origin, rotation, and radius of the triangular tiling to use for binning.
        """
        return self.parameters.get("tolerance", None)


    def define(self, sample=None, format=None):
        """Define subtargets for all input (circuit, flatmap).
        """
        format = format or "wide"
        assert format in ("wide", "long")

        subtargets = pd.concat([self.generate(c) for c in self.input_circuit[c]])
        if format == "long":
            return subtargets

        variables = ["circuit", "gid", "flat_x", "flat_y"]
        index_vars = ["circuit", "subtarget", "flat_x", "flat_y"]

        def enlist(group):
            """..."""
            return pd.Series({"flat_x": np.mean(group["flat_x"]),
                              "flat_y": np.mean(group["flat_y"]),
                              "gids": group["gid"].to_list()})

        return subtargets[variables].groupby(index_vars).apply("enlist").gids


class CentralColumns:
    """Define central-columns as circuit subtargets to run TAP on.
    """
    def __init__(self, config, spec):
        """..."""
        self._config = spec
        subtarget_items = spec["subtargets"].items()
        self._descriptions = {subtarget: description for subtarget, description in  subtarget_items
                              if subtarget != "COMMENT" and isinstance(description, Mapping)}

    def describe_subtarget(self, s):
        """..."""
        return self._descriptions[s]

    def describe_cells(self, subtarget):
        """..."""
        return self._descriptions[subtarget]["cells"]

    def describe_voxels(self, subtarget):
        """..."""
        return self._descriptions[subtarget]["voxels"]
