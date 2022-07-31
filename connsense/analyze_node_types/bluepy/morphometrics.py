#!/usr/bin/env python3
from lazy import lazy

import numpy as np
import pandas as pd

#from morphio.mut import Morphology
#from neurom.core.morphology import Morphology
import neurom as nm
from bluepy import Cell, Synapse


AXONAL = [nm.AXON]
DENDRITIC = [nm.APICAL_DENDRITE, nm.BASAL_DENDRITE]
NEURITES = AXONAL + DENDRITIC
SOMATIC = [nm.SOMA]


class Morphometrics:
    """..."""
    def __init__(self, morphology):
        """..."""
        self._morphology = morphology

    def get_neurite_type(self, nt, feature):
        """..."""
        return nm.get(feature, self._morphology, neurite_type=nt)

    @lazy
    def length(self):
        """..."""
        return {nt: self.get_neurite_type(nt, feature="total_length") for nt in NEURITES}

    @staticmethod
    def label(neurite_types, metric):
        """..."""
        if neurite_types is AXONAL:
            pattern = "axonal_{}"
        elif neurite_types is DENDRITIC:
            pattern = "dendritic_{}"
        else:
            raise ValueError(f"Unknown neurite type {neurite_types}")
        return pattern.format(metric)

    @lazy
    def volume(self):
        """..."""
        return {nt: self.get_neurite_type(nt, "total_volume") for nt in NEURITES}

    def get_length(self, neurite_type):
        """..."""
        if neurite_types in NEURITES:
            return self.length[neurite_type]
        return np.sum(self.length[nt] for nt in neurite_type)

    def get_volume(self, neurite_type):
        """..."""
        if neurite_types in NEURITES:
            return self.volume[neurite_type]
        return np.sum(self.volume[nt] for nt in neurite_type)

    def get(self, neurite_types=None):
        """..."""
        if neurite_types:
            length = self.get_length(neurite_types)
            volume = self.get_volume(neurite_types)
            return {self.label(neurite_types, "length"): length, self.label(neurite_types, "volume"): volume}

        metrics = self.get(AXONAL)
        metrics.update(self.get(DENDRITIC))
        return metrics


def groupby_shape(in_circuit):
    """Group biophysical morphologies used in a BBP circuit by their shapes.
    """
    circuit_mtypes = in_circuit.cells.get(properties=Cell.MTYPE)
    morphologies = in_circuit.cells.get(properties=Cell.MORPHOLOGY)
    return (pd.concat([circuit_mtypes.apply(lambda mtype: '_'.join(mtype.split('_')[1:])), morphologies], axis=1)
            .groupby(Cell.MTYPE).apply(lambda mtype_group: list(mtype_group.morphology.unique())))


def measure_morphology(metrics, in_circuit):
    """..."""
    morph_type = circuit.config["morphology_type"]
    morph_dir = circuit.config["morphologies"]

    def get_filepath(morphology):
        """..."""
        filename = f"{morphology}.{morph_type}"
        return morph_dir / filename

    def get_morphology(m):
        """..."""
        at_filepath = get_filepath(Morphology=m)
        return nm.load_morphology(at_filepath)

    def to_apply(m_label):
        """..."""
        morphology = get_morphology(m)
        morphometrics = Morphometrics(morphology)
        return morphometrics.get()

    return to_apply


def measure_subtarget(metrics, in_circuit):
    """..."""
    def to_apply(subtarget):
        """..."""
        morphologies = pd.Series(subtarget, name=Cell.MORPHOLOGY)
        return pd.concat([morphologies, morphologies.apply(measure_morphology(metrics, in_circuit))], axis=1)

    return to_apply


def measure(in_circuits, metrics, of_subtargets):
    """..."""
    def measure_circuit(c, labeled):
        """..."""
        measurement = of_subtargets.xs(labeled, level="circuit").apply(measure_subtarget(metrics, in_circuit=c))
        return pd.concat(measurement.values, keys=measurement.index.values, names=["mtype"])

    return pd.concat([measure_circuit(c, labeled=l) for l, c in in_circuits.items()], axis=0,
                     keys=[l for l,_ in in_circuits.items()], names=["circuit"])
