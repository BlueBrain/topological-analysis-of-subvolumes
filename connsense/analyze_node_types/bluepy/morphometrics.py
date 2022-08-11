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
        if neurite_type in NEURITES:
            return self.length[neurite_type]
        return np.sum(self.length[nt] for nt in neurite_type)

    def get_volume(self, neurite_type):
        """..."""
        if neurite_type in NEURITES:
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
        return pd.Series(metrics)



def measure(morphology):
    """..."""
    return Morphometrics(nm.load_morphology(morphology)).get()
