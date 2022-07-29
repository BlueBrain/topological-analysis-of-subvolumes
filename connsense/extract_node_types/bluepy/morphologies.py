#!/usr/bin/env python3
import numpy as np
import pandas as pd

from bluepy import Cell, Synapse


def groupby_shape(in_circuit):
    """Group biophysical morphologies used in a BBP circuit by their shapes.
    """
    circuit_mtypes = in_circuit.cells.get(properties=Cell.MTYPE)
    morphologies = in_circuit.cells.get(properties=Cell.MORPHOLOGY)
    return (pd.concat([circuit_mtypes.apply(lambda mtype: '_'.join(mtype.split('_')[1:])), morphologies], axis=1)
            .groupby(Cell.MTYPE).apply(lambda mtype_group: list(mtype_group.morphology.unique())))


def measure_morphology(metrics, in_circuit):
    """..."""
    def to_apply(morphology):
        """..."""
        return pd.Series(np.zeros_like(metrics), index=metrics)

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
