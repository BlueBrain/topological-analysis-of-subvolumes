#!/usr/bin/env python3

from bluepy import Cell

def extract_morphologies(circuit):
    """..."""
    mtype_morphologies = circuit.cells.get(properties=[Cell.MTYPE, Cell.MORPHOLOGY]).drop_duplicates()
    return mtype_morphologies.groupby(Cell.MTYPE).apply(lambda g: list(g[Cell.MORPHOLOGY]))
