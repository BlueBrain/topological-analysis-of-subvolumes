#!/usr/bin/env python3
#
from pathlib import Path

import pandas as pd

from bluepy import Cell

def extract_morphologies_by_mtype(circuit):
    """..."""
    mtype_morphologies = circuit.cells.get(properties=[Cell.MTYPE, Cell.MORPHOLOGY]).drop_duplicates()
    morphologies = (mtype_morphologies.groupby(Cell.MTYPE).apply(lambda g: list(g[Cell.MORPHOLOGY]))
                    .rename("morphologies"))
    morphologies.index.rename("subtarget_id", inplace=True)
    return morphologies


def extract_morphologies(circuit):
    """..."""
    morph_type = circuit.config["morphology_type"]
    dirpath = circuit.config["morphologies"]
    def locate_morphology(m):
        """..."""
        return f"{dirpath}/{m}.{morph_type}"

    morphologies = pd.Series(circuit.cells.get(properties=Cell.MORPHOLOGY).unique(), dtype=str).rename("name")
    morphologies.index.name = "morphology_id"
    paths = morphologies.apply(locate_morphology).rename("data")
    return pd.concat([morphologies, paths], axis=1)


def collect_hdf_modeltype(morphologies):
    """Write the extract of morphologies to tap-store HDf."""

    def save(at_path):
        file_h5, group = at_path
        morphologies["name"].to_hdf(file_h5, key=f"{group}/index", format="table")
        morphologies["data"].to_hdf(file_h5, key=f"{group}/morphology_data", format="table")
        return (file_h5, group)

    return save
