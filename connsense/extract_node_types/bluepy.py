#!/usr/bin/env python3
#
from pathlib import Path

import numpy as np
import pandas as pd

from bluepy import Cell

def inform_mecomboes(circuit):
    """..."""
    try:
        p = circuit.config["mecombo_info"]
    except KeyError as kerr:
        raise KeyError("MISSING mecombo_info used for the circuit ") from kerr
    else:
        path = Path(p)

    return pd.read_csv(path, sep='\t')


def extract_mtypes(circuit):
    """..."""
    mecomboes = inform_mecomboes(circuit)
    return pd.Series(np.sort(mecomboes.fullmtype.unique()), dtype=str, name="mtype")


def extract_morphologies(circuit):
    """..."""
    morph_names = np.sort(inform_mecomboes(circuit).morph_name.unique())
    morph_data_type = circuit.config["morphology_type"]
    morph_dirpath = circuit.config["morphologies"]

    def locate_morphology(m):
        """..."""
        return f"{morph_dirpath}/{m}.{morph_data_type}"

    morphologies = pd.Series(morph_names, dtype=str, name="morphology")
    paths = morphologies.apply(locate_morphology).rename("filepath")
    return {"name": morphologies, "data": paths}


def collect_morphologies(morphologies):
    """Write the extract of morphologies to tap-store HDf.
    """
    return {"name": morphologies["name"], "data": morphologies["data"]}


def extract_etypes(circuit):
    """..."""
    mecomboes = inform_mecomboes(circuit)
    return pd.Series(np.sort(mecomboes.etype.unique()), dtype=str, name="etype")


def extract_electrophysiologies(circuit):
    """..."""
    ephys_names = np.sort(inform_mecomboes(circuit).emodel.unique())
    ephys_data_type = "hoc"
    ephys_dirpath = circuit.config["emodels"]

    def locate_ephys(p):
        """..."""
        return f"{ephys_dirpath}/{p}.{ephys_data_type}"

    electrophysiologies = pd.Series(ephys_names, dtype=str, name="emodel")
    paths = electrophysiologies.apply(locate_ephys).rename("data")
    return {"name": electrophysiologies, "data": paths}
