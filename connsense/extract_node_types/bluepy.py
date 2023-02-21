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
    mtypes = np.sort(inform_mecomboes(circuit).fullmtype.unique())
    return pd.Series(mtypes, dtype=str, name="mtype", index=pd.RangeIndex(0, len(mtypes), 1, name="mtype_id"))

def extract_morphologies_by_morphology(circuit):
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


def extract_morphologies_by_mtype(circuit):
    """...Extract morphologies for each mtype as a list.
    """

    morph_names = np.sort(inform_mecomboes(circuit).morph_name.unique())
    morph_data_type = circuit.config["morphology_type"]
    morph_dirpath = circuit.config["morphologies"]

    def locate_morphology(m):
        """..."""
        return f"{morph_dirpath}/{m}.{morph_data_type}"

    morphology_names = pd.Series(morph_names, dtype=str, name="morphology",
                                 index=pd.RangeIndex(0, len(morph_names), 1, name="morphology_id"))
    paths = morphology_names.apply(locate_morphology).rename("filepath")

    mtypes = extract_mtypes(circuit)
    by_mtype = (inform_mecomboes(circuit)[["fullmtype", "morph_name"]]
                .rename(columns={"fullmtype": "mtype"})
                .drop_duplicates().reset_index(drop=True))

    morph_ids = pd.Series(morphology_names.index.values, index=morphology_names.values)
    mtype_ids = pd.Series(mtypes.index.values, index=mtypes.values)

    morph_subtargets = (pd.DataFrame({"mtype_id": mtype_ids[by_mtype["mtype"].values].values,
                                      "morphology_id": morph_ids.loc[by_mtype["morph_name"].values].values})
                        .groupby("mtype_id").apply(lambda g: list(g["morphology_id"])))

    return {"subtargets": morph_subtargets, "name": morphology_names, "data": paths}


def collect_morphologies(morphologies):
    """Write the extract of morphologies to tap-store HDf.
    """
    return {"name": morphologies["name"], "data": morphologies["data"]}


def extract_etypes(circuit):
    """..."""
    etypes = np.sort(inform_mecomboes(circuit).etype.unique())
    return pd.Series(etypes, dtype=str, name="etype", index=pd.RangeIndex(0, len(etypes), 1, name="etype_id"))


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


def load_models(circuit, morphologies):
    """..."""
    morph_type = circuit.config["morphology_type"]
    morph_db = Path(circuit.config["morphologies"])

    return morphologies.apply(lambda m: morph_db/f"{m}.{morph_type}")
