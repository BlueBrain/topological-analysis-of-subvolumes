#!/usr/bin/env python3
from lazy import lazy
from pathlib import Path

import numpy as np
import pandas as pd

#from morphio.mut import Morphology
#from neurom.core.morphology import Morphology
import neurom as nm
from neurom.features import morphology, neurite, section, bifurcation as bifork
from neurom.core.dataformat import COLS
from bluepy import Cell, Synapse
ur

from connsense.io import logging

LOG = logging.get_logger("measure morphometrics")

AXONAL = [nm.AXON]
DENDRITIC = [nm.APICAL_DENDRITE, nm.BASAL_DENDRITE]
NEURITES = AXONAL + DENDRITIC
SOMATIC = [nm.SOMA]

def section_mean_position(section):
    """..."""
    points = section.points[:, COLS.XYZ]
    mean_positions = (points[1:] + points[:-1]) / 2.
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)

    radii = section.points[:, COLS.R]
    mean_radii = (radii[1:] + radii[:-1]) / 2.

    volumes = mean_radii * mean_radii * lengths

    return np.dot(volumes, mean_positions) / np.sum(volumes)


def measure_section(s):
    """..."""
    position = section_mean_position(s)
    return pd.Series({"type": s.type,
                      "parent": s.parent.id if s.parent else s.parent,
                      "length": section.section_end_distance(s),
                      "volume": section.section_volume(s),
                      "branch_order": section.branch_order(s),
                      "segments": len(s.points),
                      "radius": section.section_mean_radius(s),
                      "x": position[0], "y": position[1], "z": position[2],
                      "bifurcation_angle": bifork.local_bifurcation_angle(s) if s.is_bifurcation_point() else None})


def measure_neurite(n):
    """..."""
    sections = list(n.iter_sections())
    return (pd.Series(sections, name="section", index=pd.Index([s.id for s in sections], name="section_id"))
            .apply(measure_section))


def measure_morphology(m):
    """..."""
    neurite_ids = pd.RangeIndex(0, len(m.neurites), 1, name="neurite_id")
    return pd.concat([measure_neurite(n) for n in m.neurites], axis=0, keys=neurite_ids)


def measure_one(morphology, loginfo=None):
    """..."""
    LOG.info("Measure morphology data %s (%s)", Path(morphology).stem, loginfo or "")
    m = nm.load_morphology(morphology)
    return measure_morphology(m)


def measure_many(morphologies):
    """..."""
    n = len(morphologies)
    LOG.info("Measure %s morphologies", n)
    return pd.concat([measure_one(morphology=m, loginfo=f"{i}/{n}") for i, m in enumerate(morphologies)], axis=0,
                     keys=morphologies.apply(lambda p: p.stem), names=["morphology"])


def measure(morphologies, morphdb=None):
    """..."""
    n = len(morphologies)
    LOG.info("Measure %s morphologies in %s", n, morphdb)
    return pd.concat([measure_one(morphology=(m if morphdb is None else morphdb.loc[m]),
                                  loginfo=f"{i}/{n}") for i, m in enumerate(morphologies)],
                     keys=morphologies, names=["morphology_id"])
