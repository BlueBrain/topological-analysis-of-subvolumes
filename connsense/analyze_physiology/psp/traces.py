#!/usr/bin/env python3

"""Simulate PSP traces.
"""
import os
from pprint import pformat

import joblib

import bglibpy
bglibpy.set_verbose(2) #i.e. LOG.INFO

from psp_validation.simulation import run_pair_simulation
from psp_validation.utils import isolate

from connsense.io import logging

STEP = "analyze-physiology"
LOG = logging.get_logger(STEP)


def run_simulation(spiking_pre_synaptic, recording_in_post, *, protocol, circuit_config,  ntrials=20, seed=0):
    """..."""
    LOG.info("Run %s trials for connection %s --> %s, using protocol \n%s", ntrials, spiking_pre_synaptic,
             recording_in_post, pformat(protocol))

    LOG.info("Calculate a%d holding current", recording_in_post)
    hold_I, _ = bglibpy.holding_current(protocol["hold_V"], target, circuit_config, enable_ttx=protocol["post_ttx"])
    LOG.info("......%s holding current: %s nA", target, hold_I)
    protocol["hold_I"] = hold_I

    run_isolated = joblib.delayed(isolate(run_pair_simulation))
    parallely = joblib.Parallel(n_jobs=-1, backend="loky")

    results = parallely([run_isolated(circuit_config, spiking_pre_synaptic, recording_in_post, **protocol,
                                      base_seed=seed+trial, add_projections=False, log_level=logging.INFO)
                         for trial in range(ntrials)])

    LOG.info("Run psp simulation over %s --> %s  results %s", spiking_pre_synaptic, recording_in_post, len(results))

    return pd.concat([pd.DataFrame({"time": time, "voltage": voltage,}) for _, time, voltage in results], axis=0,
                     keys=range(trials), names=["trial"])


def record(connections, protocol, *, circuit_config, ntrials):
    """..."""
    def measure_connection(c):
        """..."""
        measurement = (run_simulation(spiking_pre_synaptic=source, recording_in_post=target,
                                    protocol=protocol, circuit_config=circuit_config,
                                    ntrials==(ntrials if ntrials and ntrials > 0 else os.cpu_count()), seed=seed)
                    .reset_index()
                    .assign(source=source, target=target))
        return measurement.set_index(["source", "target"])

    return connections.apply(measure_connection, axis=1)
