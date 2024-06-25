#!/usr/bin/env python3

from bluepy import Circuit

def start_target(circuit, subtarget):
    """Get the circuit's start.target ids.
    '"""
    return circuit.cells.ids(subtarget)


def snap_target(circuit, subtarget, population):
    """..."""
    cells = circuit.nodes[population]
    return cells.ids(subtarget)

def snap_hypercolumn(circuit, subtarget, population):
    """..."""
    cells = circuit.nodes[population]
    return cells.ids({"hypercolumn": subtarget})
