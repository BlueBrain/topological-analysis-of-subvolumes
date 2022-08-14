#!/usr/bin/env python3

from bluepy import Circuit

def start_target(circuit, subtarget):
    """Get the circuit's start.target ids.
    '"""
    return circuit.cells.ids(subtarget)
