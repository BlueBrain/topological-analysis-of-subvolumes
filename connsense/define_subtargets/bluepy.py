#!/usr/bin/env python3

from bluepy import Circuit

def start_target(circuit, name):
    """Get the circuit's start.target ids.
    '"""
    return circuit.cells.ids(name)
