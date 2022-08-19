#!/usr/bin/env python3

"""Analyze circuit simulations.
"""

def setup(config, substep, subgraphs=None, controls=None, in_mode=None, parallelize=None,
          ouput=None, batch=None, sample=None, tap=None, **kwargs):
    """..."""
    assert parallelize

    from connsense.pipeline.parallelization.parallelization import run_multinode, setup_compute_node
    return run_multinode(setup_compute_node, computation=f"analyze-physiology/{substep}",
                            in_config=config, using_runtime=parallelize)
