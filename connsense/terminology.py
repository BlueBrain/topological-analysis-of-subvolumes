#!/usr/bin/env python3

"""Develop a terminology relevant to your analyses --- that `connsense` can parse.
"""

from ..io import logging

LOG = logging.getLogger("connsense terminology")


def define_term(name, description):
    """Define a term.

    TODO: Consider making a class, like HD did for DMT.
    """
    LOG.info("Define a connsense term %s: \n\t", name, description)
    return name
