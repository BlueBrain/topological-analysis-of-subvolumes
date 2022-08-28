#!/usr/bin/env python3

"""IO for PSP data.
"""

import pandas as pd

def read_protocols(pathways, filepath, default=True):
    """..."""
    protocols = pd.read_csv(filepath, index_col=[0,1])
    defval = protocols.iloc[0] if default else None
    return protocols.reindex(pd.MultiIndex.from_frame(pathways)).fillna(defval)
