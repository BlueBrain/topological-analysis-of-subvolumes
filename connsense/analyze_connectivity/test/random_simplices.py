"""Random simplices to test with"""

import numpy as np
import pandas as pd

def random_simplices_of_dim(d, n=5, N=100):
    """..."""
    return np.random.randint(0, N, (n, d + 2))

def moctoc_index(n=20):
    """..."""
    subtargets = [f"T{t}" for t in range(n)]
    return pd.MultIndex.from_tuples([("Bio_M", t, "local") for t in subtargets],
                                    names=["circuit", "subtarget", "connectome"])

def generate_simplices(adj, nodes=None, max_dim=6, **kwargs):
    """..."""
    return pd.Series([random_simplices_of_dim(d) for d in range(0, max_dim+1)],
                     index=pd.Index(range(0, max_dim+1), name="dim"))
