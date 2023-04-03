

# * Long range connectivity between ~flatmap-columns~

# We have studied the /local connectivity/ of ~flatmap-columns~ in the circuit extensively. The /local/ structure is useful towards understanding the /local/ features of physiological activity. To understand how /inter-connectivity/ between ~flatmap-columns~ influences physiology we will have to measure it. With a large number $N=240$ of ~flatmap-columns in the rat-SSCx, saving a ~connectivity-matrix~ (/i.e./ ~adjacency~) for each one of them will use too much space. We may not want to save any of the matrices if their computation is efficient. In any case, we will have to compute them.

# We develop efficient methods to compute all of the $N \times (N-1)$ /cross/ ~adjacencies~ between the ~flatmap-columns~ in ~micasa~. Here, let us notice that we are particularly interested in the ~long-range-cross-connectivity~ of ~flatmap-columns~ that are (heavily) innervated by thalamic fibers. The number of afferent fibers from the thalamus is not uniformaly distributed over the SSCx. While we can delve into characterizing thalamic-innervation, for now we can assume that these ~flatmap-columns~ of /thalamic/ interest are known.

# We can define an extractor for connectivity using ~micasa~,
# #+NAME: extract-connectivity-wm
# #+HEADER: :comments both :padline yes :tangle ./tapestry.py

# [[file:tapestry.org::extract-connectivity-wm][extract-connectivity-wm]]
from micasa.connsense.develop.extract.edge_populations import extract_connectivity

extract_long_range = ExtractorConnectivity(tap.subtarget_gids.loc[100, 0], circuit,
                                           connectome="intra_SSCX_midrange_wm")
# extract-connectivity-wm ends here


# which will extract /intra-flatmap-column/ connectivity in the white-matter connectome if we invoke,
# #+NAME: extract-connectivity-local-wm
# #+HEADER: :comments both :padline yes :tangle ./tapestry.py

# [[file:tapestry.org::extract-connectivity-local-wm][extract-connectivity-local-wm]]
local_wm_100 = extract_long_range()
# extract-connectivity-local-wm ends here


# and to get /long-range-connectivity/ that is between the selected 100th ~flatmap-column~, and another one,
# #+NAME: extract-connectivity-long-range-wm
# #+HEADER: :comments both :padline yes :tangle ./tapestry.py

# [[file:tapestry.org::extract-connectivity-long-range-wm][extract-connectivity-long-range-wm]]
long_range_wm_100_119 = extract_long_range(tap.subtarget_gids.loc[119,0])
# extract-connectivity-long-range-wm ends here



# We may need a subtarget assignment, a method that should be in tap.
# #+NAME: flatmap-column-assignment
# #+HEADER: :comments both :padline yes :tangle ./tapestry.py

# [[file:tapestry.org::flatmap-column-assignment][flatmap-column-assignment]]
def assign_subtargets(tap):
    """..."""
    def series(of_gids):
        return pd.Series(of_gids, name="gid",
                         index=pd.RangeIndex(0, len(of_gids), 1, name="node_id"))
    return (pd.concat([series(gs) for gs in tap.subtarget_gids], axis=0,
                      keys=tap.subtarget_gids.index)
            .reset_index().set_index("gid"))
# flatmap-column-assignment ends here



# * Incoming connections to a simplex

# A simplex is a fully directional one represented as a vector of integer node ids. We compute the simplices in ~connsense-TAP~ to be represented as local ~node-ids~ which we can translate to the ~global-id~ (~gid~) using the ~subtarget~'s ~node-properties~. Then we can look up the ~long-range~ connetome's ~afferent~ gids, map them to the ~flatmap-columns~, and compute a scalar or vector ~weight~ for them. Thus we will have a length ~N~ vector of ~weights~ for each ~simplex~ (of a given dimension) in a given ~flatmap-column~. Over all the columns we have a matrix of weights that can be plotted as a ~heatmap~. We can visualize individual rows or columns over a ~flatmap-grid~.

# We can compute the weights based on filters. Let us develop these ideas further in code.

# #+NAME: gather-simplex-inputs
# #+HEADER: :comments both :padline yes :tangle ./tapestry.py

# [[file:tapestry.org::gather-simplex-inputs][gather-simplex-inputs]]
def gather_inputs(circuit, subtarget, simplex, *, tap):
    """..."""
    gids = tap.
# gather-simplex-inputs ends here
