

# #+RESULTS:
# : Welcome to EMACS Jupyter

# #+title: Circuit Flatmap

# We develop to understand and analyze a circuit's flatmap.

# * Voxelated flatmap


# We want the voxels for each of the columns in a ~circuit~'s ~flatmap~. In ~connsense-TAP~ we rely on an ~.nrrd~, and a ~.h5~ that provide volumetric annotation of the ~flatmap-columns~ and their info. Here we will develop code that computes this data, and learn from it.
# #+NAME: voxelize-columns
# #+HEADER: :comments both :padline yes :tangle ./flatmap_columns.py

# [[file:flatmap_utility.org::voxelize-columns][voxelize-columns]]
def voxelize_columns(supersample_flatmap, with_depth, that_are_valid, and_annotate):
    """..."""
    import numpy as np
    import pandas as pd
    from conntility.circuit_models.neuron_groups import group_by_grid

    raw_locs = supersample_flatmap.raw.reshape((-1, 2))
    raw_loc_idx = np.nonzero(np.all(raw_locs >= 0, axis=1) & that_are_valid)
    raw_loc_depth = with_depth.raw[:, :, :, 1].flat[raw_loc_idx[0]]

    df = (pd.DataFrame(raw_locs[raw_loc_idx], columns=["ss_flat_x", "ss_flat_y"])
          .assign(depth=raw_loc_depth))

    grid_voxels = group_by_grid(df, ["ss_flat_x", "ss_flat_y"], 230)
    grid_voxels["nrrd-file-flat-index"] = raw_loc_idx[0]

    grid_info = (grid_voxels[["grid-subtarget", "grid-x", "grid-y"]]
                 .drop_duplicates().reset_index())
    grid_info["nrrd-file-id"] = np.arange(len(grid_info)) + 1

    from scipy.spatial import distance

    DD = distance.squareform(distance.pdist(grid_info[["grid-x", "grid-y"]]))
    grid_info["is-not-boundary"] = ((DD > 0) & (DD < 500)).sum(axis=0) == 6

    grid_vol = (grid_voxels.groupby(["grid-subtarget"]).apply(len)
               * and_annotate.voxel_volume / 1E9)
    grid_info["has-sufficient-volume"] = (grid_vol[grid_info["grid-subtarget"]].values
                                          > 1000 * np.pi * (230 ** 2) / 1E9)
    return grid_info
# voxelize-columns ends here
