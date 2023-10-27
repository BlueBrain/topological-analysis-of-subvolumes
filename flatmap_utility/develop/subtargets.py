

# #+RESULTS: attr-wrap
# : #+attr_html: :width 95%
# :

# * Introduction
# We will discuss ~flatmapping~ based on methods developed at BBP, and develop further utilities that help us analyze circuit ~subtargets~ based on it's ~flatmap~ here.
# #+name: fmap-util-init
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-util-init][fmap-util-init]]
import pandas as pd
import numpy as np

import bluepy
import voxcell

import conntility
from conntility.circuit_models.neuron_groups import group_by_grid
from conntility.flatmapping import supersample_flatmap
# fmap-util-init ends here



# #+name: fmap-util
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-util][fmap-util]]

# fmap-util ends here

# * Subtargets
# We define a /geometric/ subtarget as the sub-population of all neurons whose soma are located in an atlas ~subvolume~. Working in the circuit's ~flatspace~, we will place the ~flat-coordinates~ of each voxel in a grid of regular tiles. The grid's ~resolution~ is then the length of tile's sides, and inter-tile distance twice it's value.

# Let us begin by placing the flatmap positions in a grid. We will need a method to get flatmap positions from the circuit.
# #+name: fmap-coords
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-coords][fmap-coords]]
VOXEL_INDICES = ["i", "j", "k"]
FLAT_XY = ["flat_x", "flat_y"]
FLAT_DEPTH = "depth"

def flatmap_coords(circuit, regions):
    """...Get flatmap coordinates of a circuit's regions."""
    pixelated = circuit.atlas.load_data("flatmap")
    orientations = circuit.atlas.load_data("orientation")
    fmap = supersample_flatmap(pixelated, orientations).raw
    fmap_depth = supersample_flatmap(pixelated, orientations, include_depth=True).raw[:, :, :, 1]

    voxels_valid = np.all(fmap >= 0, axis=-1)
    voxels_modeled = mask_volume(circuit, regions)
    voxels_mask = voxels_valid & voxels_modeled

    by_voxel = pd.MultiIndex.from_arrays(np.nonzero(voxels_mask), names=VOXEL_INDICES)
    flat_xy = pd.DataFrame(fmap[voxels_mask], columns=FLAT_XY, index=by_voxel)
    return flat_xy.assign(depth=fmap_depth[voxels_mask])

def mask_volume(circuit, regions):
    """Get volumetric data coverging the circuit's atlas volume that intersects regions."""
    hierarchy = voxcell.RegionMap.load_json(circuit.atlas.fetch_hierarchy())
    region_ids = np.hstack([list(hierarchy.find(r, "acronym", with_descendants=True))
                            for r in regions])
    annotations = circuit.atlas.load_data("brain_regions")
    return np.isin(annotations.raw, region_ids)#.reshape((-1,))
# fmap-coords ends here


# #+RESULTS:
# :RESULTS:
# : /gpfs/bbp.cscs.ch/project/proj83/home/sood/proj83-rsync/Connectome-utilities/conntility/flatmapping/_supersample_utility.py:136: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
# :   res = Rotation.align_vectors(vtgt, vv)
# : Rotation errors: min: 0.0, median: 0.09387602600937707, mean: 0.1362824184485066, std: 0.15664142313770807, max: 2.0
# : Rotation errors: min: 0.0, median: 0.09387602600937707, mean: 0.1362824184485066, std: 0.15664142313770807, max: 2.0
# #+begin_example
#                   flat_x       flat_y        depth
# i   j   k
# 252 248 44    131.326956  6305.991114  1252.640263
#         45    131.268842  6294.558394  1289.649372
#         46    131.210727  6283.125674  1326.658481
#     249 42    133.008187  6354.961759  1088.527667
#         43    159.829737  6334.371329  1252.640263
# ...                  ...          ...          ...
# 388 259 104  6003.844212  3858.573017    -0.000000
# 389 253 113  6059.213031  3345.622218    -0.000000
#     254 110  6004.978052  3495.538957    -0.000000
#     255 109  6047.616610  3581.494568    -0.000000
#     256 110  6055.963382  3562.431813    -0.000000

# [791460 rows x 3 columns]
# #+end_example
# :END:


# To generate a grid for the resulting ~pixel-map~,
# #+name: fmap-subvolumes
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-subvolumes][fmap-subvolumes]]
def distribute_grid(points, resolution, shape="hexagon"):
    """..."""
    assert shape.lower() == "hexagon", "No other implemented!!!"
    voxel_indices = points.index.to_frame().reset_index(drop=True)
    return (group_by_grid(points, FLAT_XY, resolution).reset_index()
            .rename(columns={"grid-i": "grid_i", "grid-j": "grid_j",
                             "grid-x": "grid_x", "grid-y": "grid_y",
                             "grid-subtarget": "subtarget"})
            .astype({"grid_i": int, "grid_j": int})
            .assign(voxel_i=voxel_indices.i.values,
                    voxel_j=voxel_indices.j.values,
                    voxel_k=voxel_indices.k.values)
            .set_index(["voxel_i", "voxel_j", "voxel_k"]))
# fmap-subvolumes ends here


# #+RESULTS:
# #+begin_example
#                          grid_i  grid_j       flat_x       flat_y  \
# voxel_i voxel_j voxel_k
# 252     248     44          -27      27   131.326956  6305.991114
#                 45          -27      27   131.268842  6294.558394
#                 46          -27      27   131.210727  6283.125674
#         249     42          -27      27   133.008187  6354.961759
#                 43          -27      27   159.829737  6334.371329
# ...                         ...     ...          ...          ...
# 388     259     104          -1      32  6003.844212  3858.573017
# 389     253     113           0      30  6059.213031  3345.622218
#         254     110           0      30  6004.978052  3495.538957
#         255     109           0      30  6047.616610  3581.494568
#         256     110           0      30  6055.963382  3562.431813

#                                depth        grid_x  grid_y subtarget
# voxel_i voxel_j voxel_k
# 252     248     44       1252.640263  3.802528e-13  6210.0    R18;C0
#                 45       1289.649372  3.802528e-13  6210.0    R18;C0
#                 46       1326.658481  3.802528e-13  6210.0    R18;C0
#         249     42       1088.527667  3.802528e-13  6210.0    R18;C0
#                 43       1252.640263  3.802528e-13  6210.0    R18;C0
# ...                              ...           ...     ...       ...
# 388     259     104        -0.000000  6.174761e+03  3795.0   R11;C15
# 389     253     113        -0.000000  5.975575e+03  3450.0   R10;C15
#         254     110        -0.000000  5.975575e+03  3450.0   R10;C15
#         255     109        -0.000000  5.975575e+03  3450.0   R10;C15
#         256     110        -0.000000  5.975575e+03  3450.0   R10;C15

# [791460 rows x 8 columns]
# #+end_example

# The ~grid-assignment~ contains a label for the ~subtarget~ each voxel was assigned to. We can generate ~grid-info~ from this assignment,
# #+name: fmap-grid-info
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-grid-info][fmap-grid-info]]
def inform_grid(assignment, resolution, shape="hexagon",
                *, volume_per_voxel):
    """...Extract info about a grid from an assignment of voxels to grid points."""
    grid_ij = ["grid_i", "grid_j"]
    tiles = (assignment.set_index(grid_ij)[["subtarget", "grid_x", "grid_y"]]
             .drop_duplicates())
    depths = assignment.groupby(grid_ij).depth

    voxel_counts = assignment[grid_ij].value_counts()
    return (tiles.reset_index().set_index(grid_ij)
            .assign(subtarget_id=np.arange(len(tiles)) + 1)
            .assign(is_not_boundary=check_boundary(resolution, shape))
            .assign(number_voxels=voxel_counts)
            .assign(has_sufficient_volume=check_volume(resolution, volume_per_voxel))
            .assign(conicality=depths.apply(conicality()))
            .assign(depth=depths.apply(column_depth()))
            .assign(volume=depths.apply(column_volume(volume_per_voxel))))
# fmap-grid-info ends here

# that includes several measurements that can be used to quality check the ~subtargets~.

# #+name: fmap-grid-info-quality-check
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-grid-info-quality-check][fmap-grid-info-quality-check]]
from scipy.spatial import distance as spdist

def check_boundary(resolution, shape="hexagon"):
    """Check boundary of a assignment to a grid of given resolution."""
    n_sides = {"hexagon": 6}[shape.lower()]

    def _check_grid(points):
        distances = spdist.squareform(spdist.pdist(points[["grid_x", "grid_y"]]))
        n_neighbors = ((distances > 0) & (distances <= 2 * resolution)).sum(axis=0)
        return n_neighbors == n_sides

    return _check_grid


def check_volume(resolution, volume_per_voxel, lower_bound=None):
    """..."""
    if lower_bound is None:
        lower_bound = 1000 * np.pi * (resolution ** 2) / 1E9

    def _check_grid(points):
        volume = volume_per_voxel * points.number_voxels
        return volume >= lower_bound

    return _check_grid

def conicality(min_size=2000, bin_size=100):
    """..."""
    def histogram(values):
        bins = np.arange(0, np.max(values) + bin_size, bin_size)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        return bin_centers, np.histogram(values, bins=bins)[0]

    def _measure_voxel_depth(values):
        if np.any(np.isnan(values)): return np.NaN

        depths, n_voxels = histogram(values)
        try:
            slope, offset = np.polyfit(depths[1:-1], np.sqrt(n_voxels)[1:-1], 1)
        except TypeError:
            print("Could not measure conicality for depths: \n", depths)
            return np.NaN
        return slope

    return _measure_voxel_depth

def column_depth(min_size=2000, cutoff_perc=(2, 98)):
    """..."""
    def _measure_voxel_depth(values):
        if np.any(np.isnan(values)): return np.NaN
        return np.percentile(values, cutoff_perc[1]) - np.percentile(values, cutoff_perc[0])

    return _measure_voxel_depth

def column_volume(volume_per_voxel, min_size=2000):
    """..."""
    def _measure_voxel_depth(values):
        if np.any(np.isnan(values)): return np.NaN
        return len(values) * volume_per_voxel

    return _measure_voxel_depth
# fmap-grid-info-quality-check ends here


# We can also produce volumetric data that annotates the voxels by the ~subtarget~ they are in,
# #+name: fmap-subvolume-annotate
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-subvolume-annotate][fmap-subvolume-annotate]]
def annotate_subvolumes(atlas, grid_assignment,  grid_info, raw=False):
    """..."""
    voxels_by_subtarget = grid_assignment.subtarget.reset_index().set_index("subtarget")

    subtarget_ids = grid_info.set_index("subtarget").subtarget_id

    brain_regions = atlas.load_data("brain_regions")
    annotations = np.zeros(brain_regions.shape, dtype=int)
    for subtarget, indices in voxels_by_subtarget.groupby("subtarget"):
        annotations[(indices["voxel_i"].values,
                     indices["voxel_j"].values,
                     indices["voxel_k"].values)] = subtarget_ids[subtarget]
    return (annotations if raw else
            voxcell.VoxelData(annotations, brain_regions.voxel_dimensions,
                              offset=brain_regions.offset))
# fmap-subvolume-annotate ends here


# Having defined ~grid-tiles~ as ~subvolumes~, we can ~populate~ them with the circuit's neurons to define ~subtargets~,
# #+name: fmap-subtargets-distribute
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-subtargets-distribute][fmap-subtargets-distribute]]
from conntility.circuit_models.neuron_groups import load_group_filter

def distribute_subtargets(circuit, subvolumes):
    """..."""
    loader_cfg = {
        "loading": {
            "properties": ["x", "y", "z"],
            "atlas": [
                {"data": subvolumes, "properties": ["subtarget_id"]}
            ]
        }
    }
    neurons = load_group_filter(circuit, loader_cfg).set_index("subtarget_id").gid
    that_were_assigned_to_subtargets = neurons.index > 0
    return neurons[that_were_assigned_to_subtargets].groupby("subtarget_id").apply(list)
# fmap-subtargets-distribute ends here


# We can put our efforts together into a method to generate subtargets from ~connsense~,
# #+name: fmap-subtargets-generate
# #+header: :comments both :padline no :results silent

# [[file:../flatmap.org::fmap-subtargets-generate][fmap-subtargets-generate]]
def generate_subtargets(circuit, regions, grid_resolution, grid_shape="hexagon"):
    """..."""
    brain_regions = circuit.atlas.load_data("brain_regions")
    pixels = flatmap_coords(circuit, regions)
    grid_assignment = distribute_grid(pixels, grid_resolution, grid_shape)
    grid_info = inform_grid(grid_assignment, grid_resolution, grid_shape,
                            volume_per_voxel=brain_regions.voxel_volume/1E9)
    subvolumes = annotate_subvolumes(circuit.atlas, grid_assignment, grid_info)
    subtargets = distribute_subtargets(circuit, subvolumes)

    return (grid_info, subvolumes, subtargets)
# fmap-subtargets-generate ends here
