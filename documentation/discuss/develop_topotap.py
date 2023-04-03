
# We will study flatmap columns. In the circuit's flatmap we have the following grid,

# #+header: :comments both :file ./figures/flatmap.png :exports both

# [[file:topotap.org::*Introduction][Introduction:1]]
from flatmap_utility import subtargets as fmst, tessellate
flat_xys = fmst.fmap_positions(in_data=circuit)
tritille = tessellate.TriTille(230.0)
graphic_fmap_cells = tritille.plot_hextiles(flat_xys,
                                            annotate=False, with_grid=False, pointmarker=".", pointmarkersize=0.05)
graphic_fmap_cells[0]
# Introduction:1 ends here
