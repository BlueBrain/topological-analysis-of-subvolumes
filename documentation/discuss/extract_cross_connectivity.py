
# We have experienced extracton of white-matter connectivity to be too slow for interactive development. We can run a Python script and put the results to disc to test them.
# #+header: :comments both :padline yes :tangle ./extract_cross_connectivity.py

# [[file:tapestry.org::*Python script to run slow extractor][Python script to run slow extractor:1]]
"""A little script to extract connectivity between a couple of subrargets."""
from pathlib import Path
import argparse

from connsense.develop.parallelization import HDFStore
from connsense.io import logging

from micasa.connsense.develop.eextract.edge_populations.extract_connectivity\
    import ExtractorConnectivity, FrameEdges

LOG = logging.get_logger("Compute cross connectivity between a pair of subtargerts.")

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
CONNSPACE = PROJSPACE / "home/sood" / "topological-analysis-subvolumes" / "test/v2"

def extract_pair_subtarget(x, y, circuit, connectome, *, savedir):
    extract = ExtractorConnectivity(tap.subtarget_gids.loc[x, 0], circuit, connectome,
                                    Connectivity=FrameEdges)

    sources, edges, targets = extract(tap.subtarget_gids.loc[y, 0])

    src = x; conn = (x, y); trg = y
    conn_h5 = Path(savedir) / "connectivity.h5"
    sources.to_hdf(conn_h5, key=f"subtargets_{x}_{y}/sources")
    edges.to_hdf(conn_h5, key=f"subtargets_{x}_{y}/edges")
    targets.to_hdf(conn_h5, key=f"subtargets_{x}_{y}/targets")
    return conn_h5


def main(args):
    """..."""
    tap = HDFStore(CONNSPACE/"pipeline.yaml")
    circuit = tap.circuit(args.circuit)
    connectome = circuit.connectome if args.connectome == "local" else circuit.projection(args.connectome)
    return extract_pair_subtarget(int(subtarget_x), int(subtarget_y), circuit, connectome
                                  , Path(args.savedir))


if __name__ == "__main__":
    LOG.info("test develop extraction of cross connectivity between flatmap-columns")
    parser = argparse.ArgumentParser(description="Extract cross connectivity")
    parser.add_argument("subtarget_x", help="A subtarget in the pair to measure")
    parser.add_argument("subtarget_y", help="A subtarget in the pair to measure")
    parser.add_argument("circuit", help="Circuit variant name", default="Bio_M")
    parser.add_argument("connectome", help="Circuit connectome", default="intra_SSCX_midrange_wm")
    parser.add_argument("savedir", help="To save output h5")

    main(parser.parse_args()
# Python script to run slow extractor:1 ends here
