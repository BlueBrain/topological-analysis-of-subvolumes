"""A little script to extract connectivity between a couple of subrargets."""
from pathlib import Path
import argparse

from connsense.develop.topotap import HDFStore
from connsense.io import logging

from micasa.connsense.develop.extract.edge_populations.extract_connectivity\
    import ExtractorConnectivity, FrameEdges

LOG = logging.get_logger("Compute cross connectivity between a pair of subtargerts.")

ROOTSPACE = Path("/")
PROJSPACE = ROOTSPACE / "gpfs/bbp.cscs.ch/project/proj83"
CONNSPACE = PROJSPACE / "home/sood" / "topological-analysis-subvolumes" / "test/v2"

def extract_pair_subtarget(x, y, circuit, connectome, savedir):
    LOG.info("Extract pair subtargets %s, %s, %s, %s, save in %s", x, y, circuit, connectome, savedir)
    tap = HDFStore(CONNSPACE/"pipeline.yaml")
    circuit = tap.get_circuit(circuit)
    connectome = circuit.connectome if connectome == "local" else circuit.projection(connectome)

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
    LOG.info("Load arguments and call extration of paired subtargets %s, %s", args.subtarget_x, args.subtarget_y)
    return extract_pair_subtarget(int(args.subtarget_x), int(args.subtarget_y),
                                  args.circuit, args.connectome
                                  , Path(args.savedir))


if __name__ == "__main__":
    LOG.info("test develop extraction of cross connectivity between flatmap-columns")
    parser = argparse.ArgumentParser(description="Extract cross connectivity")
    parser.add_argument("subtarget_x", help="A subtarget in the pair to measure")
    parser.add_argument("subtarget_y", help="A subtarget in the pair to measure")
    parser.add_argument("--circuit", help="Circuit variant name", required=False, default="Bio_M")
    parser.add_argument("--connectome", help="Circuit connectome", required=False, default="intra_SSCX_midrange_wm")
    parser.add_argument("--savedir", help="To save output h5", required=False, default=Path.cwd())

    args = parser.parse_args()
    LOG.info("Run extraction of connectivity for \n%s", args)
    main(args)
