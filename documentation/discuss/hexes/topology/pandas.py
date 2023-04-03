
# The ~input~ is set to a label that should appear among ~config-paths~. The ~loader~ method used is expected to take a ~bluepy.Circuit~ instance as an argument, which will be passed by ~connsense~.

# We need to implement the ~read_csv~ method to load the ~hexmap-columns~,

# [[file:../hexes.org::*hexes][hexes:2]]
def read_csv(path):
    """..."""
    import pandas as pd
    subtargets_annotation = pd.read_csv(path)
    subtarget_ids = subtargets_annotation.subtarget_id
    subtargets = (pd.Index(subtarget_ids.unique(), name="subtarget_id").to_series()
                  .apply("Hex{}".format))
    circuit_gidses = subtargets_annotation.groupby("subtarget_id").gid.apply(list)
    return (subtargets, None, circuit_gidses)
# hexes:2 ends here
