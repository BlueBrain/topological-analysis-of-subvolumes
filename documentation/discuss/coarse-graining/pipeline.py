
# ** pipeline
# We will configure 7 ~connsense-pipelines~, for scaling factors of 1/8, 1/4, 1/2, 1, 2, 4, and 8 times a base resolution.
# #+name: pipeline-initialize-taps
# #+header: :comments both :padline no :results silent

# [[file:coarse-graining.org::pipeline-initialize-taps][pipeline-initialize-taps]]
from shutil import copyfile
from copy import deepcopy
from pathlib import Path
import yaml

from multiprocessing import Process, Manager

from connsense.io import read_config, logging
from connsense.io.slurm import SlurmConfig
from connsense.pipeline import workspace
from connsense.pipeline.pipeline import TopologicalAnalysis
from connsense.develop import parallelization as cnsprl

PIPECFG = "pipeline.yaml"
RUNCFG = "runtime.yaml"

LOG = logging.get_logger("A fountain of taps")

def initialize_taps(config={}, update=False):
    """Set up taps for coarse-graining."""
    grid = config.get("parameters", {}).get("grid", {})
    tiling = grid.get("tiliing", "hexagon")
    base_resolution = grid.get("base-resolution", 230.)
    scaling_factors = grid.get("scaling-factors", [1/8, 1/4, 1/2, 1, 2, 4, 8])
    regions = grid.get("regions", [f"S1{r}" for r in
                                   ["DZ", "DZO", "HL", "FL", "J", "Sh", "Tr", "ULp"]])

    paths = config.get("paths", {})
    coarse_graining = Path(paths.get("root") or Path.cwd())
    with open(coarse_graining / PIPECFG, 'r') as f:
        pipeline = yaml.load(f, Loader=yaml.FullLoader)

    def _configure(resolution):
        print("configure resolution", resolution)
        at_resolution = coarse_graining/str(resolution)
        at_resolution.mkdir(exist_ok=update)

        pipeline["paths"]["pipeline"]["root"] = at_resolution.as_posix()

        definitions = pipeline["parameters"]["define-subtargets"]["definitions"]
        kwargs = definitions["flatmap-columns"]["kwargs"]
        kwargs["regions"] = regions
        kwargs["grid_resolution"] = resolution
        kwargs["grid_shape"] = tiling

        with open(at_resolution/PIPECFG, 'w') as f: yaml.dump(pipeline, f)
        copyfile(coarse_graining/RUNCFG, at_resolution/RUNCFG)

        top = TopologicalAnalysis(config=at_resolution/PIPECFG, parallelize=at_resolution/RUNCFG)
        workspace.initialize((top._config, at_resolution/PIPECFG),
                             "define-subtargets", "flatmap-columns", mode="develop",
                             parallelize=(top._parallelize, at_resolution/RUNCFG))
        top.initialize(mode="develop")
        return at_resolution

    resolutions = (s * base_resolution for s in scaling_factors)
    taps = {r: _configure(resolution=r) for r in resolutions}

    config_with_taps = deepcopy(config)
    config_with_taps["taps"] = taps

    with open(coarse_graining / "fountain.yaml", 'w') as to_config_file:
        yaml.dump({"paths": {"root": config_with_taps["paths"]["root"].as_posix()},
                   "parameters": {"grid": {"tiling": tiling,
                                           "base-resolution": base_resolution,
                                           "scaling-factors": scaling_factors,
                                           "regions": regions}},
                   "taps": {tap: location.as_posix() for tap, location in taps.items()}},
                  to_config_file)


    return config_with_taps

def update_taps(config={}):
    """..."""
    return initialize_taps(config, update=True)
# pipeline-initialize-taps ends here

# which we can read with ~YAML~.

# While initialization was easy, we also want to be able to run the same ~tap-command~ for all the resolutions at once, instead of having to issue several from the CLI. That will require a shell script. Instead of devicing another ~connsense~ like tool, we will first solve the problem at hand. What we need are shell commands, one for each ~tap~ (in a ~fountain~), to be invoked at the CLI or written to a shell script that can be run. We can either launch each ~tap~ computation on a single node, or schedule them on the cluster. There are many possibilities, use-cases, /etc/, and we will choose only one. We will produce a ~launchscript~ containing ~sbatch~ submissions. The ~sbatch~ submission will be to a single node. We can ~multiprocess~ from within Python or with a shell script.

# #+name: pipeline-define-subtargets
# #+header: :comments both :padline no :results silent

# [[file:coarse-graining.org::pipeline-define-subtargets][pipeline-define-subtargets]]
def define_subtargets(config, group):
    """Define subtargets for each of the taps in a configuration."""
    from connsense.define_subtargets import run as generate

    def pushd(path): return f"pushd {path}"
    command = f"tap run define-subtargets {group}"
    def popd(): return f"popd\n"

    def install(tap, at_location, *, index, bowl):
        LOG.info("Install (%s-th) tap %s at %s", index, tap, at_location)
        bowl[index] = generate(at_location/PIPECFG, substep=group,
                               parallelize=at_location/RUNCFG, in_mode="develop")
        return bowl[index]

    manager = Manager()
    bowl = manager.dict()
    processes = []

    for i, (tap, at_location) in enumerate(config['taps'].items()):
        p = Process(target=install, args=(tap, Path(at_location)),
                    kwargs={"index": i, "bowl": bowl})
        p.start()
        processes.append(p)

    LOG.info("Launched %s processes", i + 1)

    for p in processes: p.join()

    LOG.info("Parallel computation of define-subtargets results %s", len(bowl))
    return bowl
# pipeline-define-subtargets ends here
