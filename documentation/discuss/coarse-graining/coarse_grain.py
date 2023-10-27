

# The method to ~define_subtargets~ will run the computation on a single node as multiple-processes, each deifnition of a tap's subtargets. To expose it to the CLI we will need a script,
# #+name: pipeline-scripts
# #+header: :comments both :padline no :results silent

# [[file:coarse-graining.org::pipeline-scripts][pipeline-scripts]]
from pathlib import Path
import yaml

from connsense.io import logging
from connsense.apps.topological_analysis import get_parser

import pipeline

LOG = logging.get_logger("A fountain of taps")

def main(argued=None):
    if not argued:
        parser = get_parser()
        argued = parser.parse_args()

    LOG.info("Argued coarse-graining: \n%s", argued)

    with open(Path.cwd()/"fountain.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if argued.step == "define-subtargets":
        return pipeline.define_subtargets(config, argued.substep)

    LOG.error("Not Implemented pipeline step %s", argued.step)
    raise ValueError(f"Not Implemented pipeline step {argued.step}")


if __name__ == "__main__":
    LOG.warning("Run coarse-graining taps.")
    parser = get_parser()
    main(parser.parse_args())
# pipeline-scripts ends here
