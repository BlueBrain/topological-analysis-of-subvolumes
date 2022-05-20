#!/usr/bin/env python3

"""Learn how to use the Topological Analysis Pipeline
"""


import cmd, sys
from argparse import ArgumentParser, RawTextHelpFormatter
import shutil
import os
from pathlib import Path
from pprint import pprint, pformat
from lazy import lazy

from connsense.io import read_config


class TapEnv(cmd.Cmd):
    """A custom shell to run the Topoligcal Analysis Pipeline.
    """
    intro = ("Welcome to the TAP environment:"
             " An interactive environment to use the Topological Analysis Pipeline")
    prompt = "$ tap "

    def log(self, msg):
        """..."""
        self._log.write(msg+"\n")

    def do_tutorial(self, arg):
        """Present a TAP tutorial.
        """
        pprint("To run a TAP action we will need:\n"
               "\t1. A configuration file, config.json, that specifies what computations to run, and\n"
               "\t2. A configuration file, parallel.json, that specifies how to run the computations.")
        pprint("We have preconfigured configurations for the SSCx circuit (2022)")
        pprint("We can create a work space folder with these configurations: ")
        pprint("copy path-to-sscx-2022-configs path-to-tap-workspace")
        pprint("------------------------------------------------------")
        pprint("Current working directory is {Path.cwd()}")

    def help_create(self):
        """Help about creating a project.
        """
        pprint("We need a new directory to work in a TAP environment : `create <path-workspace-dir>`")

    def do_initiate(self, workspace):
        """ a new workspace."""
        pprint(f"Intiialize  a workspace at {workspace}")
        pprint("-------------------------------------------------------------------")
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True, parents=False)
        self._log = open(self.workspace / "tap.log", 'a')
        self.log("Topological Analysis Pipeline")
        self.log("-------------------------------------------------------------------")

        pprint("workspace created")

        pprint("change to the workspace directory")
        os.chdir(self.workspace)
        pprint("-------------------------------------------------------------------")

        self.log(f"initiate {self.workspace}")

    def do_pwd(self,_):
        """..."""
        pprint(self.workspace)

    def do_copy(self, configs):
        """..."""
        self.log(f"copy {configs}")
        pprint("Copy a pipeline configuration, and update it's pipeline root path")
        pprint("-------------------------------------------------------------------")
        pprint(f"copy {configs} ")
        configs = Path(configs)

        pprint(f"Write a config.json from {configs} to {self.workspace}")
        config_json = self.workspace / "config.json"
        if config_json.exists():
            pprint(f"TAP workspace {self.workspace} already contains a config file."
                   "Please remove this file to continue to copy")
            self.log("FAILURE: Workspace already contains config.json")
            self.do_terminate()
        else:
            config = read_config.read(configs/"config.json", raw=True)
            config["paths"]["pipeline"]["root"] = self.workspace.as_posix()
            read_config.write(config, config_json)

        pprint(f"Copy a TAP-store from {configs} to {self.workspace}")
        topsamp_h5 = self.workspace / "topological_sampling.h5"
        if topsamp_h5.exists():
            pprint(f"TAP workspace {self.workspace} already contains a TAP-HDFstore."
                   f"We will continue to use this one, and not copy the one at {configs}")
            self.log("Use the TAP-HDFstore already in the workspace.")
        else:
            shutil.copyfile(configs / "topological_sampling.h5", topsamp_h5)

        pprint(f"Copy paralelization configuration from {configs} to {self.workspace}")
        parallel_json = self.workspace / "parallel.json"
        if parallel_json.exists():
            pprint(f"TAP workspace {self.workspace} already contains a parallel.json."
                   f"We will continue to use this one, and not copy the one at {configs}")
            self.log("Use the parallel.json already in the workspace.")
        else:
            shutil.copyfile(configs / "parallel.json", parallel_json)
        pprint("-------------------------------------------------------------------")


    @staticmethod
    def parse_step(args):
        """..."""
        step, substep = args.split()
        return (step, substep)

    def _do_action(self, a, step, substep):
        """..."""
        from connsense.apps import topological_analysis

        def argued():
            raise NotImplementedError("Cannot to be called.")


        argued.configure = self.workspace / "config.json"
        argued.parallelize = self.workspace / "parallel.json"
        argued.step = step
        argued.substep = substep
        argued.batch = None
        argued.controls = None
        argued.mode = None
        argued.action = a
        argued.output = None

        argued.sample = None
        argued.test = None

        topological_analysis.main(argued)

    def do_init(self, args):
        """..."""
        self.log(f"init {args}")
        step, substep = self.parse_step(args)
        pprint(f"Initialze a run of the Topological Analysis Pipeline:\n"
               f"tap --configure=config.json --parallelize=parallel.json init step substep")
        pprint("-------------------------------------------------------------------")


        self._do_action("init", step, substep)
        pprint("-------------------------------------------------------------------")

    def do_setup(self, args):
        """..."""
        self.log(f"setup {args}")
        step, substep = self.parse_step(args)
        pprint(f"Initialze a run of the Topological Analysis Pipeline:\n\n"
               f"tap --configure=config.json --parallelize=parallel.json run step substep\n\n"
               "This will setup a directory layout for the multiple compute parallel runs\n"
               "Which can then be scheduled on a slurm queue using the launchscripts provided\n"
               "within the compute-layout")
        pprint("-------------------------------------------------------------------")

        self._do_action("run", step, substep)
        pprint("-------------------------------------------------------------------")

    def do_collect(self, args):
        """..."""
        self.log(f"collect {args}")
        step, substep = self.parse_step(args)
        pprint(f"Collect results of the Topological Analysis Pipeline:\n\n"
               f"tap --configure=config.json --parallelize=parallel.json collect step substep\n\n"
               "These results are generated by the parallel run over multiple compute nodes")
        pprint("-------------------------------------------------------------------")

        self._do_action("collect", step, substep)
        pprint("-------------------------------------------------------------------")

    def do_terminate(self, _=None):
        """..."""
        self.log("TERMINATE")
        self.log("-------------------------------------------------------------------")
        pprint("Nice working with you! Thanks for your help\n"
               "Please let us know what you think.")
        self._terminate()
        return True

    def _terminate(self):
        """..."""
        self._log.close()
        sys.exit()


def get_parser():
    """A parser to interpret CLI args...
    """
    parser = ArgumentParser(description="Topological analysis of flatmapped subtargets.",
                            formatter_class=RawTextHelpFormatter)

    return parser

def main(argued=None):
    """..."""
    if not argued:
        parser = get_parser()
        argued = parser.parse_args()

    TapEnv().cmdloop()

if __name__ == "__main__":
    main()
