#!/usr/bin/env python3

"""Configure slurm.
"""

from collections import  OrderedDict
from pathlib import Path
from lazy import lazy

from connsense.io import time


class SlurmConfig:
    """Configure slurm jobs.
    """

    SLURMKEYS = {"name": "job-name"}

    @classmethod
    def _document_field(cls, field, description):
        """..."""
        try:
            doc = cls.__docfield__
        except AttributeError:
            cls.__docfield__ = OrderedDict()
            doc = cls.__docfield__
        doc[field] = description
        return doc

    def _read_field(self, f, description=None, default=None, required=False):
        """..."""
        try:
            value = self._config[f]
        except KeyError:
            if not required:
                return default
            raise KeyError(f"Missing slurm field {f}")

        self._document_field(f, description)
        return value

    def __init__(self, fldict):
        """Initialize from a dict of fields.
        """
        self._config = fldict

        self._name = self._read_field("name", "Name to give to the job.", required=True)

        self._account = self._read_field("account", "Account to bill for the run.", required=True)

        self._executable = self._read_field("executable", "The Python executable to run.", required=True)

        self._modules = self._read_field("modules", "A list of modules to load", [])

        self._env = self._read_field("env", "To run the simulation in, set and unset environment variables.")

        self._venv = self._read_field("venv", "Path to the virtual environment to use.")

        self._output = self._read_field("output", "slurm's stdout", f"{self._name}.out")
        self._error  = self._read_field("error", "slurm's stderr", f"{self._name}.err")

        self._partition = self._read_field("partition", "Slurm partition", "prod")
        self._nodes = self._read_field("nodes", "Number of nodes to allocate",  1)
        self._time = self._read_field("time", "Time to allocate", "24:00:00")
        self._exclusive = self._read_field("exclusive", "Should the allocation be exclusiven to this job",  True)
        self._constraint = self._read_field("constraint", "Constraints for allocation", "cpu")
        self._mem = self._read_field("mem", "to allocate, defaults to all of it",  0)
        self._qos = self._read_field("qos", "for heavier / longer jobs", "normal")
        self._mail = self._read_field("mail", "to mail progress to", None)


    def _keyval(self, attrname):
        """..."""
        if attrname == "exclusive":
            return ("exclusive", None) if self._exclusive else  None

        value = getattr(self, f"_{attrname}")

        if isinstance(value, Path):
            value = value.as_posix()

        return (self.SLURMKEYS.get(attrname, attrname), value)

    @lazy
    def script(self):
        """..."""
        def tag_sbatch(key_value):
            """..."""
            if not key_value:
                return None

            key, value = key_value
            if isinstance(value, str) and '-' in value:
                value = value.replace('-', '__')
            return f"#SBATCH --{key}" + ("" if value is None else f"={value}")

        return [tag_sbatch(self._keyval(attr)) for attr in ["nodes",
                                                             "time",
                                                             "exclusive",
                                                             "constraint",
                                                             "mem",
                                                             "account",
                                                             "partition",
                                                             "name",
                                                             "output",
                                                             "error"]]

    def _source_venv(self):
        """..."""
        if not self._venv:
            return None

        path = Path(self._venv)
        location = path.as_posix() + ("/bin/activate" if path.is_dir() else "")
        return f"source {location}\n"

    def save(self, to_filepath):
        """..."""
        def write_modules(to_file):
            """..."""
            if not self._modules:
                return

            to_file.write("module purge\n")
            for m in self._modules:
                to_file.write(f"module load {m}\n")

        def write_env(to_file):
            """..."""
            if not self._env:
                return

            for var, value in self._env["set"].items():
                to_file.write(f"export {var}={value}\n")

            for var in self._env["unset"]:
                to_file.write(f"unset {var}\n")

        with open(to_filepath, 'w') as f:
            f.write("#!/bin/bash -l\n")
            for line in self.script:
                f.write(line + "\n")

            write_modules(f)
            write_env(f)
            venv = self._source_venv()
            if venv:
                f.write(venv)
            f.write(f'{self._executable} "$@"\n')

        return to_filepath
