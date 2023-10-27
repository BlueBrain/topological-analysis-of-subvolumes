

# #+RESULTS:
# : Welcome to EMACS Jupyter
# #+title: Computation

# A ~connsense-TAP~ wraps a ~computation-method~ with metadata that it can use to load inputs and run on a cluster, save the results, and provide them on query. The ~HDFStore~ will need values for the ~hdf-group~ and name of the ~dataset~ to store results of a ~computation~. Considering a computation of a quantity that measures a phenomenon, we can define

# A computational study could produce several outputs, analyze them in a ~lab~, the results culminating in a ~portal~, /i.e./ a resource that indexes the result of the study along with the inputs and other metadata.

# Let us develop the concept of ~connsense-computation~  from different points of view.

# We want to help the scientist develop scan analyses over ~subtargets~ of several /circuit-phenomena/, with each ~phenomenon~ described by one or more ~quantity~.

# We should be able to provide the scientist a report describing a ~connsense-config~. Consider the following interaction. The scientist starts to configure a study, by entering the phenomena to study. After entering, they can ask the text to be converted into a full blown config with whatever ~connsense-TAP~ knows at that stage of development. The scientist can then fill up the resulting config file as a form, filling up the missing fields. Let us see what we may need to build such a feature in ~connsense-TAP~, by experimenting, and not necessarily in ~Python~ alone.

# We can describe a ~computation~ in ~Python~,
# #+name: describe-computation-py
# #+header: :comments both :padline yes

# [[file:computation.org::describe-computation-py][describe-computation-py]]
from descripy.config import (field, field_of_type, lazyfield, parameter,
                              section, Struct, Config, NA, MustBeProvidedAtInitialization)

from connsense.io import logging
LOG = logging.get_logger("connsense-Computation")

class Computation(Config):
    """Configure a computation of the measurement of a (phenomenon, quantity)."""

    @parameter
    def phenomenon(self):
        """Phenomenon that will be measured by this computation."""
        raise MustBeProvidedAtInitialization(Computation.phenomenon)

    @parameter
    def quantity(self):
        """Quantity that will be measured by this computation."""
        raise MustBeProvidedAtInitialization(Computation.quantity)

    @lazyfield
    def reference(self):
        """A tuple that names this computation."""
        return (self.phenomenon, self.quantity)

    @parameter
    def inputs(self):
        """Inputs from connsense-TAP that must be loaded and provided to the
        computation method as arguments."""
        raise MustBeProvidedAtInitialization(Computation.inputs)

    @lazyfield
    def args(self):
        """The arguments to the computation."""
        return self.inputs

    @parameter
    def kwargs(self):
        """The keyword arguments to the computation."""
        return NA

    @parameter
    def computation(self):
        """Python source/module and function as a mapping."""
        return {"source": NA, "method": NA, "output": NA}

    @parameter
    def transform(self):
        """A method that can be applied to the results of running self.computation."""
        return NA

    @parameter
    def aggregate(self):
        """A method or string reference to one that can aggregate the result
        of running self.computation."""
        return NA

    @parameter
    def output(self):
        """The final output type of this Computation."""
        raise MustBeProvidedAtInitialization

    def __init__(self, phenomenon, quantity, **kwargs):
        """..."""
        kwargs["phenomenon"] = phenomenon
        kwargs["quantity"] = quantity
        super().__init__(**kwargs)
# describe-computation-py ends here
