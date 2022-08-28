"""..."""

class ConfigurationError(ValueError): pass


class NotConfiguredError(FileNotFoundError): pass


PARAMKEY = {"define-subtargets": "definitions",
            "extract-voxels": "annotations",
            "extract-node-types": "modeltypes",
            "extract-edge-types": "models",
            "create-index": "variables",
            "extract-node-populations": "populations",
            "extract-edge-populations": "populations",
            "sample-edge-populations": "analyses",
            "randomize-connectivity": "algorithms",
            "configure-inputs": "analyses",
            "analyze-geometry": "analyses",
            "analyze-node-types": "analyses",
            "analyze-composition": "analyses",
            "analyze-connectivity": "analyses",
            "analyze-physiology": "analyses"}

COMPKEYS = ["description",
            "index", "input", "kwargs",
            "loader", "extractor",  "generator", "computation",
            "output", "collector","reindex"]
