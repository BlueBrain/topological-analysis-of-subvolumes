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
            "randomize-connectivity": "algorithms",
            "analyze-geometry": "analyses",
            "analyze-node-types": "analyses",
            "analyze-composition": "analyses",
            "analyze-connectivity": "analyses",
            "analyze-physiology": "analyses"}
