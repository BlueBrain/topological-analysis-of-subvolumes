#!/usr/bin/env python3

"""Some utilities to write alive / executable documents.
"""
from collections import namedtuple, OrderedDict

import yaml


Location = namedtuple("Location", ["name", "path"])

Content = namedtuple("Content", ["story", "figures", "tables", "subs"],
                     defaults=["", OrderedDict(), OrderedDict(), OrderedDict()])

Section = namedtuple("Section", ["name", "path", "level", "parent", "content"])


class Document:
    """Location in a document...
    that can be used to organize analyses outputs.

    The content of a document is traked with a dictionary.

    A section in a document can be configured as YAML:

    name: String
    level: Int
    path: Path
    content: Mapping[story :: String
    ~                figures :: List[Location]
    ~                tables :: List[Location]
    ~                subs :: List[Location]
    """
    @staticmethod
    def load_yaml(at_path):
        """..."""
        if not at_path.exists():
            return None

        with open(at_path, 'r') as from_file:
            config = yaml.load(from_file, Loader=yaml.FullLoader)
        return config

    @classmethod
    def load_section(cls, at_path):
        """..."""
        descrip = cls.load_yaml(at_path/"section.yaml")
        raise NotImplementedError("How do we determine the parent from a YAML?")

    @classmethod
    def load_content(cls, location, into_content=None):
        """Load content from a location.
        A (sub-)location under this  document's location will be treated as a(sub- section,
        with further nested (sub-)sections listed in file 'sections.yaml'.
        """
        subs = cls.load_yaml(at_path=location/"sections.yaml")
        raise NotImplementedError("Content should be loaded within the section loader.")

    @classmethod
    def init_section(cls, name, path, level, parent):
        """Initialize a section without any content.
        """
        section = Section(name, path, level, parent, Content())
        section.path.mkdir(parents=False, exist_ok=True)

        figures_path = section.path / "figures"
        figures_path.mkdir(parents=False, exist_ok=True)

        tables_path = section.path / "tables"
        tables_path.mkdir(parents=False, exist_ok=True)

        return section


    def __init__(self, name, path, dpi=200):
        """Initialize a document at a path on the disc.
        """
        self._name = name

        self._path = path
        self._path.mkdir(parents=False, exist_ok=True)
        path_figures = self._path / "figures"
        path_figures.mkdir(parents=False, exist_ok=True)
        path_tables = self._path / "tables"
        path_tables.mkdir(parents=False, exist_ok=True)

        self._head = self
        self._dpi = dpi
        self._content = Content()

    @property
    def name(self):
        """..."""
        return self._name

    @property
    def path(self):
        """Location on the disc.
        """
        return self._path

    @property
    def level(self):
        """..."""
        return 0

    @property
    def parent(self):
        """..."""
        return None

    @property
    def content(self):
        """..."""
        return self._content

    def append_section(self, s, at_level):
        """Append a section in the head.
        """
        assert at_level > 0, f"Cannot create a section at negative level {at_level}"

        head = self._head
        if at_level == head.level + 1:
            section = self.init_section(s, head.path / s, head.level + 1, parent=head)
            self._head.content.subs[s] = section
            self._head = section
            return self

        assert at_level <= head.level,(
            f"Cannot append a section {at_level - head.level} levels below "
            f" the current section at level {head.level} \n"
            f"Can append new sections only immediately below the current level, "
            "at the same level, or anywhere above.")

        while at_level <= head.level:
            head = head.parent

        self._head = head
        return self.append_section(s, at_level)

    def append_figure(self, graphic, with_name):
        """..."""
        figure, axes = graphic
        try:
            content_fig = self._head.content.figures[with_name]
        except KeyError:
            content_fig = []

        fpath = self._head.path / "figures"
        figure.savefig(fpath / with_name, dpi=self._dpi)
        content_fig.append(figure)

        self._head.content.figures[with_name] = content_fig
        return self

    def append_table(self, dataframe, with_name):
        """Provide a dataframe without an index."""
        try:
            content_tabs = self._head.content.tables[with_name]
        except KeyError:
            content_tabs = []

        tpath = self._head.path / "tables"
        dataframe.to_csv(tpath / with_name, index=False)
        content_tabs.append(dataframe)

        self._head.content.tables[with_name] = content_tabs
        return self
