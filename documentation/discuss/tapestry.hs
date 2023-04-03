Pandoc
  Meta
    { unMeta =
        fromList
          [ ( "author"
            , MetaInlines [ Str "Vishal" , Space , Str "Sood" ]
            )
          , ( "title"
            , MetaInlines
                [ Str "Connectivity"
                , Space
                , Str "across"
                , Space
                , Str "the"
                , Space
                , Code ( "" , [] , [] ) "flatmap"
                , Str "."
                ]
            )
          ]
    }
  [ RawBlock
      (Format "org")
      "#+property: header-args:jupyter-python :session ~/jupyter-run/active-1-ssh.json"
  , RawBlock
      (Format "org")
      "#+property: header-args:jupyter :session ~/jupyter-run/active-1-ssh.json"
  , RawBlock (Format "org") "#+startup: overview"
  , RawBlock (Format "org") "#+startup: logdrawer"
  , RawBlock (Format "org") "#+startup: hideblocks"
  , Para
      [ Str "Let"
      , Space
      , Str "us"
      , Space
      , Str "setup"
      , Space
      , Str "an"
      , Space
      , Str "interactive"
      , Space
      , Code ( "" , [] , [] ) "Python"
      , Space
      , Str "session"
      , Space
      , Str "where"
      , Space
      , Str "we"
      , Space
      , Str "can"
      , Space
      , Str "run"
      , Space
      , Str "the"
      , Space
      , Str "code"
      , Space
      , Str "developed"
      , Space
      , Str "here."
      ]
  , CodeBlock
      ( "" , [ "jupyter" ] , [] )
      "print(\"Welcome to EMACS Jupyter\")\n"
  , Para
      [ Str "We"
      , Space
      , Str "will"
      , Space
      , Str "characterize"
      , Space
      , Str "the"
      , Space
      , Str "structure"
      , Space
      , Str "of"
      , Space
      , Str "activity"
      , Space
      , Str "across"
      , Space
      , Str "flatmap"
      , Space
      , Str "columns."
      , Space
      , Str "For"
      , Space
      , Str "this"
      , Space
      , Str "we"
      , Space
      , Str "will"
      , Space
      , Str "need"
      , Space
      , Str "to"
      , Space
      , Str "look"
      , Space
      , Str "into"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "long-range"
      , Space
      , Str "connectivity"
      , Space
      , Emph [ Str "between" ]
      , Space
      , Str "pairs"
      , Space
      , Str "of"
      , Space
      , Code ( "" , [] , [] ) "flatmap-columns"
      , Str "."
      ]
  , Header 1 ( "setup" , [] , [] ) [ Str "Setup" ]
  , Para
      [ Str "To"
      , Space
      , Str "get"
      , Space
      , Str "the"
      , Space
      , Str "notebook"
      , Space
      , Str "you"
      , Space
      , Str "will"
      , Space
      , Str "have"
      , Space
      , Str "to"
      , Space
      , Str "clone,"
      ]
  , CodeBlock
      ( "" , [ "shell" ] , [] )
      "git clone https://bbpgitlab.epfl.ch/conn/structural/topological-analysis-of-subvolumes.git\ngit checkout beta\n"
  , Para
      [ Str "To"
      , Space
      , Str "read"
      , Space
      , Str "the"
      , Space
      , Str "setup"
      , Space
      , Str "code,"
      , Space
      , Str "look"
      , Space
      , Str "in"
      , Space
      , Str "the"
      , Space
      , Link ( "" , [] , [] ) [ Str "4" ] ( "#Appendix" , "" )
      , Str "."
      , Space
      , Str "Here"
      , Space
      , Str "we"
      , Space
      , Str "use"
      , Space
      , Code ( "" , [] , [] ) "noweb"
      , Space
      , Str "to"
      , Space
      , Str "include"
      , Space
      , Str "the"
      , Space
      , Str "code"
      , Space
      , Str "written"
      , Space
      , Str "there."
      ]
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "from importlib import reload\nfrom collections.abc import Mapping\nfrom collections import OrderedDict\nfrom pprint import pprint, pformat\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nimport matplotlib\n\nreload(matplotlib)\nfrom matplotlib import pylab as plt\nimport seaborn as sbn\n\nfrom IPython.display import display\n\nfrom bluepy import Synapse, Cell, Circuit\n\nGOLDEN = (1. + np.sqrt(5.))/2.\nprint(\"We will plot golden aspect ratios: \", GOLDEN)\n\nROOTSPACE = Path(\"/\")\nPROJSPACE = ROOTSPACE / \"gpfs/bbp.cscs.ch/project/proj83\"\nSOODSPACE = PROJSPACE / \"home/sood\"\nCONNSPACE = SOODSPACE / \"topological-analysis-subvolumes/test/v2\"\nDEVSPACE  = CONNSPACE / \"test\" / \"develop\"\n\nfrom connsense.develop import topotap as cnstap\ntap = cnstap.HDFStore(CONNSPACE/\"pipeline.yaml\")\ncircuit = tap.get_circuit(\"Bio_M\")\nprint(\"Available analyses: \")\npprint(tap.analyses)\ncircuit\n"
  , Header
      1
      ( "long-range-connectivity-between-flatmap-columns"
      , []
      , []
      )
      [ Str "Long"
      , Space
      , Str "range"
      , Space
      , Str "connectivity"
      , Space
      , Str "between"
      , Space
      , Code ( "" , [] , [] ) "flatmap-columns"
      ]
  , Para
      [ Str "We"
      , Space
      , Str "want"
      , Space
      , Str "to"
      , Space
      , Str "summarize"
      , Space
      , Str "the"
      , Space
      , Emph [ Str "long-range" ]
      , Space
      , Str "connectivity"
      , Space
      , Str "on"
      , Space
      , Str "top"
      , Space
      , Str "of"
      , Space
      , Emph [ Str "local-connectivity" ]
      , Space
      , Str "of"
      , Space
      , Code ( "" , [] , [] ) "flatmap-columns"
      , Str "."
      , Space
      , Str "We"
      , Space
      , Str "can"
      , Space
      , Str "develop"
      , Space
      , Str "a"
      , Space
      , Str "concept"
      , Space
      , Str "of"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "FlatmapColumn"
      , Space
      , Str "as"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "Python"
      , Space
      , Str "class"
      , Space
      , Str "that"
      , Space
      , Str "can"
      , Space
      , Str "provide"
      , Space
      , Str "us"
      , Space
      , Str "with"
      , Space
      , Emph [ Str "long-range" ]
      , Space
      , Str "sources"
      , Space
      , Str "and"
      , Space
      , Str "targets"
      , Space
      , Str "of"
      , Space
      , Str "a"
      , Space
      , Str "group"
      , Space
      , Str "of"
      , Space
      , Str "node"
      , Space
      , Str "ids"
      , Space
      , Str "in"
      , Space
      , Str "another"
      , Space
      , Code ( "" , [] , [] ) "FlatmapColumn"
      , Str ","
      ]
  , RawBlock (Format "org") "#+name: flatmap-column-sources"
  , RawBlock
      (Format "org")
      "#+header: :comments both :padline yes :tangle ./tapestry.py"
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "CONNECTION_ID = [\"source_node\", \"target_node\"]\nSUBTARGET_ID = [\"subtarget_id\", \"circuit_id\"]\nNODE_ID = [\"subtarget_id\", \"circuit_id\", \"node_id\"]\n\ndef sparse_csr(connections):\n    \"\"\"...\"\"\"\n    from scipy import sparse\n    connections_counted = connections.value_counts().rename(\"count\")\n    return sparse.csr_matrix(rows=connections.source_nodes.values,\n                             cols=connections.target_nodes.values)\n"
  , RawBlock
      (Format "org") "#+name: flatmap-column-afferent-edges"
  , RawBlock
      (Format "org")
      "#+header: :comments both :padline yes :tangle ./tapestry.py"
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "\ndef find_afferent(tap, flatmap_column, connectome):\n    \"\"\"...\"\"\"\n    target_gids = tap.nodes.dataset.loc[flatmap_column]().gid.rename(\"target_gid\")\n    target_gids.index.rename(\"target_node\", inplace=True)\n\n    incoming = target_gids.apply(connectome.afferent_gids).rename(\"source_gids\")\n    subtargets = assign_subtargets(tap)\n    sources = incoming.apply(subtargets.reindex)\n\n    return (pd.concat(sources.values, keys=sources.index).fillna(-1).astype(np.int)\n            .droplevel(\"gid\").reset_index().set_index(\"target_node\"))\n"
  , Para [ Str "For" , Space , Str "efferent," ]
  , RawBlock
      (Format "org") "#+name: flatmap-column-efferent-edges"
  , RawBlock
      (Format "org")
      "#+header: :comments both :padline yes :tangle ./tapestry.py"
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def find_efferent(tap, flatmap_column, circuit, connectome):\n    \"\"\"...\"\"\"\n    raise NotImplementedError\n"
  , Para
      [ Str "We"
      , Space
      , Str "may"
      , Space
      , Str "also"
      , Space
      , Str "want"
      , Space
      , Str "a"
      , Space
      , Str "filter"
      , Space
      , Str "of"
      , Space
      , Str "edges,"
      ]
  , RawBlock
      (Format "org") "#+name: flatmap-column-filter-edges-eff-aff"
  , RawBlock
      (Format "org")
      "#+header: :comments both :padline yes :tangle ./tapestry.py"
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def filter_edges(tapestry, flatmap_column, circuit, connectome, direction, and_apply=None):\n    \"\"\"Filter afferent or efferent edges of a flatmap-column in a circuit's connectome.\"\"\"\n    assert direction in (Direction.AFFERENT, Direction.EFFERENT),\\\n        f\"Invalid direction {direction}\"\n\n    affends = (find_afferent(tapestry, flatmap_column, circuit, connectome)\n               .reset_index().groupby(NODE_ID).target_node.apply(list))\n\n    def afferent(nodes):\n        \"\"\"Filter edges incoming from nodes.\"\"\"\n        source_nodes = index_subtarget(nodes)\n        target_nodes = (source_nodes.apply(lambda n: tuple(n.values), axis=1)\n                        .apply(lambda s: affends.loc[s]))\n        return target_nodes if not and_apply else and_apply(target_nodes)\n\n    def efferent(nodes):\n        \"\"\"Filter edges outgoing to nodes.\"\"\"\n        raise NotImplementedError(\"Efferent takes special care.\")\n\n    return afferent if direction == Direction.AFFERENT else efferent\n"
  , Para
      [ Str "We"
      , Space
      , Str "will"
      , Space
      , Str "need"
      , Space
      , Str "a"
      , Space
      , Str "subtarget"
      , Space
      , Str "assignment,"
      , Space
      , Str "a"
      , Space
      , Str "method"
      , Space
      , Str "that"
      , Space
      , Str "should"
      , Space
      , Str "be"
      , Space
      , Str "in"
      , Space
      , Str "tap."
      ]
  , RawBlock
      (Format "org") "#+name: flatmap-column-assignment"
  , RawBlock
      (Format "org")
      "#+header: :comments both :padline yes :tangle ./tapestry.py"
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def assign_subtargets(tap):\n    \"\"\"...\"\"\"\n    def series(of_gids):\n        return pd.Series(of_gids, name=\"gid\",\n                         index=pd.RangeIndex(0, len(of_gids), 1, name=\"node_id\"))\n    return (pd.concat([series(gs) for gs in tap.subtarget_gids], axis=0,\n                      keys=tap.subtarget_gids.index)\n            .reset_index().set_index(\"gid\"))\n"
  , Header 2 ( "simplices" , [] , [] ) [ Str "Simplices" ]
  , Para
      [ Str "A"
      , Space
      , Str "method"
      , Space
      , Str "to"
      , Space
      , Str "get"
      , Space
      , Str "them"
      , Space
      , Str "from"
      , Space
      , Code ( "" , [] , [] ) "topology"
      , Str ","
      ]
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def get_simplices(flatmap_column):\n    subtarget_id, circuit_id = flatmap_column\n    connectome_id = 0\n    adj = tap.adjacency.dataset.loc[subtarget_id, circuit_id, connectome_id]()\n    nodeps = tap.nodes.dataset.loc[subtarget_id, circuit_id]()\n    return pd.concat([topology.list_simplices_by_dimension(adj, nodeps)],\n                     keys=[(subtarget_id, circuit_id)], names=SUBTARGET_ID)\n\n\ndef index_subtarget(tap, flatmap_column, nodes=None):\n    \"\"\"...\"\"\"\n    subtarget_id, circuit_id = flatmap_column\n\n    if nodes is None or (isinstance(nodes, str) and nodes.lower() == \"all\"):\n        nodes = tap.nodes.dataset.loc[subtarget_id, circuit_id].index.values\n\n    return pd.DataFrame({\"subtarget_id\": subtarget_id, \"circuit_id\": circuit_id,\n                         \"node_id\": nodes})\n"
  , Para
      [ Str "We"
      , Space
      , Str "can"
      , Space
      , Str "compute"
      , Space
      , Str "simplex"
      , Space
      , Str "lists"
      , Space
      , Str "in"
      , Space
      , Str "a"
      , Space
      , Str "the"
      , Space
      , Emph [ Str "local-connectome" ]
      , Space
      , Str "of"
      , Space
      , Code ( "" , [] , [] ) "flatmap-columns"
      , Str "."
      , Space
      , Str "We"
      , Space
      , Str "would"
      , Space
      , Str "like"
      , Space
      , Str "to"
      , Space
      , Str "know"
      , Space
      , Str "if"
      , Space
      , Str "there"
      , Space
      , Str "are"
      , Space
      , Code ( "" , [] , [] ) "target-nodes"
      , Space
      , Str "in"
      , Space
      , Str "a"
      , Space
      , Str "given"
      , Space
      , Code ( "" , [] , [] ) "flatmap-column"
      , Space
      , Str "that"
      , Space
      , Str "are"
      , Space
      , Emph [ Str "post-synaptic" ]
      , Space
      , Str "to"
      , Space
      , Str "all"
      , Space
      , Str "the"
      , Space
      , Str "nodes"
      , Space
      , Str "in"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "simplex"
      , Str "."
      , Space
      , Str "We"
      , Space
      , Str "can"
      , Space
      , Str "call"
      , Space
      , Str "the"
      , Space
      , Str "number"
      , Space
      , Str "of"
      , Space
      , Str "simplices"
      , Space
      , Str "that"
      , Space
      , Code ( "" , [] , [] ) "sink"
      , Space
      , Str "at"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "target-node"
      , Space
      , Str "as"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "target-node"
      , Str "'s"
      , Space
      , Code ( "" , [] , [] ) "sink-participation"
      , Str "."
      , Space
      , Str "Analogously"
      , Space
      , Str "we"
      , Space
      , Str "can"
      , Space
      , Str "define"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "source-node"
      , Str "'s"
      , Space
      , Code ( "" , [] , [] ) "source-participation"
      , Space
      , Str "by"
      , Space
      , Str "computing"
      , Space
      , Str "the"
      , Space
      , Str "number"
      , Space
      , Str "of"
      , Space
      , Code ( "" , [] , [] ) "simplices"
      , Space
      , Str "that"
      , Space
      , Code ( "" , [] , [] ) "source"
      , Space
      , Str "at"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "source-node"
      , Str "."
      ]
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def find_sinks(tap, flatmap_column, circuit, connectome, affends=None):\n    \"\"\"Find simplices that sink at each node in a flatmap-column.\"\"\"\n\n    if affends is None:\n        affends = (find_afferent(tap, flatmap_column, circuit, connectome)\n                   .reset_index().groupby(NODE_ID).target_node.apply(list))\n\n    def of_source(flatmap_column, simplex_nodes):\n        sdim = len(simplex_nodes)\n        simplex = index_subtarget(tap, flatmap_column, simplex_nodes)\n        simplex.index.rename(\"spos\", inplace=True)\n        simplex_pos = simplex.reset_index().set_index(NODE_ID)\n\n        target_lists = (pd.concat([simplex_pos, affends.reindex(simplex_pos.index)], axis=1)\n                        .set_index(\"spos\").target_node).sort_index()\n        targets = pd.concat([pd.Series(ns, name=\"target_node\") for ns in target_lists],\n                            keys=target_lists.index).droplevel(None)\n        counts = targets.value_counts()\n        return counts.index[counts == sdim].values\n\n    of_source.afferent_edges = affends\n    return of_source\n"
  , Para
      [ Str "How"
      , Space
      , Str "does"
      , Space
      , Str "a"
      , Space
      , Str "node"
      , Space
      , Str "in"
      , Space
      , Str "a"
      , Space
      , Emph [ Str "target" ]
      , Space
      , Code ( "" , [] , [] ) "flatmap-column"
      , Space
      , Str "connect"
      , Space
      , Str "to"
      , Space
      , Code ( "" , [] , [] ) "simplices"
      , Space
      , Str "in"
      , Space
      , Str "other"
      , Space
      , Code ( "" , [] , [] ) "flatmap-columns"
      , Str "?"
      , SoftBreak
      , Str "How"
      , Space
      , Str "many"
      , Space
      , Emph [ Str "local-connnectome" ]
      , Space
      , Str "simplices"
      , Space
      , Str "in"
      , Space
      , Str "a"
      , Space
      , Str "given"
      , Space
      , Code ( "" , [] , [] ) "flatmap-column"
      , Space
      , Str "does"
      , Space
      , Str "a"
      , Space
      , Str "node"
      , Space
      , Str "connect"
      , Space
      , Str "to?"
      ]
  , Para
      [ Str "What"
      , Space
      , Str "about"
      , Space
      , Str "sources?"
      ]
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def find_sources(tap, flatmap_column, circuit, connectome, effends=None):\n    \"\"\"Find simplices that souce at each node in a flatmap-column.\"\"\"\n\n    if effends is None:\n        effends = (find_efferent(tap, flatmap_column, circuit, connectome)\n                   .reset_index().groupby(NODE_ID).target_node.apply(list))\n\n    def of_source(flatmap_column, simplex_nodes):\n        sdim = len(simplex_nodes)\n        simplex = index_subtarget(tap, flatmap_column, simplex_nodes)\n        simplex.index.rename(\"spos\", inplace=True)\n        simplex_pos = simplex.reset_index().set_index(NODE_ID)\n\n        target_lists = (pd.concat([simplex_pos, affends.reindex(simplex_pos.index)], axis=1)\n                        .set_index(\"spos\").target_node).sort_index()\n        targets = pd.concat([pd.Series(ns, name=\"target_node\") for ns in target_lists],\n                            keys=target_lists.index).droplevel(None)\n        counts = targets.value_counts()\n        return counts.index[counts == sdim].values\n\n    of_source.afferent_edges = affends\n    return of_source\n"
  , Para
      [ Str "We"
      , Space
      , Str "have"
      , Space
      , Str "not"
      , Space
      , Str "implemented"
      , Space
      , Code ( "" , [] , [] ) "find_efferent"
      , Str "."
      , Space
      , Str "We"
      , Space
      , Str "may"
      , Space
      , Str "not"
      , Space
      , Str "need"
      , Space
      , Str "it"
      , Space
      , Str "if"
      , Space
      , Str "we"
      , Space
      , Str "change"
      , Space
      , Str "our"
      , Space
      , Str "approach."
      ]
  , Para
      [ Str "Connectivity"
      , Space
      , Str "is"
      , Space
      , Str "between"
      , Space
      , Str "a"
      , Space
      , Str "group"
      , Space
      , Str "of"
      , Space
      , Str "source"
      , Space
      , Str "nodes"
      , Space
      , Str "and"
      , Space
      , Str "a"
      , Space
      , Str "group"
      , Space
      , Str "of"
      , Space
      , Str "target"
      , Space
      , Str "nodes."
      ]
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def is_subtarget(reference):\n    \"\"\"...\"\"\"\n    ints = (int, np.uint8, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64)\n    return (isinstance(reference, tuple) and len(reference) == 2\n            and isinstance(reference[0], ints) and isinstance(reference[1], ints))\n\n\ndef _resolve_subtarget(tap, reference):\n    \"\"\"...\"\"\"\n    if is_subtarget(reference):\n        return reference\n\n    s, _ = reference\n    if not is_subtarget(reference=s):\n        return None\n\n    return s\n\n\ndef _resolve_nodes(tap, reference, indexed=True):\n    \"\"\"...\"\"\"\n    if is_subtarget(reference):\n        nodes = tap.nodes.dataset.loc[reference].index.values\n        return index_subtarget(tap, reference, nodes) if indexed else nodes\n\n    s, nodes = reference\n    if not is_subtarget(reference=s):\n        return None\n\n    return index_subtarget(tap, s, nodes)\n\n\ndef find_edges(tap, sources=None, targets=None, *, connectome):\n    \"\"\"Find connectome edges from nodes among sources to nodes among targets.\"\"\"\n    source_nodes = _resolve_nodes(sources, indexed=True)\n    target_nodes = _resolve_nodes(targets, indexed=False)\n\n    afferent = (find_afferent(tap, _resolve_subtarget(targets), connectome)\n                .reset_index().groupby(NODE_ID).target_node.apply(list))\n"
  , Header
      1
      ( "incoming-connections-to-a-simplex" , [] , [] )
      [ Str "Incoming"
      , Space
      , Str "connections"
      , Space
      , Str "to"
      , Space
      , Str "a"
      , Space
      , Str "simplex"
      ]
  , Para
      [ Str "A"
      , Space
      , Str "simplex"
      , Space
      , Str "is"
      , Space
      , Str "a"
      , Space
      , Str "fully"
      , Space
      , Str "directional"
      , Space
      , Str "one"
      , Space
      , Str "represented"
      , Space
      , Str "as"
      , Space
      , Str "a"
      , Space
      , Str "vector"
      , Space
      , Str "of"
      , Space
      , Str "integer"
      , Space
      , Str "node"
      , Space
      , Str "ids."
      , Space
      , Str "We"
      , Space
      , Str "compute"
      , Space
      , Str "the"
      , Space
      , Str "simplices"
      , Space
      , Str "in"
      , Space
      , Code ( "" , [] , [] ) "connsense-TAP"
      , Space
      , Str "to"
      , Space
      , Str "be"
      , Space
      , Str "represented"
      , Space
      , Str "as"
      , Space
      , Str "local"
      , Space
      , Code ( "" , [] , [] ) "node-ids"
      , Space
      , Str "which"
      , Space
      , Str "we"
      , Space
      , Str "can"
      , Space
      , Str "translate"
      , Space
      , Str "to"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "global-id"
      , Space
      , Str "("
      , Code ( "" , [] , [] ) "gid"
      , Str ")"
      , Space
      , Str "using"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "subtarget"
      , Str "'s"
      , Space
      , Code ( "" , [] , [] ) "node-properties"
      , Str "."
      , Space
      , Str "Then"
      , Space
      , Str "we"
      , Space
      , Str "can"
      , Space
      , Str "look"
      , Space
      , Str "up"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "long-range"
      , Space
      , Str "connetome's"
      , Space
      , Code ( "" , [] , [] ) "afferent"
      , Space
      , Str "gids,"
      , Space
      , Str "map"
      , Space
      , Str "them"
      , Space
      , Str "to"
      , Space
      , Str "the"
      , Space
      , Code ( "" , [] , [] ) "flatmap-columns"
      , Str ","
      , Space
      , Str "and"
      , Space
      , Str "compute"
      , Space
      , Str "a"
      , Space
      , Str "scalar"
      , Space
      , Str "or"
      , Space
      , Str "vector"
      , Space
      , Code ( "" , [] , [] ) "weight"
      , Space
      , Str "for"
      , Space
      , Str "them."
      , Space
      , Str "Thus"
      , Space
      , Str "we"
      , Space
      , Str "will"
      , Space
      , Str "have"
      , Space
      , Str "a"
      , Space
      , Str "length"
      , Space
      , Code ( "" , [] , [] ) "N"
      , Space
      , Str "vector"
      , Space
      , Str "of"
      , Space
      , Code ( "" , [] , [] ) "weights"
      , Space
      , Str "for"
      , Space
      , Str "each"
      , Space
      , Code ( "" , [] , [] ) "simplex"
      , Space
      , Str "(of"
      , Space
      , Str "a"
      , Space
      , Str "given"
      , Space
      , Str "dimension)"
      , Space
      , Str "in"
      , Space
      , Str "a"
      , Space
      , Str "given"
      , Space
      , Code ( "" , [] , [] ) "flatmap-column"
      , Str "."
      , Space
      , Str "Over"
      , Space
      , Str "all"
      , Space
      , Str "the"
      , Space
      , Str "columns"
      , Space
      , Str "we"
      , Space
      , Str "have"
      , Space
      , Str "a"
      , Space
      , Str "matrix"
      , Space
      , Str "of"
      , Space
      , Str "weights"
      , Space
      , Str "that"
      , Space
      , Str "can"
      , Space
      , Str "be"
      , Space
      , Str "plotted"
      , Space
      , Str "as"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "heatmap"
      , Str "."
      , Space
      , Str "We"
      , Space
      , Str "can"
      , Space
      , Str "visualize"
      , Space
      , Str "individual"
      , Space
      , Str "rows"
      , Space
      , Str "or"
      , Space
      , Str "columns"
      , Space
      , Str "over"
      , Space
      , Str "a"
      , Space
      , Code ( "" , [] , [] ) "flatmap-grid"
      , Str "."
      ]
  , Para
      [ Str "We"
      , Space
      , Str "can"
      , Space
      , Str "compute"
      , Space
      , Str "the"
      , Space
      , Str "weights"
      , Space
      , Str "based"
      , Space
      , Str "on"
      , Space
      , Str "filters."
      , Space
      , Str "Let"
      , Space
      , Str "us"
      , Space
      , Str "develop"
      , Space
      , Str "these"
      , Space
      , Str "ideas"
      , Space
      , Str "further"
      , Space
      , Str "in"
      , Space
      , Str "code."
      ]
  , RawBlock (Format "org") "#+name: gather-simplex-inputs"
  , RawBlock
      (Format "org")
      "#+header: :comments both :padline yes :tangle ./tapestry.py"
  , CodeBlock
      ( "" , [ "python" , "code" ] , [] )
      "def gather_inputs(circuit, subtarget, simplex, *, tap):\n    \"\"\"...\"\"\"\n    gids = tap.\n"
  , Header 1 ( "appendix" , [] , [] ) [ Str "Appendix" ]
  ]
