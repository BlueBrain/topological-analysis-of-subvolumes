
# Table of Contents

    1.  [A Reproducible Analysis Package to accompagny a publication](#org75665f7)
        1.  [Introduce the configuration](#orgf0d5a1a)
    2.  [A Large Scale Circuit Analysis Environment](#org30362b8)
        1.  [Introduce the configuration](#orgcf75036)
1.  [Pipeline Stages](#orgd7b4f93)
2.  [Configuring the pipeline.](#org8650d4c)
3.  [TAP Environment Command Line Interface](#org19a4725)
4.  [TODO: Checking the pipeline results](#org5af3a3e)

This topic is under-discussion at [JIRA](https://bbpteam.epfl.ch/project/issues/browse/SSCXDIS-530).

The `Topological Analysis Pipeline (TAP)`&rsquo; aims to offer to the scientists:


<a id="org75665f7"></a>

## A Reproducible Analysis Package to accompagny a publication

`TAP` implements a computational pipeline that uses a JSON configuration file that
states what the pipeline&rsquo;s run, once finished, should produce. The pipeline does not
run automatically ton completion, but depends on interaction with the scientists.
This configuration can evolve with time, and be used as a **reproducibility document** for the final
computed results.
This document can be used to load the computed data as well, and used for statistical analyses
and figures.
All the computed data is saved in a `HDF data-store`, and an interface defined within `TAP` can be used
to interact with the `data-store`.


<a id="orgf0d5a1a"></a>

### TODO Introduce the configuration

Meanwhile we can just read the latest configuration at <provide-link>, which contains comments
in each section.


<a id="org30362b8"></a>

## A Large Scale Circuit Analysis Environment

`TAP` automates the parallel launch on a grid of subvolumes.
The parallelization config allows the scientist to provide estimates of their analysis&rsquo;
computational requirements. But a configuration is not the final word. `TAP` can run in a test
workspace that can be used to estimate the computational requirements.

Like any scientific work, a characterization of a circuit is an iterative procedure.
`TAP` aims to provide extensive book-keeping tools to track the progress of such studies.


<a id="orgcf75036"></a>

### TODO Introduce the configuration

Meanwhile we can just read the latest configuration at <provide-link>, which contains comments
in each section.


<a id="orgd7b4f93"></a>

# Pipeline Stages

There are six stages in the pipeline:

-   **define-subtargets:** generate the subvolumes.
    The subvolumes are saved as a `pandas.Series` of lists containing cell `gids`,
    that define `subtargets` in the *circuit* to be analyzed.
    
    Configuration allows for the definition of the circuit-subvolumes to analyze.
    
    Currently hexagonal flatmap columnar circuit sub-volumes have been defined.
    
    Plan to allow the scientists to provide a definition as a plugin.

-   **extract-neurons:** extract the neuron properties for each of the `subtargets`
    Neuron properties are saved as `pandas.DataFrame`, one for eac `subtarget`.
    
    Configuration allows for properties to extract.
    
    Plan to empower the scientist to provide their own extraction method.

-   **evaluate-subtargets:** evaluate the `subtargets` defined so far.
    
    Configuration allows for listing names of pre-defined metrics that are available in `TAP`.
    
    Plan to remove the pre-definition to the accompagnying `connectome-analysis` library.
    The idea is to rid `TAP` of any unnecssary domain-specific knowledge where it is used.
    It shouild apply to any domain with nodes and edges, *i.e.* graphs.
    The resulting configuration will use a mapping of metric names to methods.

-   **extract-connectivity:** extract the connectivity matrix for each of the `subtarget`.
    Adjacency matrices are saved in `scipy.sparse.csr` format and saved inside the HDF store.
    A table of contents is also saved that lists the HDF locations of each \`subtargets\` using
    the `TAP-index`.
    
    Configuration allows for listing the names of connectomes to extract from the circuit,
    using the code provided within `TAP`.
    
    Plan to use  this functionality from within `TAP` as default and empower the scientist to provide
    their method

-   **randomize-connectivity:** randomize the original conectivity of each `subtarget`.
    Randomized adjacency matrices are saved akin to the original one.
    
    Configuration allows for describing the shuffling method as a location of it&rsquo;s source code&#x2026;
    
    **NOTE This stage is optional** and not currently run.

-   **analyze-connectivity:** run analyses for each of the subtargets
    Each original will be analyzed and its data saved to the `HDF-strore` as configured.
    In addition analyses of configured random-controls of each original adjacency can also be run.
    
    Configuration allows for listing the analyses descriptions, including the random-controls to run.


<a id="org8650d4c"></a>

# Configuring the pipeline.

Their are two input configuration files to run the pipeline.
Easier to just open them and look at the comments in there, than repeat that information here.


<a id="org19a4725"></a>

# TAP Environment Command Line Interface

We have developed the \`SSCX-Dissemination Subvolume TAP\` iteratively, and during this process
implemented a prototype environment that aims to provide extensive book keeping.
Installed in your virtual-env, the package will provide all the CLI commands we discuss below.

While the initial steps are simpler (and not used / tested for a while) we will focus on running
analyses. These will work because we have a `TAP-store` that we can use as input providing
neuron properties and original adjacency matrices.

The first step is to create a workspace directory, and copy over the configurations or write
new ones. Since we are going to work with a paritially complete pipeline (with a TAP-HDFstore
complete upto at least adjacencies), we will also need the TAP-HDFstore over to the workspace.
For this we get an allocation from Slurm to work in. The interactive Slurm allocation will allow us
to run the lighter task of setting up the actual computation of data.

    
    tap --configure=config.json --parallelize=parallel.json init

This will create a `run` folder with configurations in it.

While we are exploring the pipeline, and getting our toes wet, we can work in a test mode
\#+being<sub>src</sub> shell

tap &#x2013;configure=config.json &#x2013;parallelize=parallel.json &#x2013;mode=test init

\#+end<sub>src</sub>

Working with a `TAP-store` that already contains the connectivity matrices,
next we go ahead and ask `TAP` to setup a launch of simplex-counts using the config
in which we find,

    "simplex-counts": {
        "source": "/gpfs/bbp.cscs.ch/project/proj83/analyses/topological-analysis-subvolumes/proj83/connectome_analysis/library/topology.py",
        "method": "simplex_counts",
        "controls-to-apply": ["erdos-renyi"],
        "output": "pandas.Series"
    }

On the CLI:

    
    tap --configure=config.json --parallelize=parallel.json init analyze-connectivity simplex-counts

Don&rsquo;t forget to use `-mode=test` if we have decided to explore the pipeline.

This command will initialize a folder where we can setup a directory structure to stage the
parallel computations.

To actually set up

    
    tap --configure=config.json --parallelize=parallel.json run analyze-connectivity simplex-counts

This will prepare the folder

    
    <path-to-workspace> / run / analyze-connectivity / simplex-counts / njobs-<count>

The count of number jobs in `njobs-<count>` will depend on the parallelization configured
for \`simplex-counts\`. Thisn directory will contain sub-directories, one each for a compute-node
where the actual computations will be run, in addition to the batches assigned in a HDF file,
and master `launchscript.sh`.

    
    1364211257 -rw-rw----+  1 sood bbp  18K May 20 11:04 batches.h5
    1364211259 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-0
    1364211261 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-1
    1364211262 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-2
    1364211263 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-3
    1364211264 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-4
    1364211268 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-5
    1364211272 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-6
    1364211276 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-7
    1364211280 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-8
    1364211284 drwxrwx---+  2 sood bbp 4.0K May 20 11:08 compute-node-9
    1364211260 -rwxrw----+  1 sood bbp 3.6K May 20 11:04 launchscript.sh

The next step will be to queue the computations in Slurm. For this we use the master launch script.
We should run launch the computations in a different terminal than the one allocated on Slurm.

    cd run/analyze_connectivity/simplex-counts/njobs-<count>
    chmod u+x launchscript.sh
    ./launchscript.sh

We can track the progress of the computations,

    
    tail -f computate-node-<index> / simplex-counts.err

Once all the runs have finished, we need to test if things went OK.
If the pipeline results are satisfactory, we need to collect the results of each compute node into a
single TAP-HDFstore &#x2014; the one that was specified in the config. We should do the collection of
results in a Slurm allocation, and in the workspace directory.

    
    tap --configure=config.json --parallelize=parallel.json collect analyze-connectivity simplex-counts


<a id="org5af3a3e"></a>

# TODO: Checking the pipeline results

