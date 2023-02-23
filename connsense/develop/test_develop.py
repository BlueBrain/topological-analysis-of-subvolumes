def to_slice_layers(subtarget):
    """..."""
    return pd.Index(range(7), name="layer").to_series()
def first_slice(with_knife, then_compute):
    def subtarget(t):
        return with_knife(t).apply(then_compute)
    return subtarget


# #+RESULTS: mock-apply-comp

# A single run of the computation is for a ~subtarget~
# #+name: example-slice
# #+header: :comments both :padline no :exports both :tangle ./test_develop.py

# [[file:notes.org::example-slice][example-slice]]
to_slice_layers(1)
# example-slice ends here
def simplex_counts(adj, nmax=None):
    """..."""
    nmax = nmax or adj + 1
    return pd.Series([np.random.randint(10**(nmax - ndim)) for ndim in range(nmax)],
                     name="simplex_count", index=pd.Index(range(nmax), name="ndim"))
display(subtargets.apply(simplex_counts))
lscounts = (subtargets
            .apply(first_slice(with_knife=to_slice_layers, then_compute=simplex_counts)))

print("simplex counts of all subtargets is a ", type(lscounts))
lscounts.iloc[0]
