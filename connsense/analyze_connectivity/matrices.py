"""An indexed HDFstore for matrices of all kinds.
"""
from lazy import lazy
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse

from ..io import logging

STEP = "analyze-connectivity"

LOG = logging.get_logger(STEP+"/matrices")

GB = 2 ** 30


class BeLazy:
    """Load a matrix / dataframe from store, lazily"""
    def __init__(self, mstore, dataset_key):
        """Save the reference to a matrix in the store,
        and load on demand.
        """
        self._store = mstore
        self._path = dataset_key

    def get_value(self):
        """Get value for the data specified lazily in this instance.
        """
        if isinstance(self._path, str):
            return self._store.read(self._path)

        assert pd.isna(self._path),\
            f"Neither a string, nor NA, what is {self._path}: type {type(self._path)}"
        return None

    @lazy
    def value(self):
        """We may not think that a dataframe is a matrix, but it is a value!"""
        return self.get_value()

    @lazy
    def matrix(self):
        """Use this even for a dataframe."""
        return self.value


class MatrixStore:
    """...
    Handle the writing and reading of matrix like data.
    Specialized matrix data such as `scipy.sparse` have their own
    efficient formats that can be used to cache the data on the disc.

    To write the data, provided write method is used to write individual
    matrices to the disc. A table of contents is updated with the address
    of the stored matrix so that it can be read, once again using the
    provided read method.

    `MatrixStore` can be used with matrix data handlers that use a
    HDF group to create the dataset.
    For pandas.DataFrame we will use a subclass.
    """
    keysize = 1

    @classmethod
    def _check_hdf_group(cls, root, group, in_mode):
        """The store needs a HDF group.
        """
        g = group
        def check_hdf(h):
            h.require_group(g)
            return True

        if in_mode=='a' or in_mode=="append":
            with h5py.File(root, 'a') as h:
                return check_hdf(h)
            return False

        if in_mode=='r' or in_mode=="read":
            with h5py.File(root, 'r') as h:
                try:
                    return check_hdf(h)
                except ValueError as error:
                    LOG.error("Does %s/%s exist?: %s", root, group, error)
                    return False
                return False
            return False

        LOG.error("Unnown mode to open a matrix store %s", in_mode)

        return False

    def __init__(self, root, group, *, in_mode='a', using_handler=None,
                 dset_pattern="matrix_{0}",
                 key_toc="toc", key_mat="payload"):
        """..."""
        LOG.info("Initialize a %s matrix store loading / writing data at %s / %s",
                 self.__class__.__name__, root, group)

        if not self._check_hdf_group(root, group, in_mode):
            LOG.error("FAILED CHECK HDF %s.\n"
                      "Data for group %s may not have been computed.", root, group)
            raise ValueError(f"Cannot open HDF {root}/{group} in mode {in_mode}. Does it exist?")

        self._root = root
        self._group = group
        self._using_handler = using_handler
        self._dset_pattern = dset_pattern
        self._key_toc = key_toc
        self._key_mat = key_mat


    @property
    def path(self):
        """..."""
        return (self._root, self._group)

    def but_lazily(self, dataset_at_path):
        """Load dataset at a path lazily."""
        return BeLazy(self, dataset_at_path)

    def prepare_toc(self, of_paths):
        """..."""
        return of_paths

    @property
    def toc(self):
        """"..."""
        key = self.group_identifier_toc
        return pd.read_hdf(self._root, key).apply(self.but_lazily).sort_index()

    @property
    def keys(self):
        """Entries in the data."""
        with h5py.File(self._root, 'a') as hdf:
            hdf_group = hdf.require_group(self.group_identifier_mat)
            keys = list(hdf_group.keys())
        return pd.Series(keys, name="key")

    @property
    def count(self):
        """..."""
        return self.keys.shape[0]

    def next_key(self):
        """..."""
        return self._dset_pattern.format(self.count)

    @property
    def group_identifier_mat(self):
        return f"{self._group}/{self._key_mat}"

    @property
    def group_identifier_toc(self):
        return f"{self._group}/{self._key_toc}"

    def write(self, matrix):
        """Write one matrix."""
        return self._using_handler.write(matrix,
                                         to_hdf_store_at_path=self._root,
                                         under_group=self.group_identifier_mat,
                                         as_dataset=self.next_key())

    def _strip_key_from(self, dataset):
        """..."""
        nested = dataset.split('/')
        key = nested[-self.keysize:]
        group = '/'.join(nested[:-self.keysize])

        if group and group not in (self.group_identifier_mat, self.group_identifier_toc):
            raise ValueError(f"Dataset {dataset} does not exist!"
                             f" Use {self.group_identifier_mat} or {self.group_identifier_toc}")

        return '/'.join(key)

    def read(self, dataset_or_path):
        """Read a dataset
        dataset : Either a single word string key for a dataset under self's group_identifier_mat,
        ~         Or the full path in the HDF store.
        ~         The full path is used to write the TOC which allows it to be,
        ~         used indepedently of this class
        """
        dataset = self._strip_key_from(dataset_or_path)
        return self._using_handler.read(dataset, under_group=self.group_identifier_mat,
                                        in_hdf_store_at_path=self._root)

    def append_toc(self, of_paths):
        """..."""
        #return of_paths.to_hdf(self._root, key=self.group_identifier_toc,
        #                       mode='a', append=True, format="table")
        try:
            toc = self.toc
        except KeyError:
            updated = of_paths
        else:
            current_paths = toc.apply(lambda l: l._path)
            updated = pd.concat([current_paths, of_paths], axis=0)

        return updated.to_hdf(self._root, key=self.group_identifier_toc,
                              mode='a', append=False, format="fixed")

    def dump(self, matrices):
        """..."""
        return self.append_toc(of_paths=matrices.apply(self.write))

    def collect(self, stores):
        """Collect a batch of stores into this one.
        """
        raise NotImplementedError("Collection of a batch of stores into {}"
                                  .format(self.__class__.__name__))


class SparseMatrixHelper:
    """Provide help for scipy sparse matrices."""
    import io
    from scipy import sparse

    @staticmethod
    def write(matrix, to_hdf_store_at_path, under_group, as_dataset):
        """..."""
        bio = io.BytesIO()
        sparse.save_npz(bio, matrix)
        bio.seek(0)
        matrix_bytes = list(bio.read())
        key = under_group + "/" + as_dataset
        with h5py.File(to_hdf_store_at_path, 'a') as hdf:
            hdf_group = hdf[under_group]
            hdf_group.create_dataset(as_dataset, data=matrix_bytes)
        return key

    @staticmethod
    def read(dataset, under_group, in_hdf_store_at_path):
        """..."""
        with h5py.File(in_hdf_store_at_path, 'r') as hdf:
            hdf_group = hdf[under_group]
            data = hdf_group[dataset]
        raw = bytes(data[:].astype(np.uint8))
        bio = io.BytesIO(raw)
        return sparse.load_npz(bio)


class DenseMatrixHelper:
    """Provide help for dense matrices."""
    @staticmethod
    def write(matrix, to_hdf_store_at_path, under_group, as_dataset):
        """..."""
        label = as_dataset
        with h5py.File(to_hdf_store_at_path, 'a') as hdf:
            hdf_group = hdf[under_group]
            hdf_group.create_dataset(label, data=matrix)
        return under_group + '/' + label

    @staticmethod
    def read(dataset, under_group, in_hdf_store_at_path):
        """..."""
        with h5py.File(in_hdf_store_at_path, 'r') as hdf:
            hdf_group = hdf[under_group]
            dset = hdf_group[dataset]
            matrix = np.array(dset)
        return matrix


class DataFrameHelper:
    """..."""
    @staticmethod
    def write(frame, to_hdf_store_at_path, under_group, as_dataset):
        """..."""
        at_path = to_hdf_store_at_path
        under_key = lambda key: '/'.join([under_group, as_dataset, key] if key else [under_group, as_dataset])
        LOG.info("Write DataFrame with shape %s at path %s, under group %s %s: \n%s",
                 frame.shape, at_path, under_group, as_dataset, frame.head())
        #frame.to_hdf(at_path, under_key, mode='a', format="table")

        index = frame.index.to_frame().reset_index(drop=True)
        index.to_hdf(at_path, under_key("index"), format="table")

        columns = frame.reset_index(drop=True)
        columns.to_hdf(at_path, under_key("columns"), format="table")

        return under_key(None)

    @staticmethod
    def read(dataset, under_group, in_hdf_store_at_path):
        """..."""
        at_path = in_hdf_store_at_path
        #under_key = under_group + "/" + dataset
        under_key = lambda key: '/'.join([under_group, dataset, key])
        columns = pd.read_hdf(at_path, under_key("columns"))
        index = pd.read_hdf(at_path, under_key("index"))
        LOG.info("read dataset %s under group %s at path %s", dataset, under_group, in_hdf_store_at_path)
        print(f"read dataset {dataset} under group {under_group} at path {in_hdf_store_at_path}")
        if len(index.columns) == 1:
            return columns.set_index(pd.Index(index[index.columns[0]]))
        return columns.set_index(pd.MultiIndex.from_frame(index))


class SeriesHelper:
    """This could be the same as DataFrameHelper above.
    """
    @staticmethod
    def write(series, to_hdf_store_at_path, under_group, as_dataset):
        """..."""
        LOG.info("SeriesHelper write series %s to %s under %s as dataset %s",
                 series.index, to_hdf_store_at_path, under_group, as_dataset)
        at_path = to_hdf_store_at_path
        under_key = lambda key: '/'.join([under_group, as_dataset, key] if key else [under_group, as_dataset])

        index = series.index.to_frame().reset_index(drop=True)
        index.to_hdf(at_path, under_key("index"), format="table")

        values = series.reset_index(drop=True)
        values.to_hdf(at_path, under_key("values"), format="table")

        return under_key(None)
        #under_key = f"{under_group}/{as_dataset}"
        #series.to_hdf(at_path, under_key, mode='a', format="fixed")
        #return under_key

    @staticmethod
    def read(dataset, under_group, in_hdf_store_at_path):
        """..."""
        at_path = in_hdf_store_at_path
        under_key = lambda key: '/'.join([under_group, dataset, key] if key else [under_group, dataset])
        values = pd.read_hdf(at_path, under_key("values"))
        index = pd.read_hdf(at_path, under_key("index"))
        if len(index.columns) == 1:
            return pd.Series(values.values, index=pd.Index(index[index.columns[0]]), name=values.name)
        return pd.Series(values.values, index=pd.MultiIndex.from_frame(index))

        #under_key = under_group + "/" + dataset
        #return pd.read_hdf(at_path, under_key)


class SeriesOfMatricesHelper:
    """Handle a series that contains matrices in its values."""
    def __init__(self, matrix_helper=DenseMatrixHelper()):
        """..."""
        self._matrix_helper = matrix_helper

    def write(self, series_of_matrices, to_hdf_store_at_path, under_group, as_dataset):
        """..."""
        group_dataset = under_group + '/' + as_dataset

        with h5py.File(to_hdf_store_at_path, 'a') as hdf:
            hdf.create_group(group_dataset)

        index_name = series_of_matrices.index.name or "matrix"

        def write(i, matrix):
            """..."""
            return self._matrix_helper.write(matrix, to_hdf_store_at_path,
                                             under_group=group_dataset,
                                             as_dataset=f"{index_name}-{i}")

        datasets = [write(i, matrix=m) for i, m in series_of_matrices.iteritems()]
        return pd.Series(datasets, index=series_of_matrices.index)

    def read(self, dataset, under_group, in_hdf_store_at_path):
        """..."""
        return self._matrix_helper.read(dataset, under_group, in_hdf_store_at_path)


class SparseMatrixStore(MatrixStore):
    """..."""
    def __init__(self, *args, **kwargs):
        """..."""
        super().__init__(*args, using_handler=SparseMatrixHelper, **kwargs)


class DenseMatrixStore(MatrixStore):
    """..."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=DenseMatrixHelper, **kwargs)


class DataFrameStore(MatrixStore):
    """..."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=DataFrameHelper, **kwargs)

    def collect(self, stores):
        """Collect a bunch of `DataFrameStores` into this one.
        """
        LOG.info("Collect %s batches of stores", len(stores))

        def move(batch, store):
            """..."""
            def write_subtarget(s):
                """..."""
                value = s.get_value()
                mem_usage = value.memory_usage(index=True, deep=True).sum()/GB
                LOG.info("\t...memory used: %s GB", mem_usage)

                written = self.write(value)
                del value
                return written

            current_size = self.count
            LOG.info("Move a %s subtarget batch %s from %s to %s containing %s subtargets.",
                     store.count, batch, Path(store._root).name, Path(self._root).name,
                     current_size)

            saved = store.toc.apply(write_subtarget)
            update = self.prepare_toc(of_paths=saved)

            update_size = update.shape[0]
            self.append_toc(update)
            updated_size = self.toc.shape[0]
            LOG.info("Collect DataFrameStores, append TOC update count from %s by %s to %s",
                     current_size, update_size, updated_size)
            return update

        return {batch: move(batch, store) for batch, store in stores.items()}


class SeriesStore(MatrixStore):
    """..."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=SeriesHelper, **kwargs)

    def collect(self, stores):
        """Collect a bunch of `DataFrameStores` into this one.
        """
        LOG.info("Collect %s batches of stores", len(stores))

        def move(batch, store):
            """..."""
            def write_subtarget(s):
                """..."""
                value = s.get_value()
                mem_usage = value.memory_usage(index=True, deep=True)/GB
                LOG.info("\t...memory used: %s GB", mem_usage)

                written = self.write(value)
                del value
                return written

            current_size = self.count
            LOG.info("Move a %s subtarget batch %s from %s to %s containing %s subtargets.",
                     store.count, batch, Path(store._root).name, Path(self._root).name,
                     current_size)

            saved = store.toc.apply(write_subtarget)
            update = self.prepare_toc(of_paths=saved)

            update_size = update.shape[0]
            self.append_toc(update)
            updated_size = self.toc.shape[0]
            LOG.info("Collect SeriesStores, append TOC update count from %s by %s to %s",
                     current_size, update_size, updated_size)
            return update

        return {batch: move(batch, store) for batch, store in stores.items()}


class SeriesOfMatricesStore(MatrixStore):
    """Store a series of matrices for each subtarget.

    For example, simplex lists.
    """
    keysize = 2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=SeriesOfMatricesHelper(), **kwargs)

    def prepare_toc(self, of_paths):
        """..."""
        LOG.info("Prepare a toc of paths %s", of_paths.shape)

        toc_long = pd.concat([p for _, p in of_paths.iteritems()], axis=0,
                             keys=[d for d, _ in of_paths.iteritems()],
                             names=[of_paths.columns.name])

        LOG.info("...a TOC of paths elongated for storage: %s --> %s: ",
                 of_paths.shape, toc_long.shape)

        names = toc_long.index.names

        toc_long.index = toc_long.index.reorder_levels(names[1:] + [names[0]])
        return toc_long

    def dump(self, content):
        """Expecting content to be a pandas Dataframe of matrices."""
        p = self.prepare_toc(of_paths=content.apply(self.write, axis=1))
        return self.append_toc(of_paths=p)

    def collect(self, stores):
        """Collect a batch of stores into this one.
        """
        LOG.info("Collect %s batches of stores", len(stores))

        def frame(b, batch):
            """A table of contents (TOC) dataframe.
            """
            long = batch.toc
            LOG.info("series of matrices %s --> a dataframe input: %s", b, long.shape)
            colvars = long.index.get_level_values(-1).unique()
            colidxname = long.index.names[-1]
            wide = pd.concat([long.xs(d, level=colidxname) for d in colvars], axis=1,
                             keys=list(colvars), names=[colidxname])
            LOG.info("series of matrices %s --> a dataframe output: %s", b,  wide.shape)
            return wide

        def move(b, batch):
            """..."""
            framed = frame(b, batch)
            i = 0

            def write_row(r):
                nonlocal i
                i += 1
                LOG.info("To collect, write batch %s row %s (%s / %s)",
                         b, r.name, i, len(framed))

                rv = r.apply(lambda l: l.get_value())
                mem_usage = rv.memory_usage(index=True, deep=True)/GB
                LOG.info("\t...memory used: %sGB", mem_usage)

                written = self.write(rv.dropna())
                del rv
                return written

            saved = framed.apply(write_row, axis=1)
            update = self.prepare_toc(of_paths=saved)
            self.append_toc(update)
            return update

        return {b: move(b, batch) for b, batch in stores.items()}


class SeriesOfMatrices:
    """Type to indicate that an analysis algorithm will return
    pandas.Series of matrices.
    """
    pass


class DataFrameOfMatricesStore(MatrixStore):
    """Each element of a dataframe is a matrix ---
    We could use this store for simplex-lists.
    Unlike the SeriesOfMatrices Store, the additional level of simplex dimension when we used
    SeriesOfMatricesStore will become a column instead.
    """
    @property
    def toc(self):
        """Stored matrices are a vector for each subtarget.
        The resulting table of contents a dataframe, not a series.
        """
        key = self.group_identifier_toc
        return pd.read_hdf(self._root, key).applymap(self.but_lazily)


def StoreType(for_matrix_type):
    """..."""
    import scipy, numpy, pandas #base imports to evaluate for_matrix_type

    try:
        matrix_type = eval(for_matrix_type)
    except NameError as name_error:
        raise ValueError(f"Could not evaluate {for_matrix_type}"
                         " matrix-type should either be a `type` or evaulate to one.\n"
                         "Please provide the full path,",
                         " for example `scipy.sparse.csc_matrix`") from name_error
    except TypeError:
        matrix_type = for_matrix_type

    if not isinstance(matrix_type, type):
        raise ValueError("argument for matrix type must be a `type` not {}"
                         .format(type(matrix_type)))

    if issubclass(matrix_type, sparse.base.spmatrix):
        return SparseMatrixStore

    if issubclass(matrix_type, np.ndarray):
        return DenseMatrixStore

    if issubclass(matrix_type, pd.DataFrame):
        return DataFrameStore

    if issubclass(matrix_type, pd.Series):
        return SeriesStore

    if issubclass(matrix_type, SeriesOfMatrices):
        return SeriesOfMatricesStore

    raise TypeError(f"Unhandled matrix value type {for_matrix_type}")


def get_store(to_hdf_at_path, under_group, for_matrix_type, in_mode='a', **kwargs):
    """..."""
    if not for_matrix_type:
        return None

    Store = StoreType(for_matrix_type)

    m = in_mode
    try:
        return Store(to_hdf_at_path, under_group, in_mode=m, **kwargs)
    except ValueError as err:
        LOG.error("Failed to load matrix store: %s", err)
    return None
