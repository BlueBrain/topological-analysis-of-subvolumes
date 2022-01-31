"""An indexed HDFstore for matrices of all kinds.
"""
from lazy import lazy

import numpy as np
import pandas as pd
import h5py
from scipy import sparse


class BeLazy:
    """Load a matrix / dataframe from store, lazily"""
    def __init__(self, mstore, dataset_key):
        """Save the reference to a matrix in the store,
        and load on demand.
        """
        self._store = mstore
        self._path = dataset_key

    @lazy
    def matrix(self):
        """Use this even for a dataframe."""
        return self._store.read(self._path)
    @lazy
    def value(self):
        """We may not think that a dataframe is a matrix, but it is a value!"""
        return self._store.read(self._path)


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

    def __init__(self, root, group, using_handler,
                 dset_pattern="matrix_{0}",
                 key_toc="toc", key_mat="payload"):
        """..."""
        self._root = root
        self._group = group
        self._using_handler = using_handler
        self._dset_pattern = dset_pattern
        self._key_toc = key_toc
        self._key_mat = key_mat

        with h5py.File(self._root, 'a') as hdf:
            _= hdf.require_group(self._group)

    def but_lazily(self, dataset_at_path):
        """Load dataset at a path lazily."""
        return BeLazy(self, dataset_at_path)

    @property
    def toc(self):
        """"..."""
        key = self.group_identifier_toc
        return pd.read_hdf(self._root, key).apply(self.but_lazily)

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

        return key

    def read(self, dataset_or_path):
        """Read a dataset
        dataset : Either a single word string key for a dataset under self's group_identifier_mat,
        ~         Or the full path in the HDF store.
        ~         The full path is used to write the TOC which allows it to be,
        ~         used indepedently of this class
        """
        dataset = '/'.join(self._strip_key_from(dataset_or_path))
        return self._using_handler.read(dataset, under_group=self.group_identifier_mat,
                                        in_hdf_store_at_path=self._root)

    def dump(self, matrices):
        """..."""
        toc = matrices.apply(self.write)
        return toc.to_hdf(self._root, key=self.group_identifier_toc,
                          mode='a', format="fixed")



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
        under_key = f"{under_group}/{as_dataset}"
        frame.to_hdf(at_path, under_key)
        return under_key

    @staticmethod
    def read(dataset, under_group, in_hdf_store_at_path):
        """..."""
        at_path = in_hdf_store_at_path
        under_key = under_group + "/" + dataset
        return pd.read_hdf(at_path, under_key)


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
        super().__init(*args, using_handler=SparseMatrixHelper, **kwargs)


class DenseMatrixStore(MatrixStore):
    """..."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=DenseMatrixHelper, **kwargs)


class DataFrameStore(MatrixStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=DataFrameHelper, **kwargs)


class SeriesOfMatricesStore(MatrixStore):
    """..."""
    keysize = 2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, using_handler=SeriesOfMatricesHelper(), **kwargs)

    def dump(self, content):
        """Expecting content to be a pandas Series containing matrices."""
        toc = content.apply(self.write, axis=1)
        toc_long = pd.concat([p for _, p in toc.iteritems()], axis=0,
                             keys=[d for d, _ in toc.iteritems()], names=[content.index.name])
        names = toc_long.index.names
        toc_long.index = toc_long.index.reorder_levels(names[1:] + [names[0]])
        return toc_long.to_hdf(self._root, key=self.group_identifier_toc,
                                mode='a', format="fixed")


class SeriesOfMatrices:
    """Type to indicate that an analysis algorithm will return
    pandas.Series of matrices.
    """
    pass


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

    if issubclass(matrix_type, SeriesOfMatrices):
        return SeriesOfMatricesStore

    raise TypeError(f"Unhandled type for matrix: {type(for_matrix_type)}")


def get_store(to_hdf_at_path, under_group, for_matrix_type, **kwargs):
    """..."""
    if not for_matrix_type:
        return None

    Store = StoreType(for_matrix_type)

    return Store(to_hdf_at_path, under_group, **kwargs)
