# #############################################################################
# array.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Tools and utilities or manipulating arrays.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import pypeline.util.argcheck as chk


class LabeledMatrix:
    """
    2D arrays with additional indexing data attached to each axis.

    Examples
    ---------
    .. testsetup::

       import numpy as np
       import pandas as pd
       from pypeline.util.array import LabeledMatrix

    .. doctest::

       >>> A = LabeledMatrix(np.arange(5 * 3).reshape(5, 3),
       ...                   pd.Index(range(0, 5), name='speed'),
       ...                   pd.MultiIndex.from_arrays([np.arange(0, 3), np.arange(4, 7)],
       ...                                             names=('B', 'C')))

       >>> A.data
       array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11],
              [12, 13, 14]])

       >>> A.index[0]
       RangeIndex(start=0, stop=5, step=1, name='speed')
    """

    @chk.check(dict(data=chk.accept_any(chk.is_array_like,
                                        chk.is_instance(sparse.spmatrix)),
                    row_idx=chk.is_instance(pd.Index),
                    col_idx=chk.is_instance(pd.Index)))
    def __init__(self, data, row_idx, col_idx):
        """
        Parameters
        ----------
        data : array-like
            (N, M) dataset (any type). Sparse CSR/CSC matrices are also accepted.
        row_idx : :py:class:`~pandas.Index`
            Row index.
        col_idx : :py:class:`~pandas.Index`
            Column index.
        """
        if chk.is_instance(sparse.spmatrix)(data):
            self.__data = data

            if not (sparse.isspmatrix_csc(self.__data) or
                    sparse.isspmatrix_csr(self.__data)):
                raise ValueError('Parameter[data] must be CSC/CSR-ordered.')
        else:
            self.__data = np.array(data, copy=False)
            self.__data.setflags(write=False)

            if self.__data.ndim != 2:
                raise ValueError('Parameter[data] must be 2D.')

        N, M = self.__data.shape
        N_row, N_col = len(row_idx), len(col_idx)
        if N_row != N:
            raise ValueError(f'Parameter[row_idx] contains {N_row} entries, '
                             f'but Parameter[data] expected {N}.')
        if N_col != M:
            raise ValueError(f'Parameter[col_idx] contains {N_col} entries, '
                             f'but Parameter[data] expected {M}.')
        self.__index = (row_idx.copy(), col_idx.copy())

    @property
    def data(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`
            (N, M) dataset.
        """
        return self.__data

    @property
    def index(self):
        """
        Returns
        -------
            row_idx : :py:class:`~pandas.Index`
                (N,) row index.

            col_idx : :py:class:`~pandas.Index`
                (M,) column index.
        """
        return self.__index

    @property
    def shape(self):
        """
        Returns
        -------
        tuple(int)
            (N_row, N_col) shape information.
        """
        return self.__data.shape

    def __str__(self):
        return self.__data.__str__()

    def __repr__(self):
        return self.__data.__repr__()

    @chk.check('axes', chk.require_all(chk.has_integers,
                                       chk.has_shape([2, ])))
    def is_consistent_with(self, lmtx, axes):
        """
        Test matrices for consistency along directions.

        Two labeled matrices are considered consistent if their indexes match along the specified dimensions.

        Parameters
        ----------
        lmtx : :py:class:`~pypeline.util.array.LabeledMatrix`
            Matrix to compare with.
        axes : tuple(int)
            (2,) tuple with dimensions along which indices of `self` and `lmtx` must match.

        Returns
        -------
        bool
            True if axes are consistent.
        """
        if not chk.is_instance(LabeledMatrix)(lmtx):
            raise ValueError('Parameter[lmtx] must be a LabeledMatrix.')

        axes = np.array(axes, copy=False)
        if not np.all((axes == 0) | (axes == 1)):
            raise ValueError('Parameter[axes] can only contain {0, 1}.')

        idxA = self.index[axes[0]]
        idxB = lmtx.index[axes[1]]

        if tuple(idxA.names) == tuple(idxB.names):
            if len(idxA) == len(idxB):
                if np.all(idxA == idxB):
                    return True

        return False


@chk.check(dict(x=chk.is_instance(np.ndarray),
                axis=chk.is_integer,
                index_spec=chk.accept_any(chk.is_integer,
                                          chk.is_instance(slice))))
def _index(x, axis, index_spec):
    """
    Form indexing tuple for NumPy arrays.

    Given an array `x`, generates the indexing tuple that has :py:class:`slice` in each axis except `axis`, where `index_spec` is used instead.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Array to index.
    axis : int
        Dimension along which to apply `index_spec`.
    index_spec : slice or int
        Index/slice to use.

    Returns
    -------
    tuple
        indexing tuple.

    Examples
    --------
    .. testsetup::

       from pypeline.util.array import _index

    .. doctest::

       >>> x = np.arange(5 * 4).reshape(5, 4)
       >>> idx = _index(x, 0, 3)
       >>> x[idx] = 0
       >>> print(x)
      [[ 0  1  2  3]
       [ 4  5  6  7]
       [ 8  9 10 11]
       [ 0  0  0  0]
       [16 17 18 19]]

    .. doctest::

       >>> x = np.arange(5 * 4).reshape(5, 4)
       >>> idx = _index(x, 1, slice(2))
       >>> x[idx] = 0
       >>> print(x)
      [[ 0  0  2  3]
       [ 0  0  6  7]
       [ 0  0 10 11]
       [ 0  0 14 15]
       [ 0  0 18 19]]
    """
    if not (-x.ndim <= axis < x.ndim):
        raise ValueError('Parameter[axis] is out-of-bounds.')

    indexer = [slice(None)] * x.ndim
    indexer[axis] = index_spec
    return tuple(indexer)


@chk.check(dict(x=chk.is_array_like,
                idx=chk.has_integers,
                N=chk.is_integer,
                axis=chk.is_integer))
def _cluster_layers(x, idx, N, axis):
    """
    Additive tensor compression along an axis.

    Parameters
    ----------
    x : array-like
        (..., K, ...) array.
    idx : array-like(int)
        (K,) cluster indices.
    N : int
        Total number of levels along compression axis.
    axis : int
        Dimension along which to compress.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (..., N, ...) array
    """
    x = np.array(x, copy=False)
    idx = np.array(idx, copy=False)

    y_shape = list(x.shape)
    y_shape[axis] = N
    y = np.zeros(y_shape, dtype=x.dtype)

    for x_id, y_id in enumerate(idx):
        y[_index(y, axis, y_id)] += x[_index(x, axis, x_id)]
    return y
