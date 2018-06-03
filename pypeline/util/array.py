# #############################################################################
# array.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Array-like objects for carrying datasets.
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
            self.__data = np.array(data)
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
