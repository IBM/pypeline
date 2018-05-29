# #############################################################################
# array.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
`NumPy <http://www.numpy.org/>`_-based objects for carrying datasets.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import pypeline.util.argcheck as chk


class LabeledArray:
    """
    NumPy n-dimensional arrays with additional indexing data attached to each axis.

    Examples
    ---------
    .. testsetup::

       import numpy as np
       import pandas as pd
       from pypeline.util.array import LabeledArray

    .. doctest::

       >>> A = LabeledArray(np.arange(5 * 3 * 4).reshape(5, 3, 4),
       ...                  pd.Index(range(1, 6), name='speed'),
       ...                  pd.Index(range(6, 9), name='force'),
       ...                  pd.MultiIndex.from_arrays([np.arange(4), np.arange(1, 5)],
       ...                                            names=('B', 'C')))

       >>> print(A.data)
      [[[ 0  1  2  3]
        [ 4  5  6  7]
        [ 8  9 10 11]]

       [[12 13 14 15]
        [16 17 18 19]
        [20 21 22 23]]

       [[24 25 26 27]
        [28 29 30 31]
        [32 33 34 35]]

       [[36 37 38 39]
        [40 41 42 43]
        [44 45 46 47]]

       [[48 49 50 51]
        [52 53 54 55]
        [56 57 58 59]]]

       >>> A.index[1]
       RangeIndex(start=6, stop=9, step=1, name='force')
    """

    @chk.check('data', chk.accept_any(chk.is_array_like,
                                      chk.is_instance(sparse.spmatrix)))
    def __init__(self, data, *args):
        """
        Parameters
        ----------
        data : array-like
            (N, M, ...) dataset (any type).
        *args : tuple(:py:class:`~pandas.Index` or :py:class:`~pandas.MultiIndex`)
            Index for each dimension of `data`.
        """
        if chk.is_instance(sparse.spmatrix):
            self._data = data
        else:
            self._data = np.array(data)
            self._data.setflags(write=False)

        sh_data = self._data.shape
        N_dim = len(sh_data)

        if len(args) != N_dim:
            raise ValueError(f'Parameter[data] is {N_dim}-dimensional, but '
                             f'{len(args)} indices were provided.')

        for i, index in enumerate(args):
            if not chk.is_instance(pd.Index):
                raise ValueError(f'The {i}-th element in Parameter[*args] '
                                 f'is not an index.')

            if len(index) != sh_data[i]:
                raise ValueError(f'The {i}-th index in Parameter[*args] has '
                                 f'{len(index)} elements, but '
                                 f'{sh_data[i]} were expected.')

        self._index = args

    @property
    def data(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (N, M, ...) dataset.
        """
        return self._data

    @property
    def index(self):
        """
        Returns
        -------
        tuple(:py:class:`~pandas.Index` or :py:class:`~pandas.MultiIndex`)
            Indexing structures per dimension of `data`.
        """
        return self._index
