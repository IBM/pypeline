# ##############################################################################
# fourier.py
# ==========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# ##############################################################################

"""
FFT-based tools.
"""

import cmath

import numpy as np
import scipy.fftpack as fftpack

import pypeline.util.argcheck as chk


@chk.check(dict(T=chk.is_real,
                N_FS=chk.is_odd,
                T_c=chk.is_real,
                N_s=chk.is_integer))
def ffs_sample(T, N_FS, T_c, N_s):
    r"""
    Signal sample positions for :py:func:`~pypeline.util.math.fourier.ffs`.

    Return the coordinates at which a signal must be sampled to use
    :py:func:`~pypeline.util.math.fourier.ffs`.

    Parameters
    ----------
    T : float
        Function period.
    N_FS : int
        Function bandwidth.
    T_c : float
        Period mid-point.
    N_s : int
        Number of samples.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (N_s,) coordinates at which to sample a signal (in the right order).

    Examples
    --------
    Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a bandlimited periodic
    function of period :math:`T = 1`, bandwidth :math:`N_{FS} = 5`, and
    with one period centered at :math:`T_{c} = \pi`.
    The sampling points :math:`t[n] \in \mathbb{R}` at which :math:`\phi`
    must be evaluated to compute the Fourier Series coefficients
    :math:`\left\{ \phi_{k}^{FS}, k = -2, \ldots, 2 \right\}` with
    :py:func:`~pypeline.util.math.fourier.ffs` are obtained as follows:

    .. testsetup::

       import numpy as np
       from pypeline.util.math.fourier import ffs_sample

    .. doctest::

       # Ideally choose N_s to be highly-composite for ffs().
       >>> smpl_pts = ffs_sample(T=1, N_FS=5, T_c=np.pi, N_s=8)
       >>> np.around(smpl_pts, 2)  # Notice points are not sorted.
       array([3.2 , 3.33, 3.45, 3.58, 2.7 , 2.83, 2.95, 3.08])

    See Also
    --------
    :py:func:`~pypeline.util.math.fourier.ffs`
    """
    if T <= 0:
        raise ValueError('Parameter[T] must be positive.')
    if N_FS < 3:
        raise ValueError('Parameter[N_FS] must be at least 3.')
    if N_s < N_FS:
        raise ValueError('Parameter[N_s] must be greater or equal to the '
                         'signal bandwidth.')

    if chk.is_odd(N_s):
        M = (N_s - 1) // 2
        idx = np.r_[0:M + 1, -M:0]
        sample_points = T_c + (T / (2 * M + 1)) * idx
    else:  # Even case
        M = N_s // 2
        idx = np.r_[0:M, -M:0]
        sample_points = T_c + (T / (2 * M)) * (0.5 + idx)

    return sample_points


@chk.check(dict(x=chk.accept_any(chk.has_reals, chk.has_complex),
                T=chk.is_real,
                T_c=chk.is_real,
                N_FS=chk.is_odd,
                axis=chk.is_integer))
def ffs(x, T, T_c, N_FS, axis=-1):
    r"""
    Fourier Series coefficients from signal samples.

    Parameters
    ----------
    x : array-like(complex)
        (..., N_s, ...) function values at sampling points specified by
        :py:func:`~pypeline.util.math.fourier.ffs_sample`.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.
    axis : int
        Dimension of `x` along which function samples are stored.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (..., N_s, ...) vectors containing entries
        :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right]`
        :math:`\in \mathbb{C}^{N_{s}}`.

    Examples
    --------
    Let :math:`\phi(t)` be a shifted Dirichlet kernel of period :math:`T` and
    bandwidth :math:`N_{FS} = 2 N + 1`:

    .. math::

       \phi(t) = \sum_{k = -N}^{N} \exp\left(
                                   j \frac{2 \pi}{T} k (t - T_{c}) \right)
               = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}
                      {\sin\left( \pi [t - T_{c}] / T \right)}.

    It's Fourier Series (FS) coefficients :math:`\phi_{k}^{FS}` can be
    analytically evaluated using the shift-modulation theorem:

    .. math::

       \phi_{k}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
           0 & \text{otherwise}.
       \end{cases}

    Being bandlimited, we can use :py:func:`~pypeline.util.math.fourier.ffs` to
    numerically evaluate :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`:

    .. testsetup::

       import numpy as np
       import math
       from pypeline.util.math.fourier import ffs_sample, ffs

       def dirichlet(x, T, T_c, N_FS):
           y = x - T_c

           n, d = np.zeros((2, len(x)))
           nan_mask = np.isclose(np.fmod(y, np.pi), 0)
           n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
           d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
           n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
           d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

           return n / d

    .. doctest::

       >>> T, T_c, N_FS = math.pi, math.e, 15
       >>> N_samples = 16  # Any >= N_FS will do, but highly-composite best.

       # Sample the kernel and do the transform.
       >>> sample_points = ffs_sample(T, N_FS, T_c, N_samples)
       >>> diric_samples = dirichlet(sample_points, T, T_c, N_FS)
       >>> diric_FS = ffs(diric_samples, T, T_c, N_FS)

       # Compare with theoretical result.
       >>> N = (N_FS - 1) // 2
       >>> diric_FS_exact = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])

       >>> np.allclose(diric_FS[:N_FS], diric_FS_exact)
       True

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pypeline.util.math.fourier.ffs_sample`,
    :py:func:`~pypeline.util.math.fourier.iffs`
    """
    x = np.array(x, copy=False)
    N_s = x.shape[axis]

    if T <= 0:
        raise ValueError('Parameter[T] must be positive.')
    if not (3 <= N_FS <= N_s):
        raise ValueError(f'Parameter[N_FS] must lie in {{3, ..., {N_s}}}.')
    if not (-x.ndim <= axis < x.ndim):
        raise ValueError('Parameter[axis] is out-of-bounds.')

    M, N = np.r_[N_s, N_FS] // 2
    E_1 = np.r_[-N:(N + 1), np.zeros(N_s - N_FS, dtype=int)]
    B_2 = np.exp(-1j * 2 * np.pi / N_s)
    if chk.is_odd(N_s):
        B_1 = np.exp(1j * (2 * np.pi / T) * T_c)
        E_2 = np.r_[0:(M + 1), -M:0]
    else:
        B_1 = np.exp(1j * (2 * np.pi / T) * (T_c + T / (2 * N_s)))
        E_2 = np.r_[0:M, -M:0]

    sh = [1] * x.ndim
    sh[axis] = N_s
    C_1 = np.reshape(B_1 ** (- E_1), sh)
    C_2 = np.reshape(B_2 ** (- N * E_2), sh)
    X_FS = fftpack.fft(x * C_2, axis=axis) * (C_1 / N_s)
    return X_FS


@chk.check(dict(x_FS=chk.accept_any(chk.has_reals, chk.has_complex),
                T=chk.is_real,
                T_c=chk.is_real,
                N_FS=chk.is_odd,
                axis=chk.is_integer))
def iffs(x_FS, T, T_c, N_FS, axis=-1):
    r"""
    Signal samples from Fourier Series coefficients.

    :py:func:`~pypeline.util.math.fourier.iffs` is basically the inverse of
    :py:func:`~pypeline.util.math.fourier.ffs`.

    Parameters
    ----------
    x_FS : array-like(complex)
        (..., N_s, ...) FS coefficients in the order
        :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right]`
        :math:`\in \mathbb{C}^{N_{s}}`.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.
    axis : int
        Dimension of `x_FS` along which FS coefficients are stored.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (..., N_s, ...) vectors containing original function samples given to
        :py:func:`~pypeline.util.math.fourier.ffs`.

        In short: :math:`(\text{iFFS} \circ \text{FFS})\{ X \} = X`.

    Notes
    -----
    Theory: :ref:`FFS_def`.

    See Also
    --------
    :py:func:`~pypeline.util.math.fourier.ffs_sample`,
    :py:func:`~pypeline.util.math.fourier.ffs`
    """
    x_FS = np.array(x_FS, copy=False)
    N_s = x_FS.shape[axis]

    if T <= 0:
        raise ValueError('Parameter[T] must be positive.')
    if not (3 <= N_FS <= N_s):
        raise ValueError(f'Parameter[N_FS] must lie in {{3, ..., {N_s}}}.')
    if not (-x_FS.ndim <= axis < x_FS.ndim):
        raise ValueError('Parameter[axis] is out-of-bounds.')

    M, N = np.r_[N_s, N_FS] // 2
    E_1 = np.r_[-N:(N + 1), np.zeros(N_s - N_FS, dtype=int)]
    B_2 = np.exp(-1j * 2 * np.pi / N_s)
    if chk.is_odd(N_s):
        B_1 = np.exp(1j * (2 * np.pi / T) * T_c)
        E_2 = np.r_[0:(M + 1), -M:0]
    else:
        B_1 = np.exp(1j * (2 * np.pi / T) * (T_c + T / (2 * N_s)))
        E_2 = np.r_[0:M, -M:0]

    sh = [1] * x_FS.ndim
    sh[axis] = N_s
    C_1 = np.reshape(B_1 ** (E_1), sh)
    C_2 = np.reshape(B_2 ** (N * E_2), sh)
    X = fftpack.ifft(x_FS * C_1, axis=axis) * (C_2 * N_s)
    return X


@chk.check(dict(x=chk.is_instance(np.ndarray),
                axis=chk.is_integer,
                index_spec=chk.accept_any(chk.is_integer,
                                          chk.is_instance(slice))))
def _index(x, axis, index_spec):
    """
    Form indexing tuple for NumPy arrays.

    Given an array `x`, generates the indexing tuple that has :py:class:`slice`
    in each axis except `axis`, where `index_spec` is used instead.

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

       from pypeline.util.math.fourier import _index

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


@chk.check(dict(x=chk.accept_any(chk.has_reals, chk.has_complex),
                A=chk.accept_any(chk.is_real, chk.is_complex),
                W=chk.accept_any(chk.is_real, chk.is_complex),
                M=chk.is_integer,
                axis=chk.is_integer))
def czt(x, A, W, M, axis=-1):
    """
    Chirp Z-Transform.

    This implementation follows the semantics defined in :ref:`CZT_def`.

    Parameters
    ----------
    x : array-like(float or complex)
        (..., N, ...) input array.
    A : complex
        Circular offset from the positive real-axis.
    W : complex
        Circular spacing between transform points.
    M : int
        Length of the transform.
    axis : int
        Dimension of `x` along which the samples are stored.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (..., M, ...) transformed input along the axis indicated by `axis`.

    Notes
    -----
    Due to numerical instability when using large `M`, this implementation only
    supports transforms where `A` and `W` have unit norm.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.fourier import czt

    Implementation of the DFT:

    .. doctest::

       >>> N = M = 10
       >>> x = np.random.randn(N, 3) + 1j * np.random.randn(N, 3)  # multi-dim

       >>> dft_x = np.fft.fft(x, axis=0)
       >>> czt_x = czt(x, A=1, W=np.exp(-1j * 2 * np.pi / N), M=M, axis=0)

       >>> np.allclose(dft_x, czt_x)
       True

    Implementation of the iDFT:

    .. doctest::

       >>> N = M = 10
       >>> x = np.random.randn(N) + 1j * np.random.randn(N)

       >>> idft_x = np.fft.ifft(x)
       >>> czt_x = czt(x, A=1, W=np.exp(1j * 2 * np.pi / N), M=M)

       >>> np.allclose(idft_x, czt_x / N)  # czt() does not do the scaling.
       True
    """
    x = np.array(x, copy=False)
    A = complex(A)
    W = complex(W)

    if not cmath.isclose(abs(A), 1):
        raise ValueError('Parameter[A] must lie on the unit circle for '
                         'numerical stability.')
    if not cmath.isclose(abs(W), 1):
        raise ValueError('Parameter[W] must lie on the unit circle.')
    if M <= 0:
        raise ValueError('Parameter[M] must be positive.')
    if not (-x.ndim <= axis < x.ndim):
        raise ValueError('Parameter[axis] is out-of-bounds.')

    # Shape Parameters
    N = x.shape[axis]
    L = fftpack.next_fast_len(N + M - 1)
    sh_N_1 = [1] * x.ndim
    sh_N_1[axis] = N - 1

    sh_N = [1] * x.ndim
    sh_N[axis] = N

    sh_M = [1] * x.ndim
    sh_M[axis] = M

    sh_L = [1] * x.ndim
    sh_L[axis] = L

    sh_Y = list(x.shape)
    sh_Y[axis] = L

    # Modulation Parameters
    n = np.arange(L)
    y_modulation = (A ** -n[:N]) * np.float_power(W, (n[:N] ** 2) / 2)
    v = np.zeros(L, dtype=complex)
    v[:M] = np.float_power(W, - (n[:M] ** 2) / 2)
    v[L - N + 1:] = np.float_power(W, - ((L - n[L - N + 1:]) ** 2) / 2)
    g_modulation = np.float_power(W, (n[:M] ** 2) / 2)

    y = np.zeros(sh_Y, dtype=complex)
    y[_index(y, axis, slice(N))] = x
    y[_index(y, axis, slice(N))] *= y_modulation.reshape(sh_N)
    G = fftpack.fft(y, axis=axis)
    G *= fftpack.fft(v).reshape(sh_L)
    g = fftpack.ifft(G, axis=axis)
    g[_index(g, axis, slice(M))] *= g_modulation.reshape(sh_M)

    X = g[_index(g, axis, slice(M))]
    return X


@chk.check(dict(x_FS=chk.accept_any(chk.has_reals, chk.has_complex),
                T=chk.is_real,
                a=chk.is_real,
                b=chk.is_real,
                M=chk.is_integer,
                axis=chk.is_integer,
                real_x=chk.is_boolean))
def fs_interp(x_FS, T, a, b, M, axis=-1, real_x=False):
    r"""
    Interpolate bandlimited periodic signal.

    If `x_FS` holds the Fourier Series coefficients of a bandlimited periodic
    function :math:`x(t): \mathbb{R} \to \mathbb{C}`, then
    :py:func:`~pypeline.util.math.fourier.fs_interp` computes the values of
    :math:`x(t)` at points
    :math:`t[k] = (a + \frac{b - a}{M - 1} k) 1_{[0,\ldots,M-1]}[k]`.

    Parameters
    ----------
    x_FS : array-like (float or complex)
        (..., N_FS, ...) FS coefficients in the order
        :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}\right]`.
    T : float
        Function period.
    a : float
        Interval LHS.
    b : float
        Interval RHS.
    M : int
        Number of points to interpolate.
    axis : int
        Dimension of `x_FS` along which the FS coefficients are stored.
    real_x : bool
        If True, assume that `x_FS` is conjugate symmetric and use a more
        efficient algorithm. In this case, the FS coefficients corresponding to
        negative frequencies are not used.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (..., M, ...) interpolated values
        :math:`\left[ x(t[0]), \ldots, x(t[M-1]) \right]` along the axis
        indicated by `axis`.
        If `real_x` is :py:obj:`True`, the output is real-valued, otherwise it
        is complex-valued.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import math
       from pypeline.util.math.fourier import fs_interp

       def dirichlet(x, T, T_c, N_FS):
           y = x - T_c

           n, d = np.zeros((2, len(x)))
           nan_mask = np.isclose(np.fmod(y, np.pi), 0)
           n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
           d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
           n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
           d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

           return n / d

       # Parameters of the signal.
       T, T_c, N_FS = math.pi, math.e, 15
       N = (N_FS - 1) // 2

       # Generate interpolated signal
       a, b = T_c + (T / 2) *  np.r_[-1, 1]
       M = 100  # We want lots of points.
       diric_FS = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])


    Let :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}` be the Fourier Series (FS)
    coefficients of a shifted Dirichlet kernel of period :math:`T`:

    .. math::

       \phi_{k}^{FS} =
       \begin{cases}
           \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
           0 & \text{otherwise}.
       \end{cases}

    .. doctest::

       # Parameters of the signal.
       >>> T, T_c, N_FS = math.pi, math.e, 15
       >>> N = (N_FS - 1) // 2

       # And the kernel's FS coefficients.
       >>> diric_FS = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])


    Being bandlimited, we can use
    :py:func:`~pypeline.util.math.fourier.fs_interp` to numerically evaluate
    :math:`\phi(t)` on the interval
    :math:`\left[ T_{c} - \frac{T}{2}, T_{c} + \frac{T}{2} \right]`.

    .. doctest::

       # Generate interpolated signal
       >>> a, b = T_c + (T / 2) *  np.r_[-1, 1]
       >>> M = 100  # We want lots of points.
       >>> diric_sig = fs_interp(diric_FS, T, a, b, M)

       # Compare with theoretical result.
       >>> t = a + (b - a) / (M - 1) * np.arange(M)
       >>> diric_sig_exact = dirichlet(t, T, T_c, N_FS)

       >>> np.allclose(diric_sig, diric_sig_exact)
       True


    The Dirichlet kernel is real-valued, so we can set `real_x` to use the
    accelerated algorithm instead:

    .. doctest::

       # Generate interpolated signal
       >>> a, b = T_c + (T / 2) *  np.r_[-1, 1]
       >>> M = 100  # We want lots of points.
       >>> diric_sig = fs_interp(diric_FS, T, a, b, M, real_x=True)

       # Compare with theoretical result.
       >>> t = a + (b - a) / (M - 1) * np.arange(M)
       >>> diric_sig_exact = dirichlet(t, T, T_c, N_FS)

       >>> np.allclose(diric_sig, diric_sig_exact)
       True


    Notes
    -----
    Theory: :ref:`fp_interp_def`.

    See Also
    --------
    :py:func:`~pypeline.util.math.fourier.czt`
    """
    x_FS = np.array(x_FS, copy=False)

    if T <= 0:
        raise ValueError('Parameter[T] must be positive.')
    if not (a < b):
        raise ValueError(f'Parameter[a] must be smaller than Parameter[b].')
    if M <= 0:
        raise ValueError('Parameter[M] must be positive.')
    if not (-x_FS.ndim <= axis < x_FS.ndim):
        raise ValueError('Parameter[axis] is out-of-bounds.')

    N_FS = x_FS.shape[axis]
    N = (N_FS - 1) // 2
    A = np.exp(-1j * 2 * np.pi / T * a)
    W = np.exp(1j * (2 * np.pi / T) * (b - a) / (M - 1))
    E = np.arange(M)
    sh = [1] * x_FS.ndim
    sh[axis] = M

    if real_x:  # Real-valued functions.
        x0_FS = x_FS[_index(x_FS, axis, slice(N, N + 1))]
        xp_FS = x_FS[_index(x_FS, axis, slice(N + 1, N_FS))]
        C_1 = 1 / A
        C_2 = np.reshape(W ** E, sh)

        x = czt(xp_FS, A, W, M, axis=axis) * (C_1 * C_2)
        x = (x0_FS + 2 * x).real
    else:  # Complex-valued functions.
        C_1 = A ** N
        C_2 = np.reshape(W ** (-N * E), sh)
        x = czt(x_FS, A, W, M, axis=axis) * (C_1 * C_2)

    return x
