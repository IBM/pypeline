# #############################################################################
# argcheck.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# Revision : 0.0
# Last updated : 2018-04-05 14:09:31 UTC
# #############################################################################

"""
Helper functions to ease argument checking.
"""

import functools
import inspect
import keyword
import math
from numbers import Complex, Integral, Real
from typing import Any, Callable, Container, Mapping, Sequence, \
    Union

import astropy.units as u
import numpy as np

BoolFunc = Callable[[Any], bool]


def check(*args) -> Callable:
    """
    Function decorator: raise :py:exc:`ValueError` when parameter fails
    boolean test(s).

    It is common to check parameters for correctness before executing the
    function/class to which they are bound using boolean functions.
    These boolean functions typically do *not* raise :py:exc:`Exception` when
    something goes wrong.
    :py:func:`check` is a decorator that intercepts the output of such boolean
    functions and raises :py:exc:`ValueError` when the result is
    :py:obj:`False`.

    :param args: several invocation modes possible:

        * 2-argument mode:

            * args[0]: name of decorated function's parameter to test;
            * args[1]: (list of) boolean function(s) to apply to parameter
              value.

        * 1-argument mode: (parameter name -> (list of) boolean function(s))
          mapping.

    :return: the decorated function is invoked if all tests return
        :py:obj:`True`, otherwise :py:exc:`ValueError` is raised.

    .. testsetup::

       from pypeline.util.argcheck import check

    Two-arg syntax:

    .. doctest::

       >>> def is_5(x):
       ...     return x == 5

       >>> class A:
       ...     @check('a', is_5)
       ...     def __init__(self, a):
       ...         self.a = a

       >>> A(5).a
       5

       >>> A(4)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[a] of A.__init__() does not satisfy is_5().

    Mapping syntax:

    .. doctest::

       >>> def is_5(x):
       ...     return int(x) == 5

       >>> def is_int(x):
       ...     return isinstance(x, int)

       >>> def is_str(x):
       ...     return isinstance(x, str)

       >>> class A:
       ...     @check(dict(a=[is_5, is_str],
       ...                 b=is_int))
       ...     def __init__(self, a, b):
       ...         self.a = a
       ...         self.b = b

       >>> obj = A('5', 3)
       >>> obj.a, obj.b
       ('5', 3)

       >>> A(5, 3)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[a] of A.__init__() does not satisfy is_str().

    Testing several boolean functions is also supported:

    .. doctest::

       >>> def is_5(x):
       ...     return int(x) == 5

       >>> def is_int(x):
       ...     return isinstance(x, int)

       >>> class A:
       ...     @check('a', [is_5, is_int])
       ...     def __init__(self, a):
       ...         self.a = a

       >>> A(5).a
       5

       >>> A(4)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[a] of A.__init__() does not satisfy is_5().

       >>> A('5')
       Traceback (most recent call last):
           ...
       ValueError: Parameter[a] of A.__init__() does not satisfy is_int().
    """
    if len(args) == 1:
        return _check(m=args[0])
    elif len(args) == 2:
        return _check(m={args[0]: args[1]})
    else:
        raise ValueError('Expected 1 or 2 arguments.')


def _check(m: Mapping[str, Union[BoolFunc, Sequence[BoolFunc]]]) -> Callable:
    if not isinstance(m, Mapping):
        a = _check.__annotations__['m']
        raise TypeError(f'Expected {a}')

    key_error = lambda k: f'Key[{k}] must be a valid string identifier.'
    value_error = lambda k: (f'Value[Key[{k}]] must be '
                             'Union[{BoolFunc}, Sequence[{BoolFunc}]].')
    for k, v in m.items():
        if not isinstance(k, str):
            raise TypeError(key_error(k))
        if not (k.isidentifier() and (not keyword.iskeyword(k))):
            raise ValueError(key_error(k))

        if isinstance(v, Sequence) and (len(v) == 0):
            raise ValueError(value_error(k))
        for _ in (v if isinstance(v, Sequence) else (v,)):
            if not callable(_):
                raise TypeError(value_error(k))

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_args = inspect.getcallargs(func, *args, **kwargs)

            for k, v in m.items():
                if k not in func_args.keys():
                    raise ValueError(f'Parameter[{k}] not part of '
                                     f'{func.__qualname__}() parameter list.')

                for fn in (v if isinstance(v, Sequence) else (v,)):
                    if fn(func_args[k]) is False:
                        raise ValueError(f'Parameter[{k}] of '
                                         f'{func.__qualname__}() does not '
                                         f'satisfy {fn.__name__}().')

            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_instance(klass) -> Callable:
    """
    Return function to test if it's argument satisfies one of the type(s)
    ``klass``.

    :param klass: type or list of types.
    :return: [:py:class:`function`]

    .. testsetup::

       import numpy as np
       from pypeline.util.argcheck import is_instance, check

    .. doctest::

       >>> is_instance([str, int])('5')
       True

       >>> is_instance(np.ndarray)([])
       False

    :py:func:`~pypeline.util.argcheck.is_instance` and
    :py:func:`~pypeline.util.argcheck.check` can be combined to validate
    parameter types:

    .. doctest::

       >>> @check('x', is_instance(int))
       ... def f(x):
       ...     return type(x)

       >>> f(5)
       <class 'int'>

       >>> f('5')
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of f() does not satisfy is_instance(int)().
    """
    if not (inspect.isclass(klass) or
            (isinstance(klass, Sequence) and
             all(inspect.isclass(_) for _ in klass))):
        raise TypeError('Parameter[klass] must be a class or list of classes')

    klass = (klass,) if inspect.isclass(klass) else tuple(klass)

    def _is_instance(x):
        if isinstance(x, klass):
            return True

        return False

    _is_instance.__name__ = f'is_instance({klass})'

    return _is_instance


def is_scalar(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a scalar object.

    :param x: object to test.

    .. testsetup::

       from pypeline.util.argcheck import is_scalar

    .. doctest::

       >>> is_scalar(5)
       True

       >>> is_scalar([5])
       False
    """
    return not isinstance(x, Container)


def is_array_like(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is an array-like object.

    :param x: object to test.

    .. testsetup::

       from pypeline.util.argcheck import is_array_like

    .. doctest::

       >>> is_array_like(5)
       False

       >>> [is_array_like(_) for _ in (tuple(), [], np.array([]), range(5))]
       [True, True, True, True]

       >>> [is_array_like(_) for _ in (set(), dict())]
       [False, False]
    """
    if isinstance(x, (np.ndarray, Sequence)):
        return True

    return False


@check('x', is_array_like)
def is_array_shape(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a valid array shape specifier.

    :param x: shape-like specifier.

    .. testsetup::

       from pypeline.util.argcheck import is_array_shape

    .. doctest::

       >>> is_array_shape((5, 4))
       True

       >>> is_array_shape((5, 0))
       False
    """
    x = np.array(x, copy=False)

    if x.ndim == 1:
        if ((len(x) > 0) and
                np.issubdtype(x.dtype, np.integer) and
                np.all(x > 0)):
            return True

    return False


@check('shape', is_array_shape)
def has_shape(shape) -> Callable:
    """
    Return function to test if it's array-like argument has dimensions
    ``shape``.

    :param shape: desired dimensions.
    :return: [:py:class:`function`]

    .. testsetup::

       from pypeline.util.argcheck import has_shape, check

    .. doctest::

       >>> has_shape((1,))([5,])
       True

       >>> has_shape([5,])((1, 2))
       False

    :py:func:`~pypeline.util.argcheck.has_shape` and
    :py:func:`~pypeline.util.argcheck.check` can be combined to validate array
    dimensions:

    .. doctest::

       >>> @check('x', has_shape([2, 2]))
       ... def f(x):
       ...     x = np.array(x)
       ...     return x.shape

       >>> f([5])
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of f() does not satisfy has_shape((2, 2))().

       >>> f([[1, 2], [3, 4]])
       (2, 2)
    """
    shape = tuple(shape)

    @check('x', is_array_like)
    def _has_shape(x) -> bool:
        x = np.array(x, copy=False)

        if x.shape == shape:
            return True

        return False

    _has_shape.__name__ = f'has_shape({shape})'

    return _has_shape


@check('x', is_scalar)
def is_integer(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is an integer.

    :param x: object to test.

    .. testsetup::

       from pypeline.util.argcheck import is_integer

    .. doctest::

       >>> is_integer(5)
       True

       >>> is_integer(5.0)
       False
    """
    return isinstance(x, Integral)


@check('x', is_array_like)
def has_integers(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains integers.

    :param x: array-like object.

    .. testsetup::

       import numpy as np
       from pypeline.util.argcheck import has_integers

    .. doctest::

       >>> has_integers([5]), has_integers(np.r_[:5])
       (True, True)

       >>> has_integers([5.]), has_integers(np.ones((5, 3)))
       (False, False)
    """
    x = np.array(x, copy=False)

    if np.issubdtype(x.dtype, np.integer):
        return True

    return False


@check('x', is_scalar)
def is_boolean(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a boolean.

    :param x: object to test.

    .. testsetup::

       from pypeline.util.argcheck import is_boolean

    .. doctest::

       >>> is_boolean(True), is_boolean(False)
       (True, True)

       >>> is_boolean(0), is_boolean(1)
       (False, False)
    """
    if isinstance(x, bool):
        return True

    return False


@check('x', is_array_like)
def has_booleans(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains booleans.

    :param x: array-like object.

    .. testsetup::

       import numpy as np
       from pypeline.util.argcheck import has_booleans

    .. doctest::

       >>> has_booleans(np.ones((1, 2), dtype=bool)), has_booleans([True])
       (True, True)

       >>> has_booleans(np.ones((1, 2)))
       False
    """
    x = np.array(x, copy=False)

    if np.issubdtype(x.dtype, np.bool_):
        return True

    return False


@check('x', is_integer)
def is_even(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is an even integer.

    :param x: integer to test.

    .. testsetup::

       from pypeline.util.argcheck import is_even

    .. doctest::

       >>> is_even(2)
       True

       >>> is_even(3)
       False
    """
    if x % 2 == 0:
        return True

    return False


@check('x', has_integers)
def has_evens(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains even integers.

    :param x: array-like of integers.

    .. testsetup::

       import numpy as np
       from pypeline.util.argcheck import has_evens

    .. doctest::

       >>> has_evens(np.arange(5))
       False

       >>> has_evens(np.arange(0, 6, 2))
       True
    """
    x = np.array(x, copy=False)

    if np.all(x % 2 == 0):
        return True

    return False


@check('x', is_integer)
def is_odd(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is an odd integer.

    :param x: integer to test.

    .. testsetup::

       from pypeline.util.argcheck import is_odd

    .. doctest::

       >>> is_odd(2)
       False

       >>> is_odd(3)
       True
    """
    if x % 2 == 1:
        return True

    return False


@check('x', has_integers)
def has_odds(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains odd integers.

    :param x: array-like of integers.

    .. testsetup::

       import numpy as np
       from pypeline.util.argcheck import has_odds

    .. doctest::

       >>> has_odds(np.arange(5))
       False

       >>> has_odds(np.arange(1, 7, 2))
       True
    """
    x = np.array(x, copy=False)

    if np.all(x % 2 == 1):
        return True

    return False


@check('x', is_integer)
def is_pow2(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a power of 2.

    :param x: integer to test.

    .. testsetup::

       from pypeline.util.argcheck import is_pow2

    .. doctest::

       >>> is_pow2(8)
       True

       >>> is_pow2(9)
       False
    """
    if x > 0:
        exp = math.log2(x)
        if math.isclose(exp, math.floor(exp)):
            return True

    return False


@check('x', has_integers)
def has_pow2s(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains powers of 2.

    :param x: array-like of integers.

    .. testsetup::

       import numpy as np
       from pypeline.util.argcheck import has_pow2s

    .. doctest::

       >>> has_pow2s([2, 4, 8])
       True

       >>> has_pow2s(np.arange(10))
       False
    """
    x = np.array(x, copy=False)

    if np.all(x > 0):
        exp = np.log2(x)
        if np.allclose(exp, np.floor(exp)):
            return True

    return False


@check('x', is_scalar)
def is_complex(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a complex number.

    :param x: object to test.

    .. testsetup::

       from pypeline.util.argcheck import is_complex

    .. doctest::

       >>> is_complex(5), is_complex(5.0)
       (False, False)

       >>> is_complex(5 + 5j), is_complex(1j * np.r_[0][0])
       (True, True)
    """
    if (isinstance(x, Complex) and
            (not isinstance(x, Real))):
        return True

    return False


@check('x', is_array_like)
def has_complex(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains complex numbers.

    :param x: array-like object.

    .. testsetup::

       from pypeline.util.argcheck import has_complex

    .. doctest::

       >>> has_complex([1j, 0])  # upcast to complex numbers.
       True

       >>> has_complex(1j * np.ones((5, 3)))
       True
    """
    x = np.array(x, copy=False)

    if np.issubdtype(x.dtype, np.complexfloating):
        return True

    return False


@check('x', is_scalar)
def is_real(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a real number.

    :param x: object to test.

    .. testsetup::

       from pypeline.util.argcheck import is_real

    .. doctest::

       >>> is_real(5), is_real(5.0)
       (True, True)

       >>> is_real(1j)
       False
    """
    return isinstance(x, Real)


@check('x', is_array_like)
def has_reals(x) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains real numbers.

    :param x: array-like object.

    .. testsetup::

       from pypeline.util.argcheck import has_reals

    .. doctest::

       >>> has_reals([5]), has_reals(np.arange(10))
       (True, True)

       >>> has_reals(1j * np.ones(5))
       False
    """
    x = np.array(x, copy=False)

    if (np.issubdtype(x.dtype, np.integer) or
            np.issubdtype(x.dtype, np.floating)):
        return True

    return False


@check('x', lambda _: isinstance(_, u.Quantity))
def is_frequency(x: u.Quantity) -> bool:
    """
    Return :py:obj:`True` if ``x`` is a (positive) frequency.

    :param x: quantity to test.

    .. testsetup::

       import astropy.units as u
       from pypeline.util.argcheck import is_frequency

    .. doctest::

       >>> is_frequency(5 * u.MHz)
       True

       >>> is_frequency(-5 * u.Hz)  # negative frequencies not allowed.
       False

       >>> is_frequency(1 * u.s)
       False
    """
    f = check('x', lambda _: _.shape == tuple())(_is_frequency)
    return f(x)


@check('x', lambda _: isinstance(_, u.Quantity))
def has_frequencies(x: u.Quantity) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains (positive) frequencies.

    :param x: quantities to test.

    .. testsetup::

       from pypeline.util.argcheck import has_frequencies

    .. doctest::

       >>> has_frequencies(np.r_[0] * u.Hz)
       True
    """
    f = check('x', lambda _: is_array_shape(_.shape))(_is_frequency)
    return f(x)


def _is_frequency(x: u.Quantity) -> bool:
    """
    Return :py:obj:`True` if ``x`` represents positive frequenc(y/ies).

    :param x: u.Quantity objects to test.
    """
    if (x.unit.is_equivalent(u.Hz) and
            np.all(x.to_value(u.Hz) >= 0)):
        return True

    return False


@check('x', lambda _: isinstance(_, u.Quantity))
def is_angle(x: u.Quantity) -> bool:
    """
    Return :py:obj:`True` if ``x`` is an angle.

    :param x: quantity to test.

    .. testsetup::

       import astropy.units as u
       from pypeline.util.argcheck import is_angle

    .. doctest::

       >>> is_angle(5 * u.deg)
       True

       >>> is_angle(1 * u.s)
       False
    """
    f = check('x', lambda _: _.shape == tuple())(_is_angle)
    return f(x)


@check('x', lambda _: isinstance(_, u.Quantity))
def has_angles(x: u.Quantity) -> bool:
    """
    Return :py:obj:`True` if ``x`` contains angles.

    :param x: quantities to test.

    .. testsetup::

       from pypeline.util.argcheck import has_angles

    .. doctest::

       >>> has_angles(np.r_[0] * u.deg)
       True
    """
    f = check('x', lambda _: is_array_shape(_.shape))(_is_angle)
    return f(x)


def _is_angle(x: u.Quantity) -> bool:
    """
    Return :py:obj:`True` if ``x`` represents angles.

    :param x: u.Quantity objects to test.
    """
    if x.unit.is_equivalent(u.rad):
        return True

    return False
