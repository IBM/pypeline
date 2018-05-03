# #############################################################################
# test_argcheck.py
# ================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# Revision : 0.0
# Last updated : 2018-04-05 14:09:31 UTC
# #############################################################################

import itertools

import numpy as np
import pytest

from pypeline.util.argcheck import check, has_shape, is_array_like, \
    is_array_shape, is_scalar


class TestCheck:
    def test_2_arg_form(self):
        wrong_param_types = (None, True, 1, tuple(), lambda x: True)
        for p in wrong_param_types:
            with pytest.raises(TypeError):
                check(p, lambda x: True)

        wrong_param_values = ('', 'while', 'if')
        for p in wrong_param_values:
            with pytest.raises(ValueError):
                check(p, lambda x: True)

        wrong_func_types = (None, True, 1,
                            tuple(), (lambda x: True, 1,), [2, lambda x: True])
        for f in wrong_func_types:
            with pytest.raises(Exception):
                check('x', f)

        assert callable(check('x', lambda y: False))

        @check('x', [lambda y: y == 5,
                     lambda z: isinstance(z, int)])
        def f(x):
            return x

        with pytest.raises(ValueError):
            f(3)
        assert f(5) is 5

    def test_1_arg_form(self):
        bool_func = lambda x: True
        bool_funcs = [bool_func] * 2

        assert callable(check(dict(x=bool_func,
                                   y=bool_funcs)))
        assert callable(check({'x': bool_func,
                               'y': bool_funcs}))

        with pytest.raises(TypeError):
            check(dict(x=None,
                       y=bool_func,
                       z=bool_funcs))
        with pytest.raises(ValueError):
            check({'while': bool_func,
                   'x': bool_funcs})

    def test_3plus_arg_form(self):
        with pytest.raises(ValueError):
            check(1, 2, 3)

    def test_docstring(self):
        @check('x', lambda y: True)
        def f(x):
            """function documentation"""
            return x

        assert f.__doc__ == """function documentation"""
        assert f.__name__ == 'f'


class TestIsScalar:
    def test_func(self):
        true_params = (5, 5.0, 1j, np.r_[0][0], True)
        for _ in true_params:
            assert is_scalar(_) is True

        false_params = (tuple(), [], set(), {}, np.array([]), np.r_[:5])
        for _ in false_params:
            assert is_scalar(_) is False


class TestIsArrayLike:
    def test_func(self):
        true_params = (tuple(), [], np.array([]), (5,), [5, ], np.r_[:5], 'ab')
        for _ in true_params:
            assert is_array_like(_) is True

        false_params = (5, 5.0, set(), {})
        for _ in false_params:
            assert is_array_like(_) is False


class TestIsArrayShape:
    def test_func(self):
        error_args = (5, 5.0, set(), {})
        for _ in error_args:
            with pytest.raises(ValueError):
                is_array_shape(_)

        true_args = ((1,), [1, ], (5, 3, 4), np.r_[2:5])
        for _ in true_args:
            assert is_array_shape(_) is True

        false_args = ((0,), (1, 0), (1, 1.0), np.ones((5, 3)))
        for _ in false_args:
            assert is_array_shape(_) is False


class TestHasShape:
    def test_func(self):
        error_x = (5, 5.0, set(), {})
        error_shape = error_x + ((1,), [1.], (5, 3, 4), np.r_[2:5])
        for x, shape in itertools.product(error_x, error_shape):
            with pytest.raises(ValueError):
                has_shape(x, shape)

        true_pairs = ((np.r_[:5], (5,)),
                      (np.ones((5, 3, 4)), (5, 3, 4)),
                      (np.zeros((1, 2, 3, 4)), (1, 2, 3, 4)))
        for x, shape in true_pairs:
            assert has_shape(x, shape) is True
