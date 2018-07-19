# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Helper functions to ease argument checking.
"""

from . import _argcheck as __py

check = __py.check
accept_any = __py.accept_any
require_all = __py.require_all
allow_None = __py.allow_None

has_shape = __py.has_shape
is_instance = __py.is_instance

has_angles = __py.has_angles
has_booleans = __py.has_booleans
has_complex = __py.has_complex
has_evens = __py.has_evens
has_frequencies = __py.has_frequencies
has_integers = __py.has_integers
has_odds = __py.has_odds
has_pow2s = __py.has_pow2s
has_reals = __py.has_reals
is_angle = __py.is_angle
is_array_like = __py.is_array_like
is_array_shape = __py.is_array_shape
is_boolean = __py.is_boolean
is_complex = __py.is_complex
is_duration = __py.is_duration
is_even = __py.is_even
is_frequency = __py.is_frequency
is_integer = __py.is_integer
is_odd = __py.is_odd
is_pow2 = __py.is_pow2
is_real = __py.is_real
is_scalar = __py.is_scalar
