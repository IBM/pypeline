# #############################################################################
# core.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# Revision : 0.1
# Last updated : 2018-05-01 07:56:45 UTC
# #############################################################################

"""
Foundational constructs.

Pypeline is structured around the concept of :py:class:`~pypeline.core.Block`:
callable objects that perform certain actions given their optional inputs.
"""

from abc import ABC, abstractmethod
from typing import Any


class Block(ABC):
    """
    Abstract Block interface.

    Blocks can have internal state of any kind and must implement
    :py:meth:`~pypeline.core.Block.__call__`.
    This function is assumed to be useable directly after block
    initialization, unless said otherwise.

    Blocks are allowed to extend this interface with public attributes and
    methods.

    .. testsetup::

       from pypeline.core import Block

    .. doctest::

       >>> class A(Block):
       ...     def __call__(self):
       ...         return 5

       >>> blk = A()
       >>> blk()  # invoke A.__call__()
       5

    .. doctest::

       >>> class B(Block):
       ...     pass

       >>> B()
       Traceback (most recent call last):
           ...
       TypeError: Can't instantiate abstract class B with abstract methods
           __call__
        """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Perform block's task given input parameters.

        As blocks encode a data-flow view on computation,
        :py:meth:`~pypeline.core.Block.__call__` is *forbidden* from mutating
        its parameters in any way.

        :param args: positional arguments
        :param kwargs: keyword arguments
        :return: output value, if any.
                 A block's internal state can also be modified.
        """
        raise NotImplementedError
