# #############################################################################
# core.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Foundational constructs.

Pypeline is structured around the concept of :py:class:`~pypeline.core.Block`: callable objects that perform certain actions given their optional inputs.
"""

import abc


class Block(abc.ABC):
    """
    Abstract Block interface.

    Blocks can have internal state of any kind and must implement :py:meth:`~pypeline.core.Block.__call__`.
    This function is assumed to be useable directly after block initialization, unless said otherwise.

    Blocks are allowed to extend this interface with public attributes and methods.

    Examples
    --------
    .. testsetup::

       from pypeline.core import Block

    .. doctest::

       >>> class A(Block):
       ...     def __call__(self):
       ...         return 5

       >>> blk = A()
       >>> blk()  # invoke A.__call__()
       5
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Perform action and return result.

        Internal state can also be modified.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        :py:obj:`~typing.Any`
            Result of action.
        """
        raise NotImplementedError
