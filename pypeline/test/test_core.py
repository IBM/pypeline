# #############################################################################
# test_core.py
# ============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# Revision : 0.0
# Last updated : 2018-04-05 14:09:31 UTC
# #############################################################################

import pytest

from pypeline.core import Block


class TestBlock:
    """
    Test :py:class:`~pypeline.core.Block`.
    """

    def test_fail_abstract_instantiation(self):
        """
        Abstract blocks cannot be instantiated.
        """
        class B(Block):
            pass

        with pytest.raises(TypeError):
            B()

    def test_concrete_instantiation(self):
        """
        Blocks overloading :py:meth:`~pypeline.core.Block.__call__` can be
        instantiated.
        """
        class A(Block):
            def __call__(self):
                return 5

        obj = A()
        assert callable(obj)
        assert obj() == 5
