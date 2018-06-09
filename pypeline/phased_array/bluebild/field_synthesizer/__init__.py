# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Field synthesizers.
"""

import pypeline.core as core


class FieldSynthesizerBlock(core.Block):
    """
    Top-level public interface for Bluebild field synthesizers.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        Compute instantaneous field statistics.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            Field statistics.
        """
        raise NotImplementedError

    def synthesize(self, *args, **kwargs):
        """
        Compute field values from statistics.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            Field values.
        """
        raise NotImplementedError
