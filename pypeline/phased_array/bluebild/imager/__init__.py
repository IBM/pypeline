# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
High-level Bluebild interfaces.
"""

import pypeline.core as core


class IntegratingMultiFieldSynthesizerBlock(core.Block):
    """
    Top-level public interface of Bluebild multi-field synthesizers.
    """

    def __init__(self):
        """

        """
        super().__init__()
        self._statistics = None

    def _update(self, stat):
        if self._statistics is None:
            self._statistics = stat
        else:
            self._statistics += stat

    def __call__(self, *args, **kwargs):
        """
        Compute integrated field statistics for least-squares and standardized estimates.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            Integrated field statistics.
        """
        raise NotImplementedError

    def as_image(self):
        """
        Transform integrated statistics to viewable image.

        Returns
        -------
        std : :py:class:`~pypeline.phased_array.util.io.SphericalImage`
            (N_level, N_height, N_width) standardized energy-levels.

        lsq : :py:class:`~pypeline.phased_array.util.io.SphericalImage`
            (N_level, N_height, N_width) least-squares energy-levels.
        """
        raise NotImplementedError
