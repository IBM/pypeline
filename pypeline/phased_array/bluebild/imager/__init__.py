# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

r"""
High-level Bluebild interfaces.

Let :math:`I_{k}(r,t)` denote the :math:`k`-th energy-level obtained at time :math:`t`.
Subclasses of :py:class:`~pypeline.phased_array.bluebild.imager.IntegratingMultiFieldSynthesizerBlock` do 3 things:

* integrate snapshot images to obtain integrated images spanning many observation periods: :math:`I_{k}(r) = \sum_{q=1}^{N_{t}} I_{k}(r, t_{q})`;
* aggregate energy levels;
* re-weight energy levels to output both a least-squares estimate and a standardized estimate of the (integrated) field.

Integrated images can then be directly output in viewable form by calling :py:meth:`~pypeline.phased_array.bluebild.imager.IntegratingMultiFieldSynthesizerBlock.as_image`.
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
        std : :py:class:`~pypeline.phased_array.util.io.image.SphericalImage`
            (N_level, N_height, N_width) standardized energy-levels.

        lsq : :py:class:`~pypeline.phased_array.util.io.image.SphericalImage`
            (N_level, N_height, N_width) least-squares energy-levels.
        """
        raise NotImplementedError
