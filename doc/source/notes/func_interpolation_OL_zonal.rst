.. ############################################################################
.. func_interpolation_OL_zonal.rst
.. ===============================
.. Author : Sepand KASHANI [sep@zurich.ibm.com]
.. ############################################################################


.. _ZOL_def:

Reconstruction of Order-limited Zonal Functions on the Sphere
=============================================================

Theory
******

Let :math:`f: \mathbb{S}^{2} \to \mathbb{C}` be a zonal function.
Then :math:`f` can be evaluated at any :math:`r \in \mathbb{S}^{2}` using

.. math::
   :label: SH_interp

   f(r) = \sum_{n \ge 0} \sum_{m = -n}^{n} f_{nm} Y_{n}^{m}(r),

where :math:`Y_{n}^{m}` is the spherical harmonic function of order :math:`n`
and degree :math:`m`, and :math:`f_{nm}` are the spherical harmonic
coefficients of :math:`f` defined as

.. math::

   f_{nm} = \langle f, Y_{n}^{m} \rangle_{\mathbb{S}^{2}}
          = \int_{\mathbb{S}^{2}} f(r) \overline{Y_{n}^{m}(r)} dr.

If :math:`f` is order-limited to :math:`N`, then the :math:`f_{nm}` can be
evaluated using cubature formulae as

.. math::
   :label: SH_cubature

   f_{nm} = \sum_{q, l} \alpha_{q l} f_{q l} \overline{Y_{n}^{m}(r_{q l})},

where the :math:`\alpha_{q l}` depend on the geometry of the sampling points
:math:`r_{q l} \in \mathbb{S}^{2}`.

Combining :eq:`SH_interp` and :eq:`SH_cubature`, we get

.. math::
   :label: SH_general_sampling

   f(r) = \frac{N + 1}{4 \pi} \sum_{q, l} \alpha_{q l} f_{q l}
                                          K_{N}(\langle r, r_{q l} \rangle),

where
:math:`K_{N}(x) = \left[P_{N+1}(x) - P_{N}(x)\right] / \left[ x - 1 \right]`
is the spherical Dirichlet kernel of order :math:`N`.
Hence an order-limited zonal function :math:`f(r)` can be perfectly
reconstructed from a finite number of samples :math:`f_{q l}`.
In the case of Equal-Angle sampling:

.. math::

   \theta_{q} & = \frac{\pi}{2 N + 2} \left( q + \frac{1}{2} \right),
                  \qquad & q \in \{ 0, \ldots, 2 N + 1 \},

   \phi_{l} & = \frac{2 \pi}{2N + 2} l,
                  \qquad & l \in \{ 0, \ldots, 2 N + 1 \},

:eq:`SH_general_sampling` simplifies to

.. math::

   f(r) & = \sum_{q = 0}^{2 N + 1} \sum_{l = 0}^{2 N + 1} \beta_{q l} f_{q l}
            K_{N}(\langle r, r_{q l} \rangle),

   \beta_{q l} & = \frac{1}{2 N + 2} \sin\theta_{q} \sum_{a = 0}^{N}
                   \frac{\sin[(2 a + 1) \theta_{q}]}{2 a + 1}.


Implementation Notes
********************

* :py:func:`~pypeline.util.math.sphere.ea_sample` and
  :py:func:`~pypeline.util.math.sphere.ea_interp` can be used to sample
  order-limited zonal functions and evaluate them at arbitrary
  :math:`r \in \mathbb{S}^{2}`.

* When :math:`N` is large, precise values of :math:`K_{N}(x)` can only be
  achieved using the recurrence relation on Legendre polynomials.
  If minor errors can be tolerated, it is computationally advantageous to use
  function interpolation to approximate :math:`K_{N}(x)`.
  Both exact and approximate values of :math:`K_{N}(x)` can be obtained using
  :py:func:`~pypeline.util.math.func.sph_dirichlet`.
