# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

r"""
Bluebild imaging tools.

Bluebild is a family of algorithms in phased-array interferometry to compute continuously-defined second-order intensity statistics directly on the sphere :math:`\mathbb{S}^{2}`.
It is decomposed into 3 main stages:

* **Data processing**: orthogonalize visibility measurements :math:`\Sigma \in \mathbb{C}^{N \times N}` with respect to the instrument Gram matrix :math:`G \in \mathbb{C}^{N \times N}` to obtain compact descriptions of the intensity field's energy levels.

* **Field synthesis**: use the interpolation operator ideally-matched to the instrument's sampling operator to evaluate individual energy levels at arbitrary points :math:`r \in \mathbb{S}^{2}`, obtaining a *functional* PCA decomposition of the intensity field.

* **Aggregation**: re-weight and combine energy levels to obtain different field estimates.

Subclasses of :py:mod:`~pypeline.phased_array.bluebild.data_processor.DataProcessorBlock`, :py:mod:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` and :py:mod:`~pypeline.phased_array.bluebild.imager.IntegratingMultiFieldSynthesizerBlock` implement the data processing, field synthesis and aggregation stages respectively.
Subclasses of :py:mod:`~pypeline.phased_array.bluebild.parameter_estimator.ParameterEstimator` can be used to optimally initialize the objects above.
"""
