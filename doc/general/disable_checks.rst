.. ############################################################################
.. disable_checks.rst
.. ==================
.. Author : Sepand KASHANI [sep@zurich.ibm.com]
.. ############################################################################


Disable Argument Validation for Faster Execution
================================================

In an effort to add static type checking to Pypeline, most functions/methods are decorated with :py:func:`~pypeline.util.argcheck.check` such as below:

.. doctest::

   >>> from pypeline.util.argcheck import check

   >>> def is_5(_):
   ...     return _ == 5

   >>> @check('x', is_5)
   ... def f(x):
   ...     return x

   >>> f(5)
   5

   >>> f('a')
   Traceback (most recent call last):
       ...
   ValueError: Parameter[x] of f() does not satisfy is_5().

Argument checks can be expensive when used extensively.
If you know your Pypeline scripts are correct (i.e., they execute without error), you can disable validation tests done with :py:func:`~pypeline.util.argcheck.check` by setting ``util.argcheck.check.ignore_checks`` to ``False`` in ``~/.pypeline/pypeline.cfg``::


   # ##########################################################################
   # pypeline.cfg
   # ============
   # ##########################################################################

   [util.argcheck.check]
   ignore_checks = True

.. hint::

   If you do not want to restart the Python interpreter to enforce this change, it is possible to reload Pypeline's configuration::

      >>> # Modify pypeline.cfg to disable checks.
      >>> import pypeline
      >>> pypeline.reload_config()

.. warning::

   Some tests will *not* pass if ``ignore_checks = True``.
