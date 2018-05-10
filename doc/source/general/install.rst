.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [sep@zurich.ibm.com]
.. ############################################################################


Installation
============

Pypeline relies on the `PyData <https://pydata.org>`_ and `NumFOCUS
<https://www.numfocus.org/>`_ stacks. Due to strong non-Python dependencies
required to achieve high performance, we highly recommend the use of
`conda <https://conda.io/docs/>`_ packages when available.

Pypeline is developed and tested on x86_64 systems running Linux.
Aside from ``pypeline.phased_array.io``, Pypeline should also run correctly on
ppc64le Linux platforms and OSX, but we provide no support for this.


Development Environment
-----------------------

After installing `Miniconda <https://conda.io/miniconda.html>`_ or
`Anaconda <https://www.anaconda.com/download/#linux>`_, run the following::

    $ cd <pypeline_dir>/
    $ conda create --name=pypeline_dev \
                   --channel=defaults \
                   --channel=conda-forge \
                   --file=conda_requirements.txt
    $ source activate pypeline_dev
    $ python3 setup.py develop
    $ tox  # Run test suites (optional)
    $ python3 setup.py build_sphinx  # Generate documentation (optional)


Release Environment
-------------------

Release environments are not supported at this point in time.
Development environments can be deployed in the meantime.
