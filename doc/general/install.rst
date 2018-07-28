.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [sep@zurich.ibm.com]
.. ############################################################################


Installation
============

Pypeline relies on the `PyData <https://pydata.org>`_ and `NumFOCUS <https://www.numfocus.org/>`_ stacks.
Due to strong non-Python dependencies required to achieve high performance, we highly recommend the use of `conda <https://conda.io/docs/>`_ packages when available.

Pypeline is developed and tested on x86_64 systems running Linux.
Aside from :py:mod:`pypeline.phased_array.util.io`, Pypeline should also run correctly on ppc64le Linux platforms and OSX, but we provide no support for this.

After installing `Miniconda <https://conda.io/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/download/#linux>`_, run the following::

    $ cd <pypeline_dir>/
    $ conda create --name=pypeline_dev   \
                   --channel=defaults    \
                   --channel=conda-forge \
                   --file=conda_requirements.txt
    $ python3 build.py --lib={Debug, Release}
    $ python3 test.py         # Run test suite (optional, recommended)
    $ python3 build.py --doc  # Generate documentation (optional)


To launch a Python3 shell containing Pypeline, run ``pypeline.sh``.


Remarks
-------

Depending on your host environment, Cmake may incorrectly link ``libpypeline.so`` with a version of OpenMP shipped with `conda` instead of the system's OpenMP shared library.
In case the compilation stage above fails, inspect Cmake's log files for OpenMP ambiguities.
