.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [sep@zurich.ibm.com]
.. ############################################################################


Installation
============

Pypeline modules are written in Python3 and C++14.
Due to strong non-Python dependencies required to achieve high performance, we recommend the use of `conda <https://conda.io/docs/>`_ to install everything.

Pypeline is developed and tested on x86_64 systems running Linux.
Aside from :py:mod:`pypeline.phased_array.util.io`, Pypeline should also run correctly on ppc64le Linux platforms and OSX, but we provide no support for this.

After installing `Miniconda <https://conda.io/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/download/#linux>`_, run the following::

    $ cd <pypeline_dir>/
    $ conda create --name=pypeline_dev   \
                   --channel=defaults    \
                   --channel=conda-forge \
                   --file=conda_requirements.txt
    $ python3 build.py --lib={Debug, Release}
                       [--C_compiler=<path_to_executable>]
                       [--CXX_compiler=<path_to_executable>]
                       [--OpenMP]
    $ python3 test.py         # Run test suite (optional, recommended)
    $ python3 build.py --doc  # Generate documentation (optional)


To launch a Python3 shell containing Pypeline, run ``pypeline.sh``.


Remarks
-------

* Pypeline is internally tested with GCC 5.4.1, 7.3.1, 8.1.1 and Clang 6.0.1.
* If building with ``--OpenMP``, Cmake may incorrectly link ``libpypeline.so`` with a version of OpenMP shipped with `conda` instead of the system's OpenMP shared library.
  In case the compilation stage above fails, inspect Cmake's log files for OpenMP ambiguities.
