.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [sep@zurich.ibm.com]
.. ############################################################################


Installation
============

Pypeline modules are written in Python3/C++14 and tested on x86_64 systems running Linux.

Third-party Python dependencies often require non-Python packages to be available on the system.
We therefore recommend using `conda <https://conda.io/docs/>`_ to install most requirements.

The C++ library ``libpypeline.so`` requires the following tools to be available:

+-------------+------------+
| Library     |    Version |
+=============+============+
| xtl         |     0.4.14 |
+-------------+------------+
| xsimd       |     6.1.5  |
+-------------+------------+
| xtensor     |     0.17.1 |
+-------------+------------+
| Eigen       |     3.3.5  |
+-------------+------------+
| PyBind11    |     2.2.3  |
+-------------+------------+
| FFTW        |     3.3.8  |
+-------------+------------+
| Intel MKL   |   2018.0.3 |
+-------------+------------+

Aside from Intel MKL, versions of these libraries shipped with `conda <https://conda.io/docs/>`_ are not configured correctly for Pypeline and must therefore be compiled manually.

After installing `Miniconda <https://conda.io/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/download/#linux>`_, run the following::

    $ PYPELINE_C_COMPILER=<path_to_executable>
    $ PYPELINE_CXX_COMPILER=<path_to_executable>
    $ cd <pypeline_dir>/
    $ conda create --name=pypeline_dev   \
                   --channel=defaults    \
                   --channel=conda-forge \
                   --file=conda_requirements.txt
    $ python3 build.py --download_dependencies
    $ python3 build.py --install_dependencies                    \
                       --C_compiler="${PYPELINE_C_COMPILER}"     \
                       --CXX_compiler="${PYPELINE_CXX_COMPILER}"
    $ python3 build.py --lib={Debug, Release}                    \
                       --C_compiler="${PYPELINE_C_COMPILER}"     \
                       --CXX_compiler="${PYPELINE_CXX_COMPILER}" \
                      [--OpenMP]
    $ python3 test.py         # Run test suite (optional, recommended)
    $ python3 build.py --doc  # Generate documentation (optional)


To launch a Python3 shell containing Pypeline, run ``pypeline.sh``.


Remarks
-------

* Pypeline is internally tested with GCC 5.4.1, 7.3.1, 8.1.1 and Clang 6.0.1.
* If building with ``--OpenMP``, Cmake may incorrectly link ``libpypeline.so`` with a version of OpenMP shipped with `conda` instead of the system's OpenMP shared library.
  In case the compilation stage above fails, inspect Cmake's log files for OpenMP ambiguities.
* The ``--install_dependencies`` command above will automatically download and install all C++ dependencies listed in the table.
  If the libraries are already available on the system and you wish to use them instead of the ones we provide, then you will have to modify ``CMakeLists.txt`` and configuration files under ``cmake/`` accordingly.
* Aside from :py:mod:`pypeline.phased_array.util.io`, Pypeline should also run correctly on ppc64le Linux platforms and OSX, but we provide no support for this.
* ``pypeline.sh`` sets up the required environment variables to access built libraries and setup key components such as OpenMP thread-count and MKL precision preferences.
  It is highly recommended to check the ``load_pypeline_env()`` function in this file and tailor some environment variables to your system.
