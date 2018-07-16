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


Development Environment
-----------------------

After installing `Miniconda <https://conda.io/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/download/#linux>`_, run the following::

    $ cd <pypeline_dir>/

    $ # C++/Python Dependencies ===================
    $ conda create --name=pypeline_dev   \
                   --channel=defaults    \
                   --channel=conda-forge \
                   --file=conda_requirements.txt
    $ source activate pypeline_dev

    $ # C++ Tools =================================
    $ PYPELINE_CPP_BUILD_DIR=build/cpp
    $ mkdir --parents "${PYPELINE_CPP_BUILD_DIR}"
    $ cd "${PYPELINE_CPP_BUILD_DIR}"
    $ cmake -DCMAKE_BUILD_TYPE=<Debug or Release> ../..
    $ make

    $ # Python Tools ==============================
    $ cd ../..
    $ python3 setup.py develop
    $ python3 test.py  # Run test suites (optional)
    $ python3 setup.py build_sphinx  # Generate documentation (optional)


.. Note::

    If accessing MS files from pypeline, run the following before launching python::

        $ export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}":<miniconda_dir>/lib


Release Environment
-------------------

* C++ tools can be built as Debug/Release.
* Python tools are currently only tested in development mode.
  Python release environments will be supported in the future.
