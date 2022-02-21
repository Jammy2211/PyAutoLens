.. _troubleshooting:

Troubleshooting
===============

LLVMLite / numba
----------------

The libraries ``numba`` and ``llvmlite`` cause known installation issues when installing via ``conda`` or ``pip``.

There are three circumstances where these errors arise:

**1) llvmlite and numba are already installed**

In this case, the installation of autolens raises an exception like the one below:

.. code-block:: bash

   Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which
   files belong to it which would lead to only a partial uninstall.

This means that ``llvmlite`` and ``numba`` are already installed, which you can check as follows:

.. code-block:: bash

   pip show llvmlite
   pip show numba

**PyAutoLens** works fine across many versions of lvvmlite and numba, so you should be ok to circumvent this error by
simply not reinstalling these libraries when you install **PyAutoLens**:

.. code-block:: bash

    pip install autolens --ignore-installed llvmlite numba

**2) llvmlite and numba are not already installed**

In this case, a dependency error will arise where one of these libraries could not be installed. If you are trying to
install via pip, we recommend you instead follow the `installation via conda <https://pyautolens.readthedocs.io/en/latest/installation/conda.html>`_ instructions
which install these libraries as part of the ``conda`` environment.

A common error for installing llvmlite is that a config file is missing:

.. code-block:: bash

   Failed to install - No such file or directory: 'llvm-config': 'llvm-config'

The first solution to try is to upgrade your pip via one of the following commands:

.. code-block:: bash

    pip install --upgrade pip
    pip3 install --upgrade pip

You may then retry the autolens installation:

.. code-block:: bash

    pip install autolens

In the above solution fails, you can manually install the following versions
of ``llvmlite==0.38.0``, ``numba==0.51.1`` and ``numpy==1.22.2`` which are known to work with **PyAutoLens**:

.. code-block:: bash

    pip install llvmlite==0.38.0
    pip install numba==0.53.1 -ignore-installed llvmlite
    pip install numpy==1.22.2

    pip install autolens --ignore-installed llvmlite numba numpy

This may raise warnings, but **PyAutoLens** has been tested with this combination of versions which have had less
installation issues.

**3) The version of numba and numpy clash**

If numba and numpy are not on versions compatible with one another the following error can arise when running autolens:

.. code-block:: bash

    TypeError: expected dtype object, got 'numpy.dtype[float64]'

The easiest solution is to downgrade to ``numpy==1.22.2``:

.. code-block:: bash

    pip install numpy==1.22.2


If you are still facing installation issues please `raise an issue on the GitHub issues page <https://github.com/Jammy2211/PyAutoLens/issues>`_.

Current Working Directory
-------------------------

**PyAutoLens** scripts assume that the ``autolens_workspace`` directory is the Python working directory. This means
that, when you run an example script, you should run it from the ``autolens_workspace`` as follows:

.. code-block:: bash

    cd path/to/autolens_workspace (if you are not already in the autolens_workspace).
    python3 examples/model/beginner/mass_total__source_parametric.py

The reasons for this are so that **PyAutoLens** can:

 - Load configuration settings from config files in the ``autolens_workspace/config`` folder.
 - Load example data from the ``autolens_workspace/dataset`` folder.
 - Output the results of models fits to your hard-disk to the ``autolens/output`` folder.
 - Import modules from the ``autolens_workspace``, for example ``from autolens_workspace.transdimensional import pipelines``.

If you have any errors relating to importing modules, loading data or outputting results it is likely because you
are not running the script with the ``autolens_workspace`` as the working directory!

Matplotlib Backend
------------------

Matplotlib uses the default backend on your computer, as set in the config file:

.. code-block:: bash

    autolens_workspace/config/visualize/general.ini

If unchanged, the backend is set to 'default', meaning it will use the backend automatically set up for Python on
your system.

.. code-block:: bash

    [general]
    backend = default

There have been reports that using the default backend causes crashes when running the test script below (either the
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: bash

    [general]
    backend = TKAgg

Support
-------

If you are still having issues with installation or using **PyAutoLens** in general, please raise an issue on the
`autolens_workspace issues page <https://github.com/Jammy2211/autolens_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).