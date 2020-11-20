.. _troubleshooting:

Troubleshooting
===============

LLVMLite / numba
----------------

The libraries **numba** and **llvmlite** cause known installation issues when installing via ``conda`` or ``pip``.
Newer versions of llvmlite (> 0.32.1) raise an error during install due to a missing configuration file. To circumvent
this the **PyAutoLens** requirements file requires that ``llvmlite<=0.32.1`` and ``numba<=0.47.0``.

However, if your conda / Python environment already has either library installed with a version above these, it may
raise an error along the lines of:

.. code-block:: bash

   Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which
   files belong to it which would lead to only a partial uninstall.

This is why in the `installation via conda <https://pyautolens.readthedocs.io/en/latest/installation/conda.html>`_
instructions we installed these libraries as part of the ``conda`` environment and ignored them when we installed
``autolens`` via pip.

**PyAutoLens** works fine with these newer versions, so if your environment already has ``llvmlite`` and ``numba``
installed you can circumvent this error by simply not installing them when you install **PyAutoLens**:

.. code-block:: bash

    pip install autolens --ignore-installed llvmlite numba


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