.. _troubleshooting:

Troubleshooting
===============

Pip Version
-----------

If an error message appears after trying to run ``pip install autolens`` first make sure you are using
the latest version of pip.

.. code-block:: bash

    pip install --upgrade pip
    pip3 install --upgrade pip

NumPy / numba
-------------

The libraries ``numpy`` and ``numba`` can be installed with incompatible versions.

An error message like the one below occurs when importing **PyAutoGalaxy**:

.. code-block:: bash

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/autolens/__init__.py", line 1, in <module>
        from autoarray import preprocess
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/autoarray/__init__.py", line 2, in <module>
        from . import type
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/autoarray/type.py", line 7, in <module>
        from autoarray.mask.mask_1d import Mask1D
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/autoarray/mask/mask_1d.py", line 8, in <module>
        from autoarray.structures.arrays import array_1d_util
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/autoarray/structures/arrays/array_1d_util.py", line 5, in <module>
        from autoarray import numba_util
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/autoarray/numba_util.py", line 2, in <module>
        import numba
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/numba/__init__.py", line 200, in <module>
        _ensure_critical_deps()
      File "/home/jammy/venvs/PyAutoMay2/lib/python3.8/site-packages/numba/__init__.py", line 140, in _ensure_critical_deps
        raise ImportError("Numba needs NumPy 1.21 or less")
    ImportError: Numba needs NumPy 1.21 or less

This can be fixed by reinstalling numpy with the version requested by the error message, in the example
numpy 1.21 (you should replace the ``==1.21.0`` with a different version if requested).

.. code-block:: bash

    pip install numpy==1.21.0

Pip / Conda
-----------

If you are trying to `install via pip <https://pyautolens.readthedocs.io/en/latest/installation/pip.html>`_ but
still haing issues, we recommend you try to `install via conda <https://pyautogalaxy.readthedocs.io/en/latest/installation/conda.html>`_
instead, or visa versa.

Support
-------

If you are still having issues with installation, please raise an issue on the
`autolens_workspace issues page <https://github.com/Jammy2211/autolens_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).

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