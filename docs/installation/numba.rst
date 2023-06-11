.. _numba:

Numba
=====

Numba (https://numba.pydata.org)  is an optional library which makes **PyAutoLens** run a lot faster, which we strongly
recommend all users have installed.

Certain functionality (pixelized source reconstructions, linear light profiles) is disabled without numba installed
because it will have too slow run-times.

However, some users have experienced difficulties installing numba, meaning they have been unable to try out
PyAutoLens and determine if it the right software for them, before committing more time to installing numba
successfully.

For this reason, numba is an optional installation, so that users can easily experiment and learn
the basic API.

If you do not have numba installed, you can do so via pip as follows:

.. code-block:: bash

    pip install numba


Troubleshooting (Conda)
-----------------------

Numba can be installed as part of your conda environment, with this version of numba used when you make the
conda environment.

If you cannot get numba to install in an existing conda environment you can try creating a new one from fresh,
which is created with numba

To install (or update) numba in conda use the following command:

.. code-block:: bash

    conda install numba

When you create the conda environment run the following command:

.. code-block:: bash

    conda create -n autolens numba astropy scikit-image scikit-learn scipy

You can then follow the standard conda installation instructions give here `<https://pyautolens.readthedocs.io/en/latest/installation/conda.html>`_

Troubleshooting (Numpy)
-----------------------

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