.. _installation:

Installation
============

Dependencies
------------

This guide installs **PyAutoLens** with the following dependencies:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**PyAutoFit** https://github.com/rhayes777/PyAutoFit

**PyAutoArray** https://github.com/Jammy2211/PyAutoArray

**PyAutoGalaxy** https://github.com/Jammy2211/PyAutoGalaxy

**PyMultiNest** http://johannesbuchner.github.io/pymultinest-tutorial/install.html

**pyquad** https://github.com/AshKelly/pyquad

**dynesty** https://github.com/joshspeagle/dynesty

**emcee** https://github.com/dfm/emcee

**PySwarms** https://github.com/ljvmiranda921/pyswarms

**astropy** https://www.astropy.org/

**corner.py** https://github.com/dfm/corner.py

**matplotlib** https://matplotlib.org/

**numba** https://github.com/numba/numba

**numpy** https://numpy.org/

**scipy** https://www.scipy.org/

Installation with pip
---------------------

The simplest way to install **PyAutoLens** is via pip:

.. code-block:: bash

    pip install autolens

Clone ``autolens workspace`` (``--depth 1`` clones only the most recent branch on the autolens_workspace, reducing the
download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py

Installation with conda
-----------------------

Installation via a conda environment circumvents compatibility issues when installing certain libraries.

First, install `conda <https://conda.io/miniconda.html>`_.

Create a conda environment:

.. code-block:: bash

    conda create -n autolens python=3.7 anaconda

Activate the conda environment:

.. code-block:: bash

    conda activate autolens

Install autolens:

.. code-block:: bash

    pip install autolens

Clone the ``autolens workspace`` (``--depth 1`` clones only the most recent branch on the autolens_workspace, reducing
the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py

Cloning / Forking
-----------------

You can clone (or fork) the **PyAutoLens** github repository and run it from the source code.

First, clone (or fork) the **PyAutoLens** GitHub repository:

.. code-block:: bash

    git clone https://github.com/Jammy2211/PyAutoLens

Next, install the **PyAutoLens** dependencies via pip:

.. code-block:: bash

   cd PyAutoLens
   pip install -r requirements.txt

Include the **PyAutoLens** source repository in your PYTHONPATH (noting that you must replace the text
``/path/to`` with the path to the **PyAutoLens** directory on your computer):

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoLens

Finally, check the **PyAutoLens** unit tests run and pass (you may need to install pytest via
``pip install pytest``):

.. code-block:: bash

    cd /path/to/PyAutoLens
   python3 -m pytest

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

Trouble Shooting
----------------

Firstly, if your installation via pip raised an error, try instead creating a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_ first and installing it there.
Alternatively, you could try using conda.

If your conda build failed, then try pip!

The libraries **numba** and **llvmlite** used for optimizing **PyAutoLens** have caused known installation issues. To
circumvent this we have added the requirement that the version of llvmlite<=0.32.1 and numba<=0.47.0. However,
if your Python / conda environment already has either library installed with a version above these, it will raise
an error.

However, **PyAutoLens** does work with these newer versions, it is simply that installing them from scratch can raise
an error. There, if you get the following error (or something related or mentioning numba):

.. code-block:: bash

    ERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine
    which files belong to it which would lead to only a partial uninstall

Then install **PyAutoLens** as follows:

.. code-block:: bash

    pip install autolens --ignore-installed llvmlite numba

If you are still having issues with installation or using **PyAutoLens** in general, please raise an issue on the
`autolens_workspace issues page <https://github.com/Jammy2211/autolens_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).