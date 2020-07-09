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

**Dynesty** https://github.com/joshspeagle/dynesty

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

Clone autolens workspace & set WORKSPACE enviroment model ('--depth 1' clones only the most recent branch on the
autolens_workspace, reducing the download size):

.. code-block:: bash

    cd /path/where/you/want/autolens_workspace
    git clone https://github.com/Jammy2211/autolens_workspace --depth 1
    export WORKSPACE=/path/to/autolens_workspace/

Set PYTHONPATH to include the autolens_workspace directory:

.. code-block:: bash

    export PYTHONPATH=/path/to/autolens_workspace

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

You can test everything is working by running the example pipeline runner in the autolens_workspace

.. code-block:: bash

    python3 /path/to/autolens_workspace/examples/model/intro/lens_sie__source_sersic.py

Installation with conda
-----------------------

Installation via a conda environment circumvents compatibility issues when installing the optional library
**PyMultiNest**.

First, install `conda <https://conda.io/miniconda.html>`_.

Create a conda environment:

.. code-block:: bash

    >> conda create -n autolens python=3.7 anaconda

Activate the conda environment:

.. code-block:: bash

    conda activate autolens

Install multinest:

.. code-block:: bash

    conda install -c conda-forge multinest

Install autolens:

.. code-block:: bash

    pip install autolens

Clone the autolens workspace & set WORKSPACE environment model:

.. code-block:: bash

    cd /path/where/you/want/autolens_workspace
    git clone https://github.com/Jammy2211/autolens_workspace
    export WORKSPACE=/path/to/autolens_workspace/

Set PYTHONPATH to include the autolens_workspace directory:

.. code-block:: bash

    export PYTHONPATH=/path/to/autolens_workspace/

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


You can test everything is working by running the example pipeline runner in the autolens_workspace

.. code-block:: bash

    python3 /path/to/autolens_workspace/examples/model/intro/lens_sie__source_sersic.py

Forking / Cloning
-----------------

Alternatively, you can fork or clone the **PyAutoLens** github repository. Note that **PyAutoLens** requires a valid
config to run. Therefore, if you fork or clone the **PyAutoLens** repository, you need the
`autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ with the PYTHONPATH and WORKSPACE environment
variables set up as described on the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ repository
or the installation instructions below.

Trouble Shooting
----------------

If you have issues with installation or using **PyAutoFit** in general, please raise an issue on the
`autolens_workspace issues page <https://github.com/Jammy2211/autolens_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).