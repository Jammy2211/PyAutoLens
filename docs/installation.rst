.. _installation:

Installation
============

We strive to make **PyAutoLens** as easy to install as possible. However, one of our dependencies, **PyMultiNest**,
has proven difficult to install on many machines. We therefore recommend users to install **PyAutoLens** via conda
following the instructions below, which handles the installation of **PyMultiNest**.

Users who wish to install PyAutoLens via pip or fork the respository will need to install **PyMultiNest** manually.

Dependencies
------------

**PyAutoLens** dependencies can be found in the requirements.txt file and they are automatically installed with
**PyAutoLens**. The libraries are:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**PyAutoFit** https://github.com/rhayes777/PyAutoFit

**PyAutoArray** https://github.com/Jammy2211/PyAutoArray

**PyAutoAstro** https://github.com/Jammy2211/PyAutoAstro

**PyMultiNest** http://johannesbuchner.github.io/pymultinest-tutorial/install.html

**emcee** https://github.com/dfm/emcee

**astropy** https://www.astropy.org/

**GetDist** https://getdist.readthedocs.io/en/latest/

**matplotlib** https://matplotlib.org/

**numba** https://github.com/numba/numba.

**numpy** https://numpy.org/

**pyquad** https://github.com/AshKelly/pyquad

**scipy** https://www.scipy.org/

Installation with conda
-----------------------

We recommend installation using a conda environment as this circumvents a number of compatibility issues when installing **PyMultiNest**.

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

    python3 /path/to/autolens_workspace/runners/beginner/no_lens_light/lens_sie__source_inversion.py


Installation with pip
---------------------

Installation is also available via pip, however there are reported issues with
installing **PyMultiNest** that can make installation difficult, see `here <https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes>`_

If **PyMultiNest** has installed correctly you may install **PyAutoLens** via pip as follows.

.. code-block:: bash

    pip install autolens

Clone autolens workspace & set WORKSPACE enviroment model:

.. code-block:: bash

    cd /path/where/you/want/autolens_workspace
    git clone https://github.com/Jammy2211/autolens_workspace
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

You can test everything is working by running the example pipeline runner in the autolens_workspace

.. code-block:: bash

    python3 /path/to/autolens_workspace/runners/beginner/no_lens_light/lens_sie__source_inversion.py

Forking / Cloning
-----------------

Alternatively, you can fork or clone the **PyAutoLens** github repository. Note that **PyAutoLens** requires a valid
config to run. Therefore, if you fork or clone the **PyAutoLens** repository, you need the
`autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ with the PYTHONPATH and WORKSPACE environment
variables set up as described on the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ repository
or the installation instructions below.