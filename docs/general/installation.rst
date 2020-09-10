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

Clone autolens workspace & set WORKSPACE environment model ('--depth 1' clones only the most recent branch on the
autolens_workspace, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Finally, run the `welcome.py` script to get started!

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

Clone autolens workspace & set WORKSPACE environment model ('--depth 1' clones only the most recent branch on the
autolens_workspace, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

We will import files from the autolens_workspace as if it were a Python module. To do this in conda, we need to
create a .pth file in our conda enviroments site-packages folder. In your browser or on the command line find your
site packages folder:

.. code-block:: bash

   cd /home/usr/anaconda3/envs/autolens/lib/python3.7/site-packages/

Now create a .pth file via a text editor and put the path to your autolens_workspace in the file and save

NOTE: As shown below, the path in the .pth file points to the directory containing the 'autolens_workspace' folder
but does not contain the 'autolens_workspace' in PYTHONPATH itself!

.. code-block:: bash

   /path/on/your/computer/you/want/to/put/the

Finally, run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py

Forking / Cloning
-----------------

If fork or clone the **PyAutoLens** github repository, note that **PyAutoLens** requires a valid autolens_workspace and
WORKSPACE environment to run (so it can find the necessary confgiuration files).

Therefore, if you fork or clone the **PyAutoLens** repository, you must also clone the
`autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_:

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Once your fork of **PyAutoLens** is setup, I recommend you run the `welcome.py` script in the autolens workspace
for an introduction to **PyAutoLens**.

.. code-block:: bash

   python3 welcome.py

Environment Variables
---------------------

**PyAutoLens** uses an environment variable called WORKSPACE to know where the 'autolens_workspace' folder is located.
This is used to locate config files and output results. It should automatically be detected and set in the `welcome.py`
script, but if something goes wrong you can set it manually using the command:

.. code-block:: bash

    export WORKSPACE=/path/on/your/computer/where/you/cloned/the/autolens_workspace

The autolens_workspace imports modules within the workspace to use them, meaning the path to the workspace must be
included in the PYTHONPATH. Your PYTHONPATH can be manual set using the command below.

NOTE: As shown below, the PYTHONPATH points to the directory containing the 'autolens_workspace' folder but does not
contain the 'autolens_workspace' in PYTHONPATH itself!

.. code-block:: bash

    export PYTHONPATH=/path/on/your/computer/you/want/to/put/the/.

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

Firstly, if your installation via pip raised an error, try instead using conda (or visa versa).

The libraries **numba** and **llvmlite** used for optimizing **PyAutoLens** can cause installation issues. If these
crop up we recommend that you either try using a conda build instead of pip (or visa versa) or try to manually
install these versions of the libraries:

.. code-block:: bash

    pip install llvmlite<=0.32.1
    pip install numba<=0.47.0

If you are still having issues with installation or using **PyAutoLens** in general, please raise an issue on the
`autolens_workspace issues page <https://github.com/Jammy2211/autolens_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).