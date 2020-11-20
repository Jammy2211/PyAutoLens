.. _conda:

Installation with conda
=======================

Installation via a conda environment circumvents compatibility issues when installing certain libraries. This guide
assumes you have a working installation of `conda <https://conda.io/miniconda.html>`_.

First, create a conda environment (we name is ``autolens`` to signify it is for the **PyAutoLens** install):

The command below creates this environment with some of the bigger package requirements, the rest will be installed
with **PyAutoFit** via pip:

.. code-block:: bash

    conda create -n autolens astropy numba numpy scikit-image scikit-learn scipy

Activate the conda environment (you will have to do this every time you want to run **PyAutoLens**):

.. code-block:: bash

    conda activate autolens

Install autolens (we assume ``numba`` and ``llvmlite`` were successfully installed when creating the ``conda`` enviroment
above, see `here <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_ for more details):

.. code-block:: bash

    pip install autolens --ignore-installed numba llvmlite

Next, clone the ``autolens workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autolens_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py