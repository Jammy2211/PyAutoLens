.. _pip:

Installation with pip
=====================

Install
-------

We strongly recommend that you install **PyAutoLens** in a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_, with the link attached
describing what a virtual environment is and how to create one.

We upgrade pip to ensure certain libraries install:

.. code-block:: bash

    pip install --upgrade pip

The latest version of **PyAutoLens** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

.. code-block:: bash

    pip install autolens==2022.05.02.1

You may get warnings which state something like:

.. code-block:: bash

    ERROR: autoarray 2022.2.14.1 has requirement numpy<=1.22.1, but you'll have numpy 1.22.2 which is incompatible.
    ERROR: numba 0.53.1 has requirement llvmlite<0.37,>=0.36.0rc1, but you'll have llvmlite 0.38.0 which is incompatible.

If you see these messages, they do not mean that the installation has failed and the instructions below will
identify clearly if the installation is a success.

If there are no errors (but only the warnings above) **PyAutoLens** is installed! 

If there is an error check out the `troubleshooting section <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Workspace
---------

Next, clone the ``autolens workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autolens_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

It should be clear that **PyAutoLens** runs without issue. 

If there is an error check out the `troubleshooting section <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Optional
--------

For interferometer analysis there are two optional dependencies that must be installed via the commands:

.. code-block:: bash

    pip install pynufft==2020.2.7
    pip install pylops==1.11.1

**PyAutoLens** will run without these libraries and it is recommended that you only install them if you intend to
do interferometer analysis.

If you run interferometer code a message explaining that you need to install these libraries will be printed, therefore
it is safe not to install them initially.