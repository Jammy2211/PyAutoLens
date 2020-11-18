.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autolens workspace <https://github.com/Jammy2211/autolens_workspace>`_
when you installed **PyAutoLens**. If you didn't, checkout the
`installation instructions <https://pyautolens.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

New users should begin by checking out the following parts of the workspace.

Config
------

Here, you'll find the configuration files used by **PyAutoLens** which customize:

    - The default settings used by every `NonLinearSearch`.
    - Visualization, including the backend used by *matplotlib*.
    - The priors and notation configs associated with the light and mass profiles used for lens model-fitting.
    - The behaviour of different (y,x) Cartesian grids used to perform lens calculations.
    - The general.ini config which customizes other aspects of **PyAutoLens**.

Checkout the `configuration <https://pyautolens.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the readthedocs for a complete description of every configuration file.

Examples
--------

Example scripts describing how to perform different lensing calculations, fit lens models using different non-linear
searches, plot results with **PyAutoLens** and use many other features. These example closely follow and expand on the
`API overview <file:///Users/Jammy/Code/PyAuto/PyAutoLens/docs/_build/overview/lensing.html>`_ found on
this readthedocs.

An overview of the example scripts can be found on the readthedocs, starting with the
`lensing API overview <https://pyautolens.readthedocs.io/en/latest/overview/lensing.html>`_

HowToLens
---------

The **HowToLens** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoLens** model
fitting project and giving illustrations of different statistical methods and techniques.

Checkout the
`tutorials section <file:///Users/Jammy/Code/PyAuto/PyAutoLens/docs/_build/tutorials/howtolens.html>`_ for a
full description of the lectures and online examples of every notebook.

Simulators
----------

Example scripts for simulating strong lens imaging and interferometer datasets, including how to create simulated
images using a variety of lens galaxy mass profiles and source light profiles and how to create datasets representative
of real instruments such as the Hubble Space Telescope, Euclid and ALMA.

Dataset
-------

The folder where ``dataset`` for your lens modeling is stored. Example ``dataset`` created using the simulators are
provided with the workspace.

Output
------

The folder where the model-fitting results of your model-fitting problem are stored.

Preprocess
----------

Example scripts and tutorials on how to preprocess CCD imaging and interferometer ``dataset``'s before analysing it with
**PyAutoLens**. These includes scripts covering the image formats and units, computing a noise-map, creating the
PSF and setting up masks for the data.

Transdimensional (Advanced)
---------------------------

Example pipelines for modeling strong lenses using **PyAutoLens**'s transdimensional model-fitting pipelines, which
fit the lens model using a sequence of linked non-linear searches which initially perform fast and efficient model
fits using simple lens model parameterization and gradually increase the lens model complexity.

See `here <https://pyautolens.readthedocs.io/en/latest/advanced/pipelines.html>`_ for an overview.

SLaM (Advanced)
---------------

Example pipelines using the Source, Light and Mass (SLaM) approach to lens modeling.

See `here <https://pyautolens.readthedocs.io/en/latest/advanced/slam.html>`_ for an overview.

Aggregator (Advanced)
---------------------

Manipulate large suites of modeling results via Jupyter notebooks, using **PyAutoFit**'s in-built results database tools.

See `here <https://pyautolens.readthedocs.io/en/latest/advanced/aggregator.html>`_ for an overview.

HPC
---

Example scripts describing how to set up **PyAutoLens** on high performance computers.