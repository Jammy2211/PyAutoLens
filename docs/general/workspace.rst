.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autolens workspace <https://github.com/Jammy2211/autolens_workspace>`_
when you installed **PyAutoLens**. If you didn't, checkout the
`installation instructions <https://pyautolens.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

Here, we give a brief tour of what is included in the workspace.

Config
------

Here, you'll find the configuration files used by **PyAutoLens** which customize:

    - The default settings used by every *non-linear search*.
    - Visualization, including the backend used by *matplotlib*.
    - The priors and notation configs associated with the light and mass profiles used for lens model-fitting.
    - The behaviour of different (y,x) Cartesian grids used to perform lens calcluations.
    - The general.ini config which customizes other aspects of **PyAutoLens**.

Checkout the `configuration <https://pyautolens.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the readthedocs for a complete description of every configuration file.

Examples
--------

Example scripts describing how to perform different lensing calculations, fit lens models using different non-linear
searches, plot results with **PyAutoLens** and use many other features. These example closely follow and expand on the
`API overview <file:///home/jammy/PycharmProjects/PyAuto/PyAutoLens/docs/_build/overview/lensing.html>`_ found on
this readthedocs.

HowToLens
---------

The **HowToLens** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoLens** model
fitting project and giving illustrations of different statistical methods and techniques.

Checkout the
`tutorials section <file:///home/jammy/PycharmProjects/PyAuto/PyAutoLens/docs/_build/tutorials/howtolens.html>`_ for a
full description of the lectures and online examples of every notebook.

Preprocess
----------

Example scripts and tutorials on how to preprocess CCD imaging and interferometer data before analysing it with
**PyAutoLens**. These includes scripts covering the image formats and units, computing a noise-map, creating the
PSF and setting up masks for the data.

Simulators
----------

Example scripts for simulating strong lens imaging and interferometer datasets, including how to create simulated
images using a variety of lens galaxy mass profiles and source light profiles and how to create datasets representative
of real instruments such as the Hubble Space Telescope, Euclid and ALMA.

Dataset
-------

The folder where data for your model-fitting problem is stored. Example data for the 1D data fitting problem
are provided in the workspace.

Output
------

The folder where the model-fitting results of your model-fitting problem are stored.

Pipelines
---------

Example pipelines for modeling strong lenses using the **PyAutoLens** *Pipeline* functionality, which perform the
model-fit using a sequence of linked non-linear searches which initially perform fast and efficient model fits using
simple lens model parameterization and gradually increase the lens model complexity.

Runners
-------

Example runner scripts which create and use the example pipelines to fit strong lens datasets using a variety of
different model parameterizations and mode-fitting approaches.

Advanced
--------

Contains example scripts for advanced **PyAutoLens** functonality, such as using the *Aggregator* for manipulating
large libraries of model-fitting results, using *hyper mode* to fit models which adapt to the properties of the
dataset being fitted and describing the **SLaM - Source Lens and Mass** framework for modeling strong lenses.