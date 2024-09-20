.. _overview_2_new_user_guide:

New User Guide
==============

**PyAutoLens** is an extensive piece of software with functionality for doing many different analysis tasks, fitting
different data types and it is used for a variety of different science cases. This means the documentation is quite
extensive, and it may be difficult to find the example script you need.

This page provides a sequential guide for news users on how to begin learning **PyAutoLens**, and can act as a useful
resource for existing users who are looking for how to do a specific task.

Before starting this guide, you should ensure you have installed **PyAutoLens** and downloaded the ``autolens_workspace``
by following the `installation guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_.

Contents
--------

One line summaries of each step in the new user guide is given below, to give you a sense of what you are going to learn:

**1) Workspace:** Read the ``start_here.ipynb`` workspace example for a quick run through of the core API for lensing.
**2) HowToLens?**: Whether you should begin with lectures aimed at inexperienced scientists (e.g. under graduate students).


1) Workspace
------------

You should now have the ``autolens_workspace`` on your computer and see many of the folder and files we'll begin
navigating.

First of all, if you have not already, you should read the `autolens_workspace/start_here.ipynb` notebook,
which provides a run through of the core API for gravitational lensing calculations and lens modeling.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release

2) HowToLens?
-------------

For experienced scientists, the **PyAutoLens** examples will be simple to follow. Concepts surrounding strong lensing may
already be familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToLens** Jupyter Notebook lectures provide exactly this. They are a 3+ chapter guide which thoroughly
take you through the core concepts of strong lensing, teach you the principles of the statistical techniques
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

To complete thoroughly, they'll probably take 2-4 days, so you may want try moving ahead to the examples but can
go back to these lectures if you find them hard to follow.

If this sounds like it suits you, checkout the ``autolens_workspace/notebooks/howtolens`` package now.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/howtolens

3) Configs
----------

The ``autolens_workspace/config`` folder contains numerous .YAML configuration files which customization many
default settings of **PyAutoLens**.

Documentation for all config settings are provided within each config file.

New users should not worry about the majority of configs for now. However, the ``config/visualize`` folder contains
config files which customization ``matplotlib`` visualization, and editing these now will ensure figures and
images display optimally in your Jupyter Notebooks.

All default ``matplotlib`` options are customized via the `mat_wrap.yaml`, `mat_wrap_1d.yaml` and `mat_wrap_2d.yaml` files
in `autolens_workspace/config/visualize/mat_wrap`. For example, if figures display with labels that are too big
or small, you can adjust their default labelsizes by changing the following options:

 - mat_wrap.yaml -> Figure -> figure: -> figsize
 - mat_wrap.yaml -> YLabel -> figure: -> fontsize
 - mat_wrap.yaml -> XLabel -> figure: -> fontsize
 - mat_wrap.yaml -> TickParams -> figure: -> labelsize
 - mat_wrap.yaml -> YTicks -> figure: -> labelsize
 - mat_wrap.yaml -> XTicks -> figure: -> labelsize

The default colormap can be changed from the default to your favour ``matplotlib`` colormap, but adjusting:

 - mat_wrap.yaml -> Cmap -> figure -> cmap

All settings have a ``figure`` and ``subplot`` option, so that single image ``figures`` and a subplot of multiple
figures can be customized independently.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/config
https://github.com/Jammy2211/autolens_workspace/tree/release/config/visualize

4) Dataset Type
---------------

**PyAutoLens** supports multiple different data types, and you as a user likely only require to learn how to use
the software to analyse one type of dataset.

Therefore, you now need to assess which dataset type is relevant to you:

- **Imaging**: CCD imaging data (e.g. from the Hubble Space Telescope or James Webb Space Telescope), in which case
you will go to the ``imaging`` packages in the workspace.

- **Interferometry**: Interferometer data from a submm or radio interferometer (e.g. ALMA or JVLA), in which case
you will go to the ``interferometer`` packages in the workspace.

- **Point Source**: Data of a lensed point source (e.g. a quasar or supernovae) where analysis is in the point source regime,
in which case you will go to the ``point_source`` packages in the workspace.

The scale of your strong lens is also important. Most examples assume that your strong lens is galaxy-scale, meaning that
there is only one lens galaxy and one lensed source. For these systems the Einstein radius is typically below 5.0".

A group scale lens is one where the lens has multiple galaxies responsible for the lensing, and all of their
mass must be modeled for an accurate analysis. Group scale lenses often have multiple sources and Einstein Radii
above 5.0".

If you are modeling group-scale data, you should go to the ``group`` packages in the workspace.

5) API and Units Guides
-----------------------

The ``autolens_workspace/guides`` package has many useful guides, including concise API reference guides (``guides/api``)
and unit conversion guides (``guides/units``).

Quickly navigate to this part of the workspace and skim read the guides quickly. You do not need to understand them in detail now
so don't spend long reading them. T

he purpose of looking at them now is you know they exist and can refer to them if you get stuck using **PyAutoLens**.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/guides
https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/guides/api
https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/guides/units

6) Simulations
--------------

Learning how to simulate your type of data is the best way to understanding how to analyse it.

Therefore, in the ``autolens_workspace/simulators`` folder, find the ``start_here.ipynb`` of your dataset.

For example, if your dataset type is CCD imaging data, you'll read the notebook ``autolens_workspace/simulators/imaging/start_here.ipynb``.

Your **PyAutoLens** use case might only require you to be able to simulate strong lenses, for example if you are
training a neural network. In this case, you can stop the guide and use the tools in the ``simulators`` package
to start doing your science!

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/simulators

7) Modeling
-----------

Having simulated a dataset, you are now ready to learn how to model it.

Therefore, in the ``autolens_workspace/modeling`` folder, find the ``start_here.ipynb`` of your dataset.

For example, if your dataset type is CCD imaging data, you'll read the notebook ``autolens_workspace/modeling/imaging/start_here.ipynb``.

Your **PyAutoLens** use case might only require you to be able to model simulated strong lenses, for example if you are
investigating what lens models can be used to learn from strong lenses. In this case, you can skip the data preparation
step below and go straight to learning about results.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/modeling

8) Data Preparation
-------------------

If you have real observations of strong lenses you want to model, you need to prepare the data so that it
is appropriate for **PyAutoLens**.

This includes reducing the data so the strong lens is in the centre of the image, making sure all units
are defined correctly and reducing extra data products like the Point Spread Function for CCD imaging data.

Therefore, in the ``autolens_workspace/data_preparation`` folder, find the ``start_here.ipynb`` of your dataset.

For example, if your dataset type is CCD imaging data, you'll read the notebook ``autolens_workspace/data_preparation/imaging/start_here.ipynb``.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/data_preparation

9) Results
----------

Lens modeling infers many results, including parameter estimates, posteriors and a Bayesian evidence of the model.
Furthermore, you may wish to inspect the results, the quality of the fit and produce visuals to determine
if you think its a good fit.

Therefore, now read the ``autolens_workspace/*/results/start_here.ipynb`` notebook.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/results

10) Plotting
------------

**PyAutoLens** has an in depth visualizaiton library that allows for high levels of customization via ``matplotlib``.

Plotting has its own dedicated API, which you should become familiar with via the example ``autolens_workspace/*/plot/start_here.ipynb``.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/blob/main/notebooks/plot/start_here.ipynb

11) Features
------------

You now have a comprehensive understanding of the **PyAutoLens** API and how to use it to simulate, model and
plot your data.

**PyAutoLens** has many more features, which may or may not be useful for your science case.

Example notebooks for every feature are provided in the ``autolens_workspace/*/features`` package and a high-level
summary of each feature is provided on the next page of this readthedocs.

What features you need depend on many factors: (i) your science case; (ii) the quality of your data; (iii) how
much time you are willing to invest in learning **PyAutoLens**. We recommend you read the literature in conjunction
with assessing what features are available, and then make an informed decision on what is appropriate for you.

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/features

12) Advanced
------------

The ``autolens_workspace/*/advanced`` folder has numerous advanced examples which only a user experienced with
**PyAutoLens** should use.

These include examples of how to fit multiple datasets simultaneously (e.g. multi-wavelength CCD imaging datasets),
automated pipelines for modeling large lens samples (called the Source, Light and Mass (SLaM) pipelines in the
literature) and a step-by-step guide of the **PyAutoLens** likelihood function.

New users should ignore this folder for now, but note that you may find it has important functionality for
your science research in a couple of months time once you are experienced with **PyAutoLens**!

GitHub Links:

https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/advanced

Wrap Up
-------

After completing this guide, you should be able to use **PyAutoLens** for your science research.

The biggest decisions you'll need to make are what features and functionality your specific science case requires,
which the next readthedocs page gives an overview of to help you decide.