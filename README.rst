PyAutoLens
==========

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy appears multiple times. This is called strong gravitational lensing, & **PyAutoLens** makes it simple to model strong gravitational lenses, like this one:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/gitimage.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** is based on the following papers:

`Adaptive Semi-linear Inversion of Strong Gravitational Lens Imaging <https://arxiv.org/abs/1412.7436>`_

`AutoLens: Automated Modeling of a Strong Lens's Light, Mass & Source <https://arxiv.org/abs/1708.07377>`_

Example
-------

With **PyAutoLens**, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits the foreground lens galaxy's mass & the background source galaxy's light.

.. code-block:: python

    import autofit as af
    import autolens as al

    import os

    # In this example, we'll fit a simple lens galaxy + source galaxy system.
    dataset_path = '{}/../data/'.format(os.path.dirname(os.path.realpath(__file__)))

    lens_name = 'example_lens'

    # Get the relative path to the data in our workspace & load the imaging data.
    imaging = al.Imaging.from_fits(
        image_path=dataset_path + lens_name + '/image.fits',
        psf_path=dataset_path+lens_name+'/psf.fits',
        noise_map_path=dataset_path+lens_name+'/noise_map.fits',
        pixel_scales=0.1)

    # Create a mask for the data, which we setup as a 3.0" circle.
    mask = al.Mask.circular(shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0)

    # We model our lens galaxy using a mass profile (a singular isothermal ellipsoid) & our source galaxy
    # a light profile (an elliptical Sersic).
    lens_mass_profile = al.mp.EllipticalIsothermal
    source_light_profile = al.lp.EllipticalSersic

    # To setup our model galaxies, we use the GalaxyModel class, which represents a galaxy whose parameters
    # are model & fitted for by PyAutoLens. The galaxies are also assigned redshifts.
    lens_galaxy_model = al.GalaxyModel(redshift=0.5, mass=lens_mass_profile)
    source_galaxy_model = al.GalaxyModel(redshift=1.0, light=source_light_profile)

    # To perform the analysis we set up a phase, which takes our galaxy models & fits their parameters using a non-linear
    # search (in this case, MultiNest).
    phase = al.PhaseImaging(
        galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
        phase_name='example/phase_example', non_linear_class=af.MultiNest)

    # We pass the imaging data and mask to the phase, thereby fitting it with the lens model above & plot the resulting fit.
    result = phase.run(data=imaging, mask=mask)
    al.plot.FitImaging.subplot_fit_imaging(fit=result.most_likely_fit)

Features
--------

**PyAutoLens's** advanced modeling features include:

- **Galaxies** - Use light & mass profiles to make galaxies & perform lensing calculations.
- **Pipelines** - Write automated analysis pipelines to fit complex lens models to large samples of strong lenses.
- **Extended Sources** - Reconstruct complex source galaxy morphologies on a variety of pixel-grids.
- **Adaption** - Adapt the lensing analysis to the features of the observed strong lens imaging.
- **Multi-Plane** - Perform multi-plane ray-tracing & model multi-plane lens systems.
- **Visualization** - Custom visualization libraries for plotting physical lensing quantities & modeling results.

HowToLens
---------

Included with **PyAutoLens** is the **HowToLens** lecture series, which provides an introduction to strong gravitational lens modeling with **PyAutoLens**. It can be found in the workspace & consists of 5 chapters:

- **Introduction** - An introduction to strong gravitational lensing & **PyAutolens**.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines & tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.
- **Hyper-Mode** - How to use **PyAutoLens** advanced modeling features that adapt the model to the strong lens being analysed.

Workspace
---------

**PyAutoLens** comes with a workspace, which can be found `here <https://github.com/Jammy2211/autolens_workspace>`_ & which includes:

- **Aggregator** - Manipulate large suites of modeling results via Jupyter notebooks, using **PyAutoFit**'s in-built results database.
- **API** - Illustrative scripts of the **PyAutoLens** interface, for examples on how to make plots, peform lensing calculations, etc.
- **Config** - Configuration files which customize **PyAutoLens**'s behaviour.
- **Dataset** - Where data is stored, including example datasets distributed with **PyAutoLens**.
- **HowToLens** - The **HowToLens** lecture series.
- **Output** - Where the **PyAutoLens** analysis and visualization are output.
- **Pipelines** - Example pipelines for modeling strong lenses.
- **Preprocess** - Tools to preprocess data before an analysis (e.g. convert units, create masks).
- **Quick Start** - A quick start guide, so you can begin modeling your lenses within hours.
- **Runners** - Scripts for running a **PyAutoLens** pipeline.
- **Simulators** - Scripts for simulating strong lens datasets with **PyAutoLens**.

Slack
-----

We're building a **PyAutoLens** community on Slack, so you should contact us on our `Slack channel <https://pyautolens.slack.com/>`_ before getting started. Here, I will give you the latest updates on the software & discuss how best to use **PyAutoLens** for your science case.

Unfortunately, Slack is invitation-only, so first send me an `email <https://github.com/Jammy2211>`_ requesting an invite.

Documentation & Installation
----------------------------

The PyAutoLens documentation can be found at our `readthedocs  <https://pyautolens.readthedocs.io/en/master>`_, including instructions on `installation <https://pyautolens.readthedocs.io/en/master/installation.html>`_.

Contributing
------------

If you have any suggestions or would like to contribute please get in touch.

Papers
------

A list of published articles using **PyAutoLens** can be found `here <https://pyautolens.readthedocs.io/en/master/papers.html>`_ .

Credits
-------

**Developers**:

`James Nightingale <https://github.com/Jammy2211>`_ - Lead developer & PyAutoLens guru.

`Richard Hayes <https://github.com/rhayes777>`_ - Lead developer & `PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_ guru.

`Ashley Kelly <https://github.com/AshKelly>`_ - Developer of `pyquad <https://github.com/AshKelly/pyquad>`_ for fast deflections computations.

`Amy Etherington <https://github.com/amyetherington>`_ - Magnification, Critical Curves and Caustic Calculations.

`Xiaoyue Cao <https://github.com/caoxiaoyue>`_ - Analytic Ellipitcal Power-Law Deflection Angle Calculations.

Qiuhan He  - NFW Profile Lensing Calculations.

`Nan Li <https://github.com/linan7788626>`_ - Docker integration & support.

**Code Donors**:

`Andrew Robertson <https://github.com/Andrew-Robertson>`_ - Critical curve & caustic calculations.

Mattia Negrello - Visibility models in the uv-plane via direct Fourier transforms.

`Andrea Enia <https://github.com/AndreaEnia>`_ - Voronoi source-plane plotting tools.

`Aristeidis Amvrosiadis <https://github.com/Sketos>`_ - ALMA imaging data loading.

Conor O'Riordan  - Broken Power-Law mass profile.
