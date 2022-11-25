PyAutoLens: Open-Source Strong Lensing
======================================

.. |nbsp| unicode:: 0xA0
    :trim:

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/HEAD

.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02825/status.svg
   :target: https://doi.org/10.21105/joss.02825

.. |arXiv| image:: https://img.shields.io/badge/arXiv-1708.07377-blue
    :target: https://arxiv.org/abs/1708.07377

|binder| |code-style| |JOSS| |arXiv|

`Installation Guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautolens.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=introduction.ipynb>`_ |
`HowToLens <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy appears multiple times.
This is called strong gravitational lensing and **PyAutoLens** makes it simple to model strong gravitational lenses,
like this one:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/master/files/imageaxis.png

Getting Started
---------------

The following links are useful for new starters:

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=introduction.ipynb>`_, where you can try **PyAutoLens** in a web browser (without installation).

- `The PyAutoLens readthedocs <https://pyautolens.readthedocs.io/en/latest>`_, which includes `an installation guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ and an overview of **PyAutoLens**'s core features.

- `The autolens_workspace GitHub repository <https://github.com/Jammy2211/autolens_workspace>`_, which includes example scripts and the `HowToLens Jupyter notebook tutorials <https://github.com/Jammy2211/autolens_workspace/tree/master/notebooks/howtolens>`_ which give new users a step-by-step introduction to **PyAutoLens**.

API Overview
------------

Lensing calculations are performed in **PyAutoLens** by building a ``Tracer`` object from ``LightProfile``,
``MassProfile`` and ``Galaxy`` objects. Below, we create a simple strong lens system where a redshift 0.5
lens ``Galaxy`` with an ``Isothermal`` ``MassProfile`` lenses a background source at redshift 1.0 with an
``Exponential`` ``LightProfile`` representing a disk.

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt
    from astropy import cosmology as cosmo

    """
    To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
    coordinates are used.
    """
    grid = al.Grid2D.uniform(
        shape_native=(50, 50),
        pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
    )

    """
    The lens galaxy has an elliptical isothermal mass profile and is at redshift 0.5.
    """
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

    """
    The source galaxy has an elliptical exponential light profile and is at redshift 1.0.
    """
    disk = al.lp.Exponential(
        centre=(0.3, 0.2),
        ell_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, disk=disk)

    """
    We create the strong lens using a Tracer, which uses the galaxies, their redshifts
    and an input cosmology to determine how light is deflected on its path to Earth.
    """
    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15()
    )

    """
    We can use the Grid2D and Tracer to perform many lensing calculations, for example
    plotting the image of the lensed source.
    """
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

With **PyAutoLens**, you can begin modeling a lens in minutes. The example below demonstrates a simple analysis which
fits the lens galaxy's mass with an ``Isothermal`` and the source galaxy's light with an ``Sersic``.

.. code-block:: python

    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """
    Load Imaging data of the strong lens from the dataset folder of the workspace.
    """
    imaging = al.Imaging.from_fits(
        image_path="/path/to/dataset/image.fits",
        noise_map_path="/path/to/dataset/noise_map.fits",
        psf_path="/path/to/dataset/psf.fits",
        pixel_scales=0.1,
    )

    """
    Create a mask for the imaging data, which we setup as a 3.0" circle, and apply it.
    """
    mask = al.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )
    imaging = imaging.apply_mask(mask=mask)

    """
    We model the lens galaxy using an elliptical isothermal mass profile and
    the source galaxy using an elliptical sersic light profile.
    """
    lens_mass_profile = al.mp.Isothermal
    source_light_profile = al.lp.Sersic

    """
    To setup these profiles as model components whose parameters are free & fitted for
    we set up each Galaxy as a Model and define the model as a Collection of all galaxies.
    """
    lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=lens_mass_profile)
    source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, disk=source_light_profile)
    model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))

    """
    We define the non-linear search used to fit the model to the data (in this case, Dynesty).
    """
    search = af.DynestyStatic(name="search[example]", nlive=50)

    """
    We next set up the `Analysis`, which contains the `log likelihood function` that the
    non-linear search calls to fit the lens model to the data.
    """
    analysis = al.AnalysisImaging(dataset=imaging)

    """
    To perform the model-fit we pass the model and analysis to the search's fit method. This will
    output results (e.g., dynesty samples, model parameters, visualization) to hard-disk.
    """
    result = search.fit(model=model, analysis=analysis)

    """
    The results contain information on the fit, for example the maximum likelihood
    model from the Dynesty parameter space search.
    """
    print(result.samples.max_log_likelihood())

Support
-------

Support for installation issues, help with lens modeling and using **PyAutoLens** is available by
`raising an issue on the GitHub issues page <https://github.com/Jammy2211/PyAutoLens/issues>`_.

We also offer support on the **PyAutoLens** `Slack channel <https://pyautolens.slack.com/>`_, where we also provide the
latest updates on **PyAutoLens**. Slack is invitation-only, so if you'd like to join send
an `email <https://github.com/Jammy2211>`_ requesting an invite.
