PyAutoLens
==========

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy appears multiple times.
This is called strong gravitational lensing, & **PyAutoLens** makes it simple to model strong gravitational lenses,
like this one:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/gitimage.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** is based on the following papers:

`Adaptive Semi-linear Inversion of Strong Gravitational Lens Imaging <https://arxiv.org/abs/1412.7436>`_

`AutoLens: Automated Modeling of a Strong Lens's Light, Mass & Source <https://arxiv.org/abs/1708.07377>`_

API Overview
------------

Lensing calculations are performed in **PyAutoLens** by building a *Tracer* object from _LightProfile_, _MassProfile_
and *Galaxy* objects. Below, we create a simple strong lens system where a redshift 0.5 lens galaxy with an isothermal
mass profile lenses a background source at redshift 1.0 with an Exponential light profile.

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt

    """
    To describe the deflection of light grids are used which are two-dimensional Cartesian grids
    of (y,x) coordinates which are deflected by mass profiles.
    """

    grid = al.Grid.uniform(
        shape_2d=(50, 50),
        pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
    )

    """The lens galaxy is at redshift 0.5 and its mass profile is an elliptical Isothermal."""

    sie = al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=sie)

    """The source galaxy is at redshift 1.0, and its light profile is elliptical Exponential."""

    exponential = al.lp.EllipticalExponential(
        centre=(0.3, 0.2),
        elliptical_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, light=exponential)

    """
    We create the strong lens system by performing ray-tracing via a Tracer object, which uses the
    galaxies above, their redshifts and an input cosmology to determine how light is deflected on
    its path to Earth.
    """

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
    )

    """
    We can use the tracer to perform many lensing calculations, for example plotting the
    image of the lensed source.
    """

    aplt.Tracer.image(tracer=tracer, grid=grid)

With **PyAutoLens**, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple
analysis which fits the foreground lens galaxy's mass & the background source galaxy's light.

.. code-block:: python

    import autofit as af
    import autolens as al

    import os

    """In this example, we'll fit a simple lens galaxy + source galaxy system."""

    dataset_path = "{}/../data".format(os.path.dirname(os.path.realpath(__file__)))
    lens_name = "example_lens"

    """Use the relative path to the dataset to load the imaging data."""

    imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/{lens_name}/image.fits",
        noise_map_path=f"{dataset_path}/{lens_name}/noise_map.fits",
        psf_path=f"{dataset_path}/{lens_name}/psf.fits",
        pixel_scales=0.1,
    )

    """Create a mask for the data, which we setup as a 3.0" circle."""

    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    """
    We model our lens galaxy using a mass profile (a singular isothermal ellipsoid) &
    our source galaxy a light profile (an elliptical Sersic).
    """

    lens_mass_profile = al.mp.EllipticalIsothermal
    source_light_profile = al.lp.EllipticalSersic

    """
    To setup our model galaxies, we use the GalaxyModel class, which represents a
    galaxy whose parameters are free & fitted for by PyAutoLens.
    """

    lens_galaxy_model = al.GalaxyModel(redshift=0.5, mass=lens_mass_profile)
    source_galaxy_model = al.GalaxyModel(redshift=1.0, light=source_light_profile)

    """
    To perform the analysis we set up a phase, which takes our galaxy models & fits
    their parameters using a non-linear search (in this case, Dynesty).
    """

    phase = al.PhaseImaging(
        galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
        phase_name="example/phase_example",
        search=af.DynestyStatic(n_live_points=50, sampling_efficiency=0.5),
    )

    """
    We pass the imaging data and mask to the phase, thereby fitting it with the lens
    model & plot the resulting fit.
    """

    result = phase.run(data=imaging, mask=mask)
    al.plot.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

Getting Started
---------------

To get started checkout our `readthedocs <https://pyautolens.readthedocs.io/>`_,
where you'll find our installation guide, a complete overview of **PyAutoLens**'s features, examples scripts and
tutorials and detailed API documentation.

Slack
-----

We're building a **PyAutoLens** community on Slack, so you should contact us on our
`Slack channel <https://pyautolens.slack.com/>`_ before getting started. Here, I will give you the latest updates on
the software & discuss how best to use **PyAutoLens** for your science case.

Unfortunately, Slack is invitation-only, so first send me an `email <https://github.com/Jammy2211>`_ requesting an
invite.
