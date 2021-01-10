What is PyAutoLens?
===================

**PyAutoLens** is open source software for the analysis and modeling of strong gravitational lenses. Its target audience
is anyone with an interest in strong gravitational lensing, whether that be study the mass structure properties of
the foreground lens galaxy or the magnified properties of the backgrounds source.

An overview of its core features can be found in
the `overview <https://pyautolens.readthedocs.io/en/latest/overview/lensing.html>`_ section of the readthedoc.

Strong Gravitational Lensing
============================

When two galaxies are aligned down the line-of-sight to Earth, light rays from the background galaxy are deflected by the
intervening mass of one or more foreground galaxies. Sometimes its light is fully bent around the foreground galaxies,
traversing multiple paths to the Earth, meaning that the background galaxy is observed multiple times. This alignment
of galaxies is called a strong gravitational lens, an example of which, SLACS1430+4105, is shown in the image
below. The massive elliptical lens galaxy can be seen in the centre of the left panel, surrounded by a multiply
imaged source galaxy whose light has been distorted into an `Einstein ring'. The central and right panels shows
reconstructions of the source's lensed and unlensed light distributions, which are created using a model of the lens
galaxy's mass to trace backwards how the source's light is gravitationally lensed.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/development/imageaxis.png
  :width: 1100
  :alt: Alternative text

Strong lensing provides astronomers with an invaluable tool to study a diverse range of topics, including the
`structure of galaxies <https://academic.oup.com/mnras/article-abstract/489/2/2049/5550746>`_,
`dark matter <https://academic.oup.com/mnras/article/442/3/2017/1048278>`_ and the
`expansion of the Universe <https://academic.oup.com/mnras/article/468/3/2590/3055701>`_.

The past decade has seen the discovery of many hundreds of new strong lenses, however the modeling of a strong lens is historically a
time-intensive process that requires significant human intervention to perform, restricting the scope of any scientific
analysis. In the next decade of order of `one hundred thousand` strong lenses will be discovered by surveys such as
Euclid, the Vera Rubin Observatory and Square Kilometer Array.

How does PyAutoLens Work?
=========================

A strong lens system can be quickly assembled from abstracted objects. A ``Galaxy`` object contains one or
more ``LightProfile``'s and ``MassProfile``'s, which represent its two dimensional distribution of starlight and mass.
``Galaxy``’s lie at a particular distance (redshift) from the observer, and are grouped into ``Plane``'s. Raytracing
through multiple ``Plane``'s is achieved by passing them to a ``Tracer`` with an ``astropy`` Cosmology. By passing
these objects a ``Grid`` strong lens sightlines are computed, including multi-plane ray-tracing. All of these
objects are extensible, making it straightforward to compose highly customized lensing system. The example code
below shows this in action:

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt

    """
    To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
    coordinates are used.
    """

    grid = al.Grid.uniform(
        shape_2d=(50, 50),
        pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
    )

    """The lens galaxy has an EllipticalIsothermal MassProfile and is at redshift 0.5."""

    sie = al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=sie)

    """The source galaxy has an EllipticalExponential LightProfile and is at redshift 1.0."""

    exponential = al.lp.EllipticalExponential(
        centre=(0.3, 0.2),
        elliptical_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, light=exponential)

    """
    We create the strong lens using a Tracer, which uses the galaxies, their redshifts
    and an input cosmology to determine how light is deflected on its path to Earth.
    """

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
    )

    """
    We can use the Grid and Tracer to perform many lensing calculations, for example
    plotting the image of the lensed source.
    """

    aplt.Tracer.image(tracer=tracer, grid=grid)

To perform lens modeling, **PyAutoLens** adopts the probabilistic programming
language `PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_. **PyAutoFit** allows users to compose a
lens model from ``LightProfile``, ``MassProfile`` and ``Galaxy`` objects, customize the model parameterization and
fit it to data via a `NonLinearSearch` (e.g. `dynesty <https://github.com/joshspeagle/dynesty>`_,
`emcee <https://github.com/dfm/emcee>`_ or `PySwarms <https://pyswarms.readthedocs.io/en/latest/>`_). The example
code below shows how to setup and fit a lens model to a dataset:


.. code-block:: python

    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """In this example, we'll fit a simple lens galaxy + source galaxy system."""

    dataset_path = "/path/to/dataset"
    lens_name = "example_lens"

    """Use the dataset path and lens name to load the imaging data."""

    imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/{lens_name}/image.fits",
        noise_map_path=f"{dataset_path}/{lens_name}/noise_map.fits",
        psf_path=f"{dataset_path}/{lens_name}/psf.fits",
        pixel_scales=0.1,
    )

    """Create a mask for the data, which we setup as a 3.0" circle."""

    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    """
    We model our lens galaxy using an EllipticalIsothermal MassProfile &
    our source galaxy as an EllipticalSersic LightProfile.
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
    their parameters using a `NonLinearSearch` (in this case, Dynesty).
    """

    phase = al.PhaseImaging(
        galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
        name="example/phase_example",
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    We pass the imaging `data` and `mask` to the phase, thereby fitting it with the lens
    model & plot the resulting fit.
    """

    result = phase.run(dataset=imaging, mask=mask)
    fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

Getting Started
===============

To get started, users can check-out the **PyAutoLens**'s rich feature-set by going through the `overview` section
of our readthedocs. This illustrates the API for all of **PyAutoLens**'s core features, including how to simulate
strong lens datasets, reconstructing the lensed source galaxy on adaptive pixel-grids and fitting interferometer
datasets.

For new **PyAutoLens** users, we recommend they start by
`installing PyAutoLens <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ (if you haven't
already!), read through the example scripts on
the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ and take the
`HowToLens Jupyter notebook lecture series <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_ on
strong gravitational lensing with **PyAutoLens**.

.. toctree::
   :caption: Overview:
   :maxdepth: 1
   :hidden:

   overview/lensing
   overview/fitting
   overview/modeling
   overview/simulate
   overview/pixelizations
   overview/interferometry

.. toctree::
   :caption: Installation:
   :maxdepth: 1
   :hidden:

   installation/overview
   installation/conda
   installation/pip
   installation/source
   installation/troubleshooting

.. toctree::
   :caption: General:
   :maxdepth: 1
   :hidden:

   general/workspace
   general/configs
   general/papers
   general/citations
   general/credits

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   howtolens/howtolens
   howtolens/chapter_1_introduction/index
   howtolens/chapter_2_lens_modeling/index
   howtolens/chapter_3_pipelines/index
   howtolens/chapter_4_inversions/index
   howtolens/chapter_5_hyper_mode/index

.. toctree::
   :caption: API Reference:
   :maxdepth: 1
   :hidden:

   api/api

.. toctree::
   :caption: Advanced:
   :maxdepth: 1
   :hidden:

   advanced/pipelines
   advanced/slam
   advanced/aggregator
   advanced/hyper_mode
