What is PyAutoLens?
===================

**PyAutoLens** is open source software for the analysis and modeling of strong gravitational lenses, with its target
audience anyone with an interest in astronomy and cosmology.

The software comes distributed with the **HowToLens** Jupyter notebook lectures, which are written assuming no
previous knowledge about what gravitational lensing is and teach a new user the theory and statistics required to analyse
strong lens data. Checkout `the howtolens section of
the readthedocs <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_.


An overview of **PyAutoLens**'s core features can be found in
the `overview section of the readthedocs <https://pyautolens.readthedocs.io/en/latest/overview/lensing.html>`_.

Strong Gravitational Lensing
============================

When two galaxies are aligned down the line-of-sight to Earth, light rays from the background galaxy are deflected by the
intervening mass of one or more foreground galaxies. Sometimes its light is fully deflected around the foreground galaxies,
traversing multiple paths to the Earth, meaning that the background galaxy is observed multiple times. This alignment
of galaxies is called a strong gravitational lens, an example of which, SLACS1430+4105, is shown in the image
below. The massive elliptical lens galaxy can be seen in the centre of the left panel, surrounded by a multiply
imaged source galaxy whose light has been distorted into an 'Einstein ring'. The central and right panels shows
reconstructions of the source's lensed and unlensed light distributions, which are created using a model of the lens
galaxy's mass to trace backwards how the source's light is gravitationally lensed.

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/master/files/imageaxis.png?raw=true

Strong lensing provides astronomers with an invaluable tool to study a diverse range of topics, including the
`structure of galaxies <https://academic.oup.com/mnras/article-abstract/489/2/2049/5550746>`_,
`dark matter <https://academic.oup.com/mnras/article/442/3/2017/1048278>`_ and the
`expansion of the Universe <https://academic.oup.com/mnras/article/468/3/2590/3055701>`_.

The past decade has seen the discovery of many hundreds of new strong lenses, however the modeling of a strong lens is
historically a time-intensive process that requires significant human intervention to perform, restricting the scope of
any scientific analysis. In the next decade of order of `one hundred thousand` strong lenses will be discovered by
surveys such as Euclid, the Vera Rubin Observatory and Square Kilometer Array.

The goal of **PyAutoLens** is to enable fully automated strong lens analysis, such that these large samples of strong
lenses can be exploited to their fullest.

How does PyAutoLens Work?
=========================

A strong lens system can be quickly assembled from abstracted objects. A ``Galaxy`` object contains one or
more ``LightProfile``'s and ``MassProfile``'s, which represent its two dimensional distribution of starlight and mass.
``Galaxy``’s lie at a particular distance (redshift) from the observer, and are grouped into ``Plane``'s. Raytracing
through multiple ``Plane``'s is achieved by passing them to a ``Tracer`` with an ``astropy`` Cosmology. By passing
these objects a ``Grid2D`` strong lens sightlines are computed, including multi-plane ray-tracing. All of these
objects are extensible, making it straightforward to compose highly customized lensing system. The example code
below shows this in action:

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt

    """
    To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
    coordinates are used.
    """

    grid_2d = al.Grid2D.uniform(
        shape_native=(50, 50),
        pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
    )

    """
    The lens galaxy has an elliptical isothermal mass profile and is at redshift 0.5.
    """

    sie = al.mp.EllIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=sie)

    """The source galaxy has an elliptical exponential light profile and is at redshift 1.0."""

    exponential = al.lp.EllExponential(
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
        galaxies=[lens_galaxy, source_galaxy], cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15()
    )

    """
    We can use the Grid2D and Tracer to perform many lensing calculations, for example
    plotting the image of the lensed source.
    """

    aplt.Tracer.image(tracer=tracer, grid=grid_2d)

To perform lens modeling, **PyAutoLens** adopts the probabilistic programming
language `PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_. **PyAutoFit** allows users to compose a
lens model from ``LightProfile``, ``MassProfile`` and ``Galaxy`` objects, customize the model parameterization and
fit it to data via a non-linear search (e.g. `dynesty <https://github.com/joshspeagle/dynesty>`_,
`emcee <https://github.com/dfm/emcee>`_ or `PySwarms <https://pyswarms.readthedocs.io/en/latest/>`_). The example
code below shows how to setup and fit a lens model to a dataset:


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
    Create a mask for the data, which we setup as a 3.0" circle.
    """
    mask = al.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )
    imaging = imaging.apply_mask(mask=mask_2d)

    """
    We model the lens galaxy using an elliptical isothermal mass profile and
    the source galaxy using an elliptical sersic light profile.
    """
    lens_mass_profile = al.mp.EllIsothermal
    source_light_profile = al.lp.EllSersic

    """
    To setup these profiles as model components whose parameters are free & fitted for
    we set up each Galaxy as a Model and define the model as a Collection of all galaxies.
    """
    lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=lens_mass_profile)
    source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, disk=source_light_profile)
    model = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)

    """
    We define the non-linear search used to fit the model to the data (in this case, Dynesty).
    """
    search = af.DynestyStatic(name="search[example]", n_live_points=50)

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
    print(result.samples.max_log_likelihood_instance)

Getting Started
===============

To get started, users can check-out the **PyAutoLens**'s rich feature-set by going through the ``overview`` section
of our readthedocs. This illustrates the API for all of **PyAutoLens**'s core features, including how to simulate
strong lens datasets, reconstructing the lensed source galaxy on adaptive pixel-grids and fitting interferometer
datasets.

For new **PyAutoLens** users, we recommend they start by
`installing PyAutoLens <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ (if you haven't
already!), read through the ``introduction.ipynb`` notebook on
the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ and take the
`HowToLens Jupyter notebook lecture series <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_ on
strong gravitational lensing with **PyAutoLens**.

.. toctree::
   :caption: Overview:
   :maxdepth: 1
   :hidden:

   overview/overview_1_lensing
   overview/overview_2_fitting
   overview/overview_3_modeling
   overview/overview_4_simulate
   overview/overview_5_pixelizations
   overview/overview_6_interferometry
   overview/overview_7_multi_wavelength
   overview/overview_8_point_sources
   overview/overview_9_groups

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
   general/likelihood_function
   general/citations
   general/papers
   general/credits

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   howtolens/howtolens
   howtolens/chapter_1_introduction
   howtolens/chapter_2_lens_modeling
   howtolens/chapter_3_search_chaining
   howtolens/chapter_4_pixelizations
   howtolens/chapter_5_hyper_mode
   howtolens/chapter_optional

.. toctree::
   :caption: API Reference:
   :maxdepth: 1
   :hidden:

   api/api

.. toctree::
   :caption: Advanced:
   :maxdepth: 1
   :hidden:

   advanced/database
   advanced/chaining
   advanced/slam
   advanced/graphical
   advanced/hyper_mode
