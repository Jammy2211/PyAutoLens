PyAutoLens
==========

.. |license| image:: https://img.shields.io/github/license/Jammy2211/PyAutoLens    :alt: GitHub license     
   :target: https://github.com/Jammy2211/PyAutoLens/blob/master/LICENSE  

.. |nbsp| unicode:: 0xA0
    :trim:

.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |arXiv| image:: https://img.shields.io/badge/arXiv-1708.07377-blue
    :target: https://arxiv.org/abs/1708.07377

|license| |nbsp| |code-style| |nbsp| |arXiv|

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy appears multiple times.
This is called strong gravitational lensing, & **PyAutoLens** makes it simple to model strong gravitational lenses,
like this one:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/development/imageaxis.png

Getting Started
---------------

To get started checkout our `readthedocs <https://pyautolens.readthedocs.io/>`_,
where you'll find the installation guide, a complete overview of **PyAutoLens**'s features, examples
scripts and the `HowToLens Jupyter notebook tutorials <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_
which introduces new users to **PyAutoLens**.

Installation
------------

**PyAutoLens** requires Python 3.6+ and you can install it via ``pip`` or ``conda`` (see
`this link <https://pyautolens.readthedocs.io/en/latest/installation/conda.html>`_
for ``conda`` instructions).

.. code-block:: bash

    pip install autolens

Next, clone the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_, which includes
**PyAutoLens** configuration files, example scripts and more!

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Finally, run ``welcome.py`` in the ``autolens_workspace`` to get started!

.. code-block:: bash

   python3 welcome.py

If your installation had an error, check the
`troubleshooting section <https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html>`_ on
our readthedocs.

If you would prefer to Fork or Clone the **PyAutoLens** GitHub repo, checkout the
`cloning section <https://pyautolens.readthedocs.io/en/latest/installation/source.html>`_ on our
readthedocs.

API Overview
------------

Lensing calculations are performed in **PyAutoLens** by building a ``Tracer`` object from ``LightProfile``,
``MassProfile`` and ``Galaxy`` objects. Below, we create a simple strong lens system where a redshift 0.5
lens ``Galaxy`` with an ``EllipticalIsothermal`` ``MassProfile`` lenses a background source at redshift 1.0 with an
``EllipticalExponential`` ``LightProfile`` representing a disk.

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt
    from astropy import cosmology as cosmo

    """
    To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
    coordinates are used.
    """

    grid = al.Grid.uniform(
        shape_2d=(50, 50),
        pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
    )

    """The lens galaxy has an EllipticalIsothermal MassProfile and is at redshift 0.5."""

    mass = al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

    """The source galaxy has an EllipticalExponential LightProfile and is at redshift 1.0."""

    disk = al.lp.EllipticalExponential(
        centre=(0.3, 0.2),
        elliptical_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, disk=disk)

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

With **PyAutoLens**, you can begin modeling a lens in just a couple of minutes. The example below demonstrates
a simple analysis which fits the lens galaxy's mass with an ``EllipticalIsothermal`` and the source galaxy's light
with an ``EllipticalSersic``.

.. code-block:: python

    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    """Load Imaging data of the strong lens from the dataset folder of the workspace."""

    imaging = al.Imaging.from_fits(
        image_path="/path/to/dataset/image.fits",
        noise_map_path="/path/to/dataset/noise_map.fits",
        psf_path="/path/to/dataset/psf.fits",
        pixel_scales=0.1,
    )

    """Create a mask for the data, which we setup as a 3.0" circle."""

    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    """
    We model the lens galaxy using an EllipticalIsothermal MassProfile and
    the source galaxy using an EllipticalSersic LightProfile.
    """

    lens_mass_profile = al.mp.EllipticalIsothermal
    source_light_profile = al.lp.EllipticalSersic

    """
    To setup these profiles as model components whose parameters are free & fitted for
    we use the GalaxyModel class.
    """

    lens_galaxy_model = al.GalaxyModel(redshift=0.5, mass=lens_mass_profile)
    source_galaxy_model = al.GalaxyModel(redshift=1.0, disk=source_light_profile)

    """
    To perform the analysis we set up a phase, which takes our galaxy models & fits
    their parameters using a NonLinearSearch (in this case, Dynesty).
    """

    phase = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[example]",n_live_points=50),
        galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
    )

    """
    We pass the imaging dataset and mask to the phase's run function, fitting it
    with the lens model & outputting the results (dynesty samples, visualization,
    etc.) to hard-disk.
    """

    result = phase.run(dataset=imaging, mask=mask)

    """
    The results contain information on the fit, for example the maximum likelihood
    model from the Dynesty parameter space search.
    """

    print(result.samples.max_log_likelihood_instance)

Support
-------

Support for installation issues, help with lens modeling and using **PyAutoLens** is available by
`raising an issue on the autolens_workspace GitHub page <https://github.com/Jammy2211/autolens_workspace/issues>`_. or
joining the **PyAutoLens** `Slack channel <https://pyautolens.slack.com/>`_, where we also provide the latest updates on
**PyAutoLens**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.
