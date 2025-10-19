.. _overview_1_start_here:

Start Here
==========

**PyAutoLens** is software for analysing strong gravitational lenses, an astrophysical phenomenon where a galaxy
appears multiple times because its light is bent by the gravitational field of an intervening foreground lens galaxy.

It uses JAX to accelerate lensing calculations, with the example code below all running significantly faster on GPU.

Here is a schematic of a strong gravitational lens:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/schematic.jpg
  :width: 600
  :alt: Alternative text

**Credit: F. Courbin, S. G. Djorgovski, G. Meylan, et al., Caltech / EPFL / WMKO**
https://www.astro.caltech.edu/~george/qsolens/

This notebook gives an overview of **PyAutoLens**'s features and API.

Imports
-------

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.

.. code:: python

    import autolens as al
    import autolens.plot as aplt

    import matplotlib.pyplot as plt
    from os import path

Lets illustrate a simple gravitational lensing calculation, creating an an image of a lensed galaxy using a
light profile and mass profile.

Grid
----

The emission of light from a source galaxy, which is gravitationally lensed around the lens galaxy, is described 
using the ``Grid2D`` data structure, which is two-dimensional Cartesian grids of (y,x) coordinates.

We make and plot a uniform Cartesian grid:

.. code:: python

    grid = al.Grid2D.uniform(
        shape_native=(150, 150),  # The [pixels x pixels] shape of the grid in 2D.
        pixel_scales=0.05,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

    grid_plotter = aplt.Grid2DPlotter(grid=grid)
    grid_plotter.figure_2d()

The ``Grid2D`` looks like this:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/0_grid.png
  :width: 600
  :alt: Alternative text

Light Profiles
--------------

Our aim is to create an image of the source galaxy after its light has been deflected by the mass of the foreground
lens galaxy. We therefore need to ray-trace the ``Grid2D``'s coordinates from the 'image-plane' to the 'source-plane'.

This uses analytic functions representing a galaxy's light and mass distributions, referred to as ``LightProfile`` and
``MassProfile`` objects.

A common light profile in Astronomy is the elliptical Sersic, which we create an instance of below:

.. code:: python

    sersic_light_profile = al.lp.Sersic(
        centre=(0.0, 0.0),  # The light profile centre [units of arc-seconds].
        ell_comps=(
            0.2,
            0.1,
        ),  # The light profile elliptical components [can be converted to axis-ratio and position angle].
        intensity=0.005,  # The overall intensity normalisation [units arbitrary and are matched to the data].
        effective_radius=2.0,  # The effective radius containing half the profile's total luminosity [units of arc-seconds].
        sersic_index=4.0,  # Describes the profile's shape [higher value -> more concentrated profile].
    )


By passing the light profile the ``grid``, we evaluate the light emitted at every (y,x) coordinate and therefore create 
an image of the Sersic light profile.

.. code:: python

    image = sersic_light_profile.image_2d_from(grid=grid)

Plotting
--------

The **PyAutoLens** in-built plot module provides methods for plotting objects and their properties, like the image of
a light profile we just created.

By using a ``LightProfilePlotter`` to plot the light profile's image, the figured is improved. 

Its axis units are scaled to arc-seconds, a color-bar is added, its given a descriptive labels, etc.

The plot module is highly customizable and designed to make it straight forward to create clean and informative figures
for fits to large datasets.

.. code:: python

    light_profile_plotter = aplt.LightProfilePlotter(
        light_profile=sersic_light_profile, grid=grid
    )
    light_profile_plotter.figures_2d(image=True)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/1_image_2d.png
  :width: 600
  :alt: Alternative text

Mass Profiles
-------------

PyAutoLens uses MassProfile objects to represent a galaxy’s mass distribution and perform ray-tracing calculations.

Below we create an elliptical isothermal MassProfile and compute its deflection angles on our Cartesian grid, where 
the deflection angles describe how the lens galaxy’s mass bends the source’s light:

.. code:: python

    isothermal_mass_profile = al.mp.Isothermal(
        centre=(0.0, 0.0),  # The mass profile centre [units of arc-seconds].
        ell_comps=(
            0.1,
            0.0,
        ),  # The mass profile elliptical components [can be converted to axis-ratio and position angle].
        einstein_radius=1.6,  # The Einstein radius [units of arc-seconds].
    )

    deflections = isothermal_mass_profile.deflections_yx_2d_from(grid=grid)

The deflection angles are easily plotted using the **PyAutoLens** plot module.

(Many other lensing quantities are also easily plotted, for example the ``convergence`` and ``potential``).

.. code:: python

    mass_profile_plotter = aplt.MassProfilePlotter(
        mass_profile=isothermal_mass_profile, grid=grid
    )
    mass_profile_plotter.figures_2d(
        deflections_y=True,
        deflections_x=True,
        # convergence=True,
        # potential=True
    )

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/2_deflections_y_2d.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/3_deflections_x_2d.png
  :width: 600
  :alt: Alternative text

Galaxy
------

A ``Galaxy`` object is a collection of light profiles at a specific redshift.

This object is highly extensible and is what ultimately allows us to fit complex models to strong lens images.

We create two galaxies representing the lens and source galaxies shown in the strong lensing diagram above.

.. code:: python

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=sersic_light_profile,  # The foreground lens's light is typically observed in a strong lens.
        mass=isothermal_mass_profile,  # Its mass is what causes the strong lensing effect.
    )

    source_light_profile = al.lp.Exponential(
        centre=(
            0.3,
            0.2,
        ),  # The source galaxy's light is observed, appearing as multiple images around the lens galaxy.
        ell_comps=(
            0.1,
            0.0,
        ),  # However, the mass of the source does not impact the strong lensing effect.
        intensity=0.1,  # and is not included.
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, light=source_light_profile)

The ``GalaxyPlotter`` object plots properties of the lens and source galaxies.

.. code:: python

    lens_galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=grid)
    lens_galaxy_plotter.figures_2d(image=True, deflections_y=True, deflections_x=True)

    source_galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_galaxy, grid=grid)
    source_galaxy_plotter.figures_2d(image=True)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/4_image_2d.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/7_image_2d.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/5_deflections_y_2d.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/6_deflections_x_2d.png
  :width: 400
  :alt: Alternative text

One example of the plotter's customizability is the ability to plot the individual light profiles of the galaxy
on a subplot.

.. code:: python

    lens_galaxy_plotter.subplot_of_light_profiles(image=True)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/8_subplot_image.png
  :width: 600
  :alt: Alternative text

Tracer
------

The ``Tracer`` object is the most important object in **PyAutoLens**. 

It is a collection of galaxies at different redshifts (often referred to as planes). 

It uses these galaxies to perform ray-tracing, using the mass profiles of the galaxies to bend the light of the source
galaxy(s) into the multiple images we observe in a strong lens system. 

This is shown below, where the image of the tracer shows a distinct Einstein ring of the source galaxy.

.. code:: python

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.Planck15())

    image = tracer.image_2d_from(grid=grid)

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/9_image_2d.png
  :width: 600
  :alt: Alternative text

Units
-----

The units used throughout the strong lensing literature vary, therefore lets quickly describe the units used in
**PyAutoLens**.

The ``Tracer`` object and all mass profiles describe their quantities in terms of angles, which are defined in units
of arc-seconds. To convert these to physical units (e.g. kiloparsecs), we use the redshift of the lens and source
galaxies and an input cosmology. A run through of all normal unit conversions is given in guides in the workspace
that are discussed later.

The use of angles in arc-seconds has an important property, it means that for a two-plane strong lens system 
(e.g. a lens galaxy at one redshift and source galaxy at another redshift) lensing calculations are independent of
the galaxies' redshifts and the input cosmology. This has a number of benefits, for example it makes it straight
forward to compare the lensing properties of different strong lens systems even when the redshifts of the galaxies
are unknown.

Multi-plane lensing is when there are more than two planes. The tracer fully supports this, if you input 3+ galaxies
with different redshifts into the tracer it will use their redshifts and its cosmology to perform multi-plane lensing
calculations that depend on them.

Extensibility
-------------

All of the objects we've introduced so far are highly extensible, for example a tracer can be made of many galaxies, a 
galaxy can be made up of any number of light profiles and many galaxy objects can be combined into a galaxies object.

Below, wecreate a ``Tracer`` with 3 galaxies at 3 different redshifts, forming a system with two distinct Einstein
rings! The mass distribution of the first galaxy has separate components for its stellar mass and dark matter, where
the stellar components use a ``LightAndMassProfile`` via the ``lmp`` module.

.. code:: python

    lens_galaxy_0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lmp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.05),
            intensity=0.5,
            effective_radius=0.3,
            sersic_index=3.5,
            mass_to_light_ratio=0.6,
        ),
        disk=al.lmp.Exponential(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.1),
            intensity=1.0,
            effective_radius=2.0,
            mass_to_light_ratio=0.2,
        ),
        dark=al.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
    )

    lens_galaxy_1 = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Exponential(
            centre=(0.00, 0.00),
            ell_comps=(0.05, 0.05),
            intensity=1.2,
            effective_radius=0.1,
        ),
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.05, 0.05), einstein_radius=0.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=2.0,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.111111),
            intensity=0.7,
            effective_radius=0.1,
            sersic_index=1.5,
        ),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_1/10_image_2d.png
  :width: 600
  :alt: Alternative text


Simulator
---------

Let’s now switch gears and simulate our own strong lens imaging. This is a great way to:

- Practice lens modeling before using real data.
- Build large training sets (e.g. for machine learning).
- Test lensing theory in a controlled environment.

In this example. we simulate “perfect” images without telescope effects. This means no blurring
from a PSF and no noise — just the raw light from galaxies and deflections from gravity.

In fact, this exactly what the image above is: a perfect image of a double Einstein ring system. The only
thing we need to do then, is output it to a .fits file so we can load it elsewhere.

.. code:: python

    al.output_to_fits(
        values=image.native,
        file_path=Path("image.fits"),
        overwrite=True,
    )

Samples
-------

Often we want to simulate *many* strong lenses — for example, to train a neural network
or to explore population-level statistics.

This uses the model composition API to define the distribution of the light and mass profiles
of the lens and source galaxies we draw from. The model composition is a little too complex for
the first example, thus we use a helper function to create a simple lens and source model.

We then generate 3 lenses for speed, and plot their images so you can see the variety of lenses
we create.

If you want to simulate lenses yourself (e.g. for training a neural network), checkout the
`autolens_workspace/simulators` package for a full description of how to do this and customize
the simulated lenses to your science.

The images below are perfect lenses of strong lenses, the next examples will show us how to
instead output realistic observations of strong lenses (e.g. CCD imaging, interferometer data, etc).

.. code:: python

    lens_model, source_model = al.model_util.simulator_start_here_model_from()

    total_datasets = 3

    for sample_index in range(total_datasets):

        lens_galaxy = lens_model.random_instance()
        source_galaxy = source_model.random_instance()

        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

        tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
        tracer_plotter.figures_2d(image=True)

Lens Modeling
-------------

Lens modeling is the process where given data on a strong lens, we fit the data with a model to infer the properties
of the lens and source galaxies.

The animation below shows a slide-show of the lens modeling procedure. Many lens models are fitted to the data over
and over, gradually improving the quality of the fit to the data and looking more and more like the observed image.

We can see that initial models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true
  :width: 600

![Lens Modeling Animation](https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true "model")

**Credit: Amy Etherington**

**PyAutoLens**'s main goal is to make lens modeling **simple** for everyone, **scale** to large datasets
and **run very fast** thanks to GPU acceleration via JAX.

Wrap Up
-------

We have now completed the API overview of **PyAutoLens**, including a brief introduction to the core API for
creating galaxies, simulating data and performing lens modeling.

The next overview describes how a new user should navigate the **PyAutoLens** workspace, which contains many examples
and tutorials, in order to get up and running with the software.