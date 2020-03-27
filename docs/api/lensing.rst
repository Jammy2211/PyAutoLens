.. _api:

API - Lensing
-------------

When two galaxies are aligned perfectly down the line-of-sight to Earth, the background galaxy's light is bent by the
intervening mass of the foreground galaxy. Its light can be fully bent around the foreground galaxy, traversing multiple
paths to the Earth, meaning that the background galaxy is observed multiple times. This by-chance alignment of two
galaxies is called a strong gravitational lens and a two-dimensional scheme of such a system is pictured below.



To begin, lets import autolens and its plot module - these conventions are adopted throughout all tutorials!

.. code-block:: bash

   import autolens as al
   import autolens.plot as aplt

To describe the deflection of light, **PyAutoLens** uses *grid* data structures, which are two-dimensional
Cartesian grids of (y,x) coordinates. Below, we make and plot a uniform Cartesian grid:

.. code-block:: bash

    grid = al.Grid.uniform(
        shape_2d=(50, 50), pixel_scales=0.05 # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

    aplt.grid(grid=grid)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/grid.png
  :width: 400
  :alt: Alternative text

Our aim is to ray-trace this grid's coordinates to calculate how the lens galaxy's mass deflects the source galaxy's
light. We therefore need analytic functions representing a light or mass distribution. For this, **PyAutoLens** uses
*Profile* objects and below we use the elliptical Sersic *LightProfile* object to represent a light distribution:

.. code-block:: bash

    sersic_light_profile = al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=60.0,
        intensity=0.05,
        effective_radius=2.0,
        sersic_index=4.0,
    )

By passing this profile a grid, we can evaluate the light at every coordinate on that grid and create an image
of the light profile:

.. code-block:: bash

    image = sersic_light_profile.profile_image_from_grid(grid=grid)

The plot module provides convinience methods for plotting properties of objects, like the image of a *LightProfile*:

.. code-block:: bash

    aplt.lp(light_profile=sersic_light_profile)

Heres the image of the light profile:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/sersic_light_profile.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** uses *MassProfile* objects to represent different mass distributions and use them perform ray-tracing
calculations. Below we create an elliptical isothermal *MassProfile* and compute its convergence, gravitational
potential and deflection angles on our Cartesian grid:

.. code-block:: bash

    isothermal_mass_profile = al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        axis_ratio=0.8,
        phi=120.0,
        einstein_radius=1.6,
    )

    convergence = isothermal_mass_profile.convergence_from_grid(grid=grid)
    potential = isothermal_mass_profile.potential_from_grid(grid=grid)
    deflections = isothermal_mass_profile.deflections_from_grid(grid=grid)

    aplt.mp.convergence(mass_profile=isothermal_mass_profile)
    aplt.mp.potential(mass_profile=isothermal_mass_profile)
    aplt.mp.deflections(mass_profile=isothermal_mass_profile)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/isothermal_mass_profile_convergence.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/isothermal_mass_profile_potential.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/isothermal_mass_profile_deflections_y.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/isothermal_mass_profile_deflections_x.png
  :width: 400
  :alt: Alternative text

For anyone not familiar with gravitational lensing, don't worry about what the convergence and potential are. The key
thing to note is that the deflection angles describe how a given mass distribution deflections light-rays and this
will allow us create strong lens systems like the one shown above!

In **PyAutoLens**, a *Galaxy* object is a collection of *LightProfile* and *MassProfile* objects at an input redshift.
The code below creates two galaxies representing the lens and source galaxies shown in the strong lensing diagram above.

.. code-block:: bash

   lens_galaxy = al.Galaxy(
       redshift=0.5, light=sersic_light_profile, mass=isothermal_mass_profile
   )

   source_galaxy = al.Galaxy(redshift=1.0, light=another_light_profile)

The geometry of the strong lens system depends on the cosmological distances between the Earth, lens and source and
therefore the redshifts of the lens galaxy and source galaxy objects. By passing these *Galaxy* objects to the
*Tracer* class **PyAutoLens** uses these galaxy redshifts and a cosmological model to create the appropriate strong
lens system.

.. code-block:: bash

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.LambdaCDM
    )

    image = tracer.profile_image_from_grid(grid=grid)

    aplt.tracer.profile_image(tracer=tracer, grid=grid)

When computing the imae from the tracer above, the tracer performs all ray-tracing for the given strong lens system.
This includes using the lens galaxy's mass profile to deflect the light-rays that are traced to the source galaxy.
This makes the image below, where the source's light appears as a multiply imaged and strongly lensed Einstein ring.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/tracer_image.png
  :width: 400
  :alt: Alternative text

To finish, let me emphasise that all of the objects above are extensible. Galaxies can take many profiles and tracers
many galaxies. If these galaxies are at different redshifts a strong lensing system with multiple lens planes will be
created, performing the complex multi-plane ray-tracing calculations necessary.

To finish, lets illustrate this by creating a tracer using 3 galaxies at different redshifts. The mass distribution
of the first lens galaxy has separate components for its stellar mass and dark matter. This forms a system with two
distinct Einstein rings!

.. code-block:: bash

    lens_galaxy_0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lmp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
            intensity=0.5,
            effective_radius=0.3,
            sersic_index=2.5,
            mass_to_light_ratio=0.3,
        ),
        disk=al.lmp.EllipticalExponential(
            centre=(0.0, 0.0),
            axis_ratio=0.6,
            phi=45.0,
            intensity=1.0,
            effective_radius=2.0,
            mass_to_light_ratio=0.2,
        ),
        dark=al.mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
    )

    lens_galaxy_1 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalExponential(
            centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, intensity=3.0, effective_radius=0.1
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, einstein_radius=0.4
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=2.0,
        light=al.lp.EllipticalSersic(
            centre=(0.2, 0.2),
            axis_ratio=0.8,
            phi=60.0,
            intensity=2.0,
            effective_radius=0.1,
            sersic_index=1.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

    aplt.tracer.profile_image(tracer=tracer, grid=grid)

This is what the lens looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/complex_lens.png
  :width: 400
  :alt: Alternative text