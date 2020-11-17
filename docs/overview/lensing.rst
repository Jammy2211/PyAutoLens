.. _lensing:

Lensing
-------

When two galaxies are aligned perfectly down the line-of-sight to Earth, the background galaxy's light is bent by the
intervening mass of the foreground galaxy. Its light can be fully bent around the foreground galaxy, traversing multiple
paths to the Earth, meaning that the background galaxy is observed multiple times. This by-chance alignment of two
galaxies is called a strong gravitational lens and a two-dimensional scheme of such a system is pictured below.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/schematic.jpg
  :width: 600
  :alt: Alternative text

(Image Credit: Image credit: F. Courbin, S. G. Djorgovski, G. Meylan, et al., Caltech / EPFL / WMKO,
https://www.astro.caltech.edu/~george/qsolens/)

**PyAutoLens** is software designed for modeling these strong lensing systems! To begin, lets import autolens and the
plot module.

.. code-block:: bash

   import autolens as al
   import autolens.plot as aplt

To describe the deflection of light due to the lens galaxy's mass, **PyAutoLens** uses ``Grid`` ``data`` structures, which
are two-dimensional Cartesian grids of (y,x) coordinates. Below, we make and plot a uniform Cartesian ``Grid`` (the
pixel-scale describes the conversion from pixel units to arc-seconds):

.. code-block:: bash

    grid = al.Grid.uniform(
        shape_2d=(50, 50), pixel_scales=0.05
    )

    aplt.Grid(grid=grid)

This is what our ``Grid`` looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/grid.png
  :width: 400
  :alt: Alternative text

We will ray-trace this ``Grid``'s coordinates to calculate how a lens galaxy's mass deflects the source galaxy's
light.

For this, we need analytic functions representing light and mass distributions. **PyAutoLens** uses ``Profile`` objects
to do this, for example the ``EllipticalSersic`` ``LightProfile`` object which represents a galaxy's light distribution:

.. code-block:: bash

    sersic_light_profile = al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=(0.1, 0.1),
        intensity=0.05,
        effective_radius=2.0,
        sersic_index=4.0,
    )

By passing this ``Profile`` a ``Grid``, we can evaluate the light at every coordinate on that ``Grid``, creating an image
of the ``LightProfile``:

.. code-block:: bash

    image = sersic_light_profile.image_from_grid(grid=grid)

The plot module provides convenience methods for plotting properties of objects, like the image of a ``LightProfile``:

.. code-block:: bash

    aplt.LightProfile.image(light_profile=sersic_light_profile, grid=grid)

Here's how the image of the light profile appears:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/sersic_light_profile.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** uses ``MassProfile`` objects to represent a galaxy's mass distribution, which is used to perform
ray-tracing calculations. Below we create an ``EllipticalIsothermal`` ``MassProfile`` and compute its convergence,
gravitational potential and deflection angles on our Cartesian ``Grid``:

.. code-block:: bash

    isothermal_mass_profile = al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        elliptical_comps=(0.1, 0.1),
        einstein_radius=1.6,
    )

    convergence = isothermal_mass_profile.convergence_from_grid(grid=grid)
    potential = isothermal_mass_profile.potential_from_grid(grid=grid)
    deflections = isothermal_mass_profile.deflections_from_grid(grid=grid)

    aplt.MassProfile.convergence(mass_profile=isothermal_mass_profile, grid=grid)
    aplt.MassProfile.potential(mass_profile=isothermal_mass_profile, grid=grid)
    aplt.MassProfile.deflections(mass_profile=isothermal_mass_profile, grid=grid)

Heres how the convergence, potential and deflection angles appear:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/isothermal_mass_profile_convergence.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/isothermal_mass_profile_potential.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/isothermal_mass_profile_deflections_y.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/isothermal_mass_profile_deflections_x.png
  :width: 400
  :alt: Alternative text

For anyone not familiar with gravitational lensing, don't worry about what the convergence and potential are for now.
The key thing to note is that the deflection angles describe how a given mass distribution deflects light-rays as they
travel towards us in the Universe.

This allows us create strong lens systems like the one shown above!

In **PyAutoLens**, a ``Galaxy`` object is a collection of ``LightProfile`` and ``MassProfile`` objects at a given redshift.
The code below creates two ``Galaxy``'s representing the lens and source galaxies shown in the strong lensing diagram
above.

.. code-block:: bash

   lens_galaxy = al.Galaxy(
       redshift=0.5, light=sersic_light_profile, mass=isothermal_mass_profile
   )

   source_galaxy = al.Galaxy(redshift=1.0, light=another_light_profile)

The geometry of the strong lens system depends on the cosmological distances between the Earth, lens and source and
therefore the redshifts of the lens galaxy and source galaxy objects. By passing these ``Galaxy`` objects to the
``Tracer`` class **PyAutoLens** uses these galaxy redshifts and a cosmological model to create the appropriate strong
lens system.

.. code-block:: bash

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
    )

    image = tracer.image_from_grid(grid=grid)

    aplt.Tracer.image(tracer=tracer, grid=grid)

When computing the image from the ``Tracer`` above, the ``Tracer`` performs all ray-tracing for the given strong lens
system. This includes using the lens ``Galaxy``'s ``MassProfile`` to deflect the light-rays that are traced to the source
``Galaxy``.

This makes the image below, where the source's light appears as a multiply imaged and strongly lensed Einstein ring.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/tracer_image.png
  :width: 400
  :alt: Alternative text

The **PyAutoLens** API has been designed such that all of the objects introduced above are extensible. ``Galaxy`` objects
can take many ``Profile``'s and ``Tracer`` objects many ``Galaxy``'s. If the ``Galaxy``'s are at different redshifts a strong
lensing system with multiple lens planes will be created, performing complex multi-plane ray-tracing calculations.

To finish, lets create a tracer using 3 ``Galaxy``'s at different redshifts. The ``MassProfile`` of the first lens
galaxy has separate components for its stellar mass and dark matter. This forms a system with two distinct Einstein
rings!

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
            centre=(0.1, 0.1), , elliptical_comps=(0.1, 0.1), intensity=3.0, effective_radius=0.1
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.1, 0.1), , elliptical_comps=(0.1, 0.1), einstein_radius=0.4
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=2.0,
        light=al.lp.EllipticalSersic(
            centre=(0.2, 0.2),
            e1=-0.055555,
            e2=0.096225,
            intensity=2.0,
            effective_radius=0.1,
            sersic_index=1.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

    aplt.Tracer.image(tracer=tracer, grid=grid)

This is what the lens looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/lensing/complex_source.png
  :width: 400
  :alt: Alternative text

If you are unfamilar with strong lensing and not clear what the above quantities or plots mean, fear not, in chapter 1
of the **HowToLens** lecture series we'll take you through strong lensing theory in detail, whilst teaching
you how to use **PyAutoLens** at the same time! Checkout the
`tutorials <https://pyautolens.readthedocs.io/en/latest/tutorials/howtolens.html>`_ section of the readthedocs!