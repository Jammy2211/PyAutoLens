.. _api:

API Overview
------------

When two galaxies are aligned perfectly down the line-of-sight to Earth, the background galaxy's light is bent by the
intervening mass of the foreground galaxy. Its light can be fully bent around the foreground galaxy, traversing multiple
paths to the Earth, meaning that the background galaxy is observed multiple times. This by-chance alignment of two
galaxies is called a strong gravitational lens and a two-dimensional scheme of such a system is pictured below.



To describe the deflection of light, **PyAutoLens** uses *grid* data structures, which are simply two-dimensional
Cartesian grids of (y,x) coordinates. Below, we make a uniform Cartesian grid:

.. code-block:: bash

    grid = al.grid.uniform(
        shape_2d=(50, 50), pixel_scales=0.05 # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/grid.png
  :width: 400
  :alt: Alternative text

In a moment, we'll ray-trace this grid's coordinates to calculate how the lens galaxy's mass deflects the source
galaxy's light. To do this, we need analytic functions that representing a light or mass distribution. To do this,
**PyAutoLens** uses *Profile* objects and below we use the elliptical Sersic *LightProfile* object to represent a
light distribution:

.. code-block:: bash

    sersic_light_profile = al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=60.0,
        intensity=0.1,
        effective_radius=2.0,
        sersic_index=4.0,
    )

By passing this profile a grid, we can evaluate the light at every coordinate on that grid and create an image
of the light profile:

.. code-block:: bash

    image = sersic_light_profile.profile_image_from_grid(grid=grid)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/sersic_light_profile.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** uses *MassProfile* objects to represent different mass distributions and use them perform ray-tracing
calculations. Below we create an elliptical isothermal *MassProfile* and computes its convergence, gravitational
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


The geometry of the strong lens system depends on the cosmological distances between the Earth, lens and source and therefore
the redshifts of th lens galaxy and source galaxy objects. By passing these *Galaxy* objects to the *Tracer* class
**PyAutoLens** uses these galaxy redshifts to create the appropriate strong lens system.