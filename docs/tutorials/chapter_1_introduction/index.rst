Chapter 1: Strong Lensing
=========================

In chapter 1, we'll introduce you to strong gravitational lensing and the core PyAutoLens API.

The chapter contains the following 9 tutorials:

Visualization -
Set up and customize Matplotlib visualization in Jupyter notebooks to ensure they display correctly.

Grids -
Using 2D uniform _Grid_ objects of (y,x) coordinates which will be used for lensing calculations.

Profiles -
Creating _LightProfile_ and _MassProfile_ objects which provide analytic descriptions of a galaxy's light and mass.

Galaxies -
Creating _Galaxy_ objects from light and mass profiles and using them to perform lensing calculations.

Planes -
Creating _Planes_ objects, which collect _Galaxy_'s at the same redshift and using them to perform lensing calculations.

Ray Tracing -
Creating _Tracer_ objects from collections of _Galaxy_'s at different reshifts and combining the _Tracer_ with a _Grid_
to perform strong lens ray-tracing.

More Ray Tracing -
Advanced lensing calculations with the _Tracer_

Data -
Loading _Imaging_ data of a strong lens representative of Hubble Space Telescope imaging.

Fitting -
How to fit _Imaging_ data with a strong lens image predicted from a _Tracer_.

Summary -
A Summary of the chapter.

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   tutorial_0_visualization
   tutorial_1_grids
   tutorial_2_profiles
   tutorial_3_galaxies
   tutorial_4_planes
   tutorial_5_ray_tracing
   tutorial_6_more_ray_tracing
   tutorial_7_data
   tutorial_8_fitting
   tutorial_9_summary