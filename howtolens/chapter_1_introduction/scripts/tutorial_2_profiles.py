# %%
"""
Tutorial 2: Profiles
====================

In this example, we'll create a `Grid` of Cartesian $(y,x)$ coordinates and pass it to the `light_profiles`  module to
create images on this `Grid` and the `mass_profiles` module to create deflection-angle maps on this grid.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

# %%
"""
Lets use the same `Grid` as the previous tutorial (if you skipped that tutorial, I recommend you go back to it!)
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

# %%
"""
Next, lets create a `LightProfile` using the `light_profiles` module, which in **PyAutoLens** is imported as `lp` for 
conciseness. we'll use an `EllipticalSersic` function, which is an analytic function often use to depict galaxies.

(If you are unsure what the `elliptical_comps` are, I`ll give a description of them at the end of the tutorial.)
"""

# %%
sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

# %%
"""
We can print a `Profile` to confirm its parameters.
"""

# %%
print(sersic_light_profile)

# %%
"""
We can pass a `Grid` to a `LightProfile` to compute its intensity at every `Grid` coordinate, using a `_from_grid`
method
"""

# %%
light_image = sersic_light_profile.image_from_grid(grid=grid)

# %%
"""
Much like the `Grid` objects in the previous tutorials, these functions return **PyAutoLens** `Array` objects which are 
accessible in both 2D and 1D.
"""

# %%
print(light_image.shape_2d)
print(light_image.shape_1d)
print(light_image.in_2d[0, 0])
print(light_image.in_1d[0])
print(light_image.in_2d)
print(light_image.in_1d)

# %%
"""
The values computed (e.g. the image) are calculated on the sub-grid and the returned values are stored on the sub-grid, 
which in this case is a 200 x 200 grid.
"""

# %%
print(light_image.sub_shape_2d)
print(light_image.sub_shape_1d)
print(light_image.in_2d[0, 0])
print(light_image[0])

# %%
"""
The benefit of storing all the values on the sub-grid, is that we can now use these values to bin-up the regular grid`s 
shape by taking the mean of each intensity value computed on the sub-grid. This ensures that aliasing effects due to 
computing intensities at only one pixel coordinate inside a full pixel does not degrade the image we create.
"""

# %%
print("intensity of top-left `Grid` pixel:")
print(light_image.in_2d_binned[0, 0])
print(light_image.in_1d_binned[0])

# %%
"""
If you find these 2D and 1D `Array`'s confusing - I wouldn't worry about it. From here on, we'll pretty much just use 
these `Array`'s as they returned to us from functions and not think about if they should be in 2D or 1D. Nevertheless, 
its important that you understand **PyAutoLens** offers these 2D and 1D representations - as it`ll help us later when we 
cover fititng lens data!

We can use a `Profile` `Plotter`.to plot this image.
"""

# %%
aplt.LightProfile.image(light_profile=sersic_light_profile, grid=grid)

# %%
"""
To perform ray-tracing, we need to create a `MassProfile` from the `mass_profiles` module, which we import as `mp` for 
conciseness. 

A `MassProfile` is an analytic function that describes the distribution of mass in a galaxy, and therefore 
can be used to derive its surface-density, gravitational potential and most importantly, its deflection angles. For 
those unfamiliar with lensing, the deflection angles describe how light is bent by the `MassProfile` due to the 
curvature of space-time.
"""

# %%
sis_mass_profile = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)

print(sis_mass_profile)

# %%
"""
Just like above, we can pass a `Grid` to a `MassProfile` to compute its deflection angles. These are returned as the 
_Grid_`s we used in the previous tutorials, so have full access to the 2D / 1D methods and mappings. And, just like 
the image above, they are computed on the sub-grid, so that we can bin up their values to compute more accurate 
deflection angles.

(If you are new to gravitiational lensing, and are unclear on what a `deflection-angle` means or what it is used for, 
then I`ll explain all in tutorial 4 of this chapter. For now, just look at the pretty pictures they make, and worry 
about what they mean in tutorial 4!).
"""

# %%
mass_profile_deflections = sis_mass_profile.deflections_from_grid(grid=grid)

print("deflection-angles of `Grid` sub-pixel 0:")
print(mass_profile_deflections.in_2d[0, 0])
print("deflection-angles of `Grid` sub-pixel 1:")
print(mass_profile_deflections.in_2d[0, 1])
print()
print("deflection-angles of `Grid` pixel 0:")
print(mass_profile_deflections.in_2d_binned[0, 1])
print()
print("deflection-angles of central `Grid` pixels:")
print(mass_profile_deflections.in_2d_binned[49, 49])
print(mass_profile_deflections.in_2d_binned[49, 50])
print(mass_profile_deflections.in_2d_binned[50, 49])
print(mass_profile_deflections.in_2d_binned[50, 50])

# %%
"""
A `Profile` `Plotter`.can plot these deflection angles.

(The black and red lines are the `critical curve` and `caustic` of the `MassProfile`. we'll cover what these are in 
a later tutorial.)
"""

# %%
aplt.MassProfile.deflections_y(mass_profile=sis_mass_profile, grid=grid)
aplt.MassProfile.deflections_x(mass_profile=sis_mass_profile, grid=grid)

# %%
"""
`MassProfile`'s have a range of other properties that are used for lensing calculations, a couple of which we've plotted 
images of below:

   - Convergence: The surface mass density of the `MassProfile` in dimensionless units which are convenient for 
                  lensing calcuations.
   - Potential: The gravitational of the `MassProfile` again in convenient dimensionless units.
   - Magnification: Describes how much brighter each image-pixel appears due to focusing of light rays by the `MassProfile`.

Extracting `Array`'s of these quantities from **PyAutoLens** is exactly the same as for the image and deflection angles above.
"""

# %%
mass_profile_convergence = sis_mass_profile.convergence_from_grid(grid=grid)

mass_profile_potential = sis_mass_profile.potential_from_grid(grid=grid)

mass_profile_magnification = sis_mass_profile.magnification_from_grid(grid=grid)

# %%
"""
Plotting them is equally straight forward.
"""

# %%
aplt.MassProfile.convergence(mass_profile=sis_mass_profile, grid=grid)

aplt.MassProfile.potential(mass_profile=sis_mass_profile, grid=grid)

aplt.MassProfile.magnification(mass_profile=sis_mass_profile, grid=grid)

# %%
"""
Congratulations, you`ve completed your second **PyAutoLens** tutorial! Before moving on to the next one, experiment with 
__PyAutoLens__ by doing the following:

1) Change the `LightProfile`'s effective radius and Sersic index - how does the image`s appearance change?
2) Change the `MassProfile`'s einstein radius - what happens to the deflection angles, potential and convergence?
3) Experiment with different `LightProfile`'s and `MassProfile`'s in the light_profiles and mass_profiles modules. 
In particular, use the `EllipticalIsothermal` `Profile`.to introduce ellipticity into a `MassProfile`.
"""

# %%
"""
___Elliptical Components___

The `elliptical_comps` describe the ellipticity of the geometry of the light and mass profiles. You may be more 
familiar with a coordinate system where the ellipse is defined in terms of:

   - axis_ratio = semi-major axis / semi-minor axis = b/a
   - position angle phi, where phi is in degrees.

We can use the **PyAutoLens** `convert` module to determine the elliptical components from the axis-ratio and phi,
noting that the position angle phi is defined counter-clockwise from the positive x-axis.
"""

# %%
elliptical_comps = al.convert.elliptical_comps_from(axis_ratio=0.5, phi=45.0)

print(elliptical_comps)

# %%
"""

The elliptical components are related to the axis-ratio and position angle phi as follows:

    fac = (1 - axis_ratio) / (1 + axis_ratio)
    
    elliptical_comp[0] = elliptical_comp_y = fac * np.sin(2 * phi)
    elliptical_comp[1] = elliptical_comp_x = fac * np.cos(2 * phi)

The reason we use the elliptical components, instead of the axis-ratio and phi, to define a `Profile` geometry is that it
improves the lens modeling process. What is lens modeling? You`ll find out in chapter 2!
"""
