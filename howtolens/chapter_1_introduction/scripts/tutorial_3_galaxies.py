# %%
"""
Tutorial 3: Galaxies
====================

In this example, we'll use the `mass_profiles` and `light_profiles` modules introduced previously, along with the
`galaxy` module to create `Galaxy` objects in **PyAutoLens**. we'll see that:

1) `Galaxy`'s can be made from multiple `LightProfile`'s and `MassProfile`'s.

2) By taking multiple components, the summed image / deflection angle`s of the profiles are computed.

3) `Galaxy`'s have redshifts, defining where they are relative to one another in lensing calculations.
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
Lets use an identical `Grid` to the previous example.
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

# %%
"""
Lets make a `Galaxy` with an `EllipticalSersic` `LightProfile`, by making an `EllipticalSersic` `LightProfile`.and 
passing it to a `Galaxy` object.
"""

# %%
sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

galaxy_with_light_profile = al.Galaxy(redshift=0.5, light=sersic_light_profile)

print(galaxy_with_light_profile)

# %%
"""
In the previous example, we passed `Grid`'s to the `LightProfile` module to compute its image. We can do the exact same 
with `Galaxy`'s to again compute the `Galaxy`'s image.
"""

# %%
galaxy_image = galaxy_with_light_profile.image_from_grid(grid=grid)

print("intensity of `Grid` pixel 0:")
print(galaxy_image.in_2d[0, 0])
print("intensity of `Grid` pixel 1:")
print(galaxy_image.in_2d[0, 1])
print("intensity of `Grid` pixel 2:")
print(galaxy_image.in_2d[0, 2])
print("etc.")

# %%
"""
A galaxy `Plotter` allows us to the plot the image, just like the `Profile` plotters did for a `LightProfile`'s.
"""

# %%
aplt.Galaxy.image(galaxy=galaxy_with_light_profile, grid=grid)

# %%
"""
We can pass galaxies as many profiles as we like. Lets create a `Galaxy` with three `LightProfile`'s.
"""

# %%
light_profile_1 = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.5
)

light_profile_2 = al.lp.SphericalSersic(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=3.0
)

light_profile_3 = al.lp.SphericalSersic(
    centre=(1.0, -1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

galaxy_with_3_light_profiles = al.Galaxy(
    redshift=0.5,
    light_1=light_profile_1,
    light_2=light_profile_2,
    light_3=light_profile_3,
)

print(galaxy_with_3_light_profiles)

# %%
"""
If we plot the galaxy, we see 3 blobs of light!
"""

# %%
aplt.Galaxy.image(galaxy=galaxy_with_3_light_profiles, grid=grid)

# %%
"""
We can also plot each individual `LightProfile` using the `subplot` galaxy plotter.
"""

# %%
aplt.Galaxy.image_subplot(galaxy=galaxy_with_3_light_profiles, grid=grid)

# %%
"""
`MassProfile`'s interact with `Galaxy` objects in the exact same way as `LightProfile`'s. Lets create a `Galaxy` with three 
`SphericalIsothermal` `MassProfile`'s. 
"""

# %%
mass_profile_1 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.0)

mass_profile_3 = al.mp.SphericalIsothermal(centre=(1.0, -1.0), einstein_radius=1.0)

galaxy_with_3_mass_profiles = al.Galaxy(
    redshift=0.5, mass_1=mass_profile_1, mass_2=mass_profile_2, mass_3=mass_profile_3
)

print(galaxy_with_3_mass_profiles)

# %%
"""
We can use a `Galaxy` `Plotter`.to plot these deflection angles. (Deflection angles of `MassProfile`'s add together just 
like the `LightProfile` image`s above)
"""

# %%
aplt.Galaxy.deflections_y(galaxy=galaxy_with_3_mass_profiles, grid=grid)

aplt.Galaxy.deflections_x(galaxy=galaxy_with_3_mass_profiles, grid=grid)

# %%
"""
I wonder what 3 summed convergence maps or potential`s look like ;)
"""

# %%
aplt.Galaxy.convergence(galaxy=galaxy_with_3_mass_profiles, grid=grid)

aplt.Galaxy.potential(galaxy=galaxy_with_3_mass_profiles, grid=grid)

# %%
"""
Finally, a `Galaxy` can take both light and mass profiles, and there is no limit to how many we pass it.
"""

# %%
light_profile_1 = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)

light_profile_2 = al.lp.SphericalSersic(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

light_profile_3 = al.lp.SphericalSersic(
    centre=(2.0, 2.0), intensity=1.0, effective_radius=3.0, sersic_index=3.0
)

light_profile_4 = al.lp.EllipticalSersic(
    centre=(1.0, -1.0),
    elliptical_comps=(0.3, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=1.0,
)

mass_profile_1 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=2.0)

mass_profile_3 = al.mp.SphericalIsothermal(centre=(2.0, 2.0), einstein_radius=3.0)

mass_profile_4 = al.mp.EllipticalIsothermal(
    centre=(1.0, -1.0), elliptical_comps=(0.333333, 0.0), einstein_radius=2.0
)

galaxy_with_many_profiles = al.Galaxy(
    redshift=0.5,
    light_1=light_profile_1,
    light_2=light_profile_2,
    light_3=light_profile_3,
    light_4=light_profile_4,
    mass_1=mass_profile_1,
    mass_2=mass_profile_2,
    mass_3=mass_profile_3,
    mass_4=mass_profile_4,
)

# %%
"""
Suffice to say, the `Galaxy`'s images, convergence, potential and deflections look pretty interesting.
"""

# %%
aplt.Galaxy.image(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.Galaxy.convergence(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.Galaxy.potential(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.Galaxy.deflections_y(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.Galaxy.deflections_x(galaxy=galaxy_with_many_profiles, grid=grid)

# %%
"""
And we`re done. Lets finished by just thinking about one question:

 1) We've learnt we can group profiles into `Galaxy`'s to essentially sum the contribution of each `LightProfile` to 
 the galaxy image`s intensity, or sum the contribution of each `MassProfile` to the convergence, potential and 
 deflection angles. In strong lensing, there are often multiple galaxies next to one another responsible for the 
 lensing - how might we account for this?
"""
