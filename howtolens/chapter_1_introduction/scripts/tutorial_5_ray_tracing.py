# %%
"""
Tutorial 5: Ray Tracing
=======================

In the last tutorial, our use of `Plane`'s was a bit clunky. We manually had to input `Grid`'s to trace them, and keep
track of which `Grid`'s were the image-plane`s and which were the source planes. It was easy to make mistakes!

Fotunately, in **PyAutoLens**, you won't actually spend much hands-on time with the `Plane` objects. Instead, you`ll
primarily use the `ray-tracing` module, which we'll cover in this example. Lets look at how easy it is to setup the
same lens-plane + source-plane strong lens configuration as the previous tutorial, but with a lot less lines of code!
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
Let use the same `Grid` we've all grown to know and love by now!
"""

# %%
image_plane_grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

# %%
"""
For our lens galaxy, we'll use the same SIS `MassProfile` as before.
"""

# %%
sis_mass_profile = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)

lens_galaxy = al.Galaxy(redshift=0.5, mass=sis_mass_profile)

print(lens_galaxy)

# %%
"""
And for our source galaxy, the same `SphericalSersic` `LightProfile`
"""

# %%
sersic_light_profile = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)

source_galaxy = al.Galaxy(redshift=1.0, light=sersic_light_profile)

print(source_galaxy)

# %%
"""
Now, lets use the lens and source galaxies to ray-trace our `Grid`, using a `Tracer` from the ray-tracing module. 
When we pass our galaxies into the `Tracer` below, the following happens:

1) The galaxies are ordered in ascending redshift.
2) Planes are created at every one of these redshifts, with the galaxies at those redshifts associated with those planes.
"""

# %%
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# %%
"""
This `Tracer` is composed of a list of planes, in this case two `Plane`'s (the image and source plane).
"""

# %%
print(tracer.planes)

# %%
"""
We can access these using the `image-plane` and `source-plane` attributes.
"""

# %%
print("Image Plane:")
print(tracer.planes[0])
print(tracer.image_plane)
print()
print("Source Plane:")
print(tracer.planes[1])
print(tracer.source_plane)

# %%
"""
The most convenient part of the `Tracer` is we can use it to create fully `ray-traced` images, without manually 
setting up the `Plane`'s to do this. The function below does the following

1) Using the lens-total mass distribution, the deflection angle of every image-plane `Grid` coordinate is computed.
2) These deflection angles are used to trace every image-plane coordinate to a source-plane coordinate.
3) The light of each traced source-plane coordinate is evaluated using the source-plane `Galaxy`'s `LightProfile`.
"""

# %%
traced_image = tracer.image_from_grid(grid=image_plane_grid)
print("traced image pixel 1")
print(traced_image.in_2d[0, 0])
print("traced image pixel 2")
print(traced_image.in_2d[0, 1])
print("traced image pixel 3")
print(traced_image.in_2d[0, 2])

# %%
"""
This image appears as the Einstein ring we saw in the previous tutorial.
"""

# %%
aplt.Tracer.image(tracer=tracer, grid=image_plane_grid)

# %%
"""
We can also use the `Tracer` to compute the traced `Grid` of every plane, instead of getting the traced image itself:
"""

# %%
traced_grids = tracer.traced_grids_of_planes_from_grid(grid=image_plane_grid)

# %%
"""
And the source-plane`s `Grid` has been deflected.
"""

# %%
print("grid source-plane coordinate 1")
print(traced_grids[1].in_2d[0, 0])
print("grid source-plane coordinate 2")
print(traced_grids[1].in_2d[0, 1])
print("grid source-plane coordinate 3")
print(traced_grids[1].in_2d[0, 2])

# %%
"""
We can use the plane_plotter to plot these grids, like we did before.
"""

# %%
plotter = aplt.Plotter(labels=aplt.Labels(title="Image-plane Grid"))

aplt.Plane.plane_grid(plane=tracer.image_plane, grid=traced_grids[0], plotter=plotter)

plotter = aplt.Plotter(labels=aplt.Labels(title="Source-plane Grid"))

aplt.Plane.plane_grid(plane=tracer.source_plane, grid=traced_grids[1], plotter=plotter)

aplt.Plane.plane_grid(
    plane=tracer.source_plane,
    grid=traced_grids[1],
    axis_limits=[-0.1, 0.1, -0.1, 0.1],
    plotter=plotter,
)

# %%
"""
__PyAutoLens__ has tools for plotting a `Tracer`. A ray-tracing subplot plots the following:

1) The image, computed by tracing the source-`Galaxy`'s light `forwards` through the `Tracer`.

2) The source-plane image, showing the source-`Galaxy`'s true appearance (i.e. if it were not lensed).

3) The image-plane convergence, computed using the lens `Galaxy`'s total mass distribution.

4) The image-plane gravitational potential, computed using the lens `Galaxy`'s total mass distribution.

5) The image-plane deflection angles, computed using the lens `Galaxy`'s total mass distribution.
"""

# %%
aplt.Tracer.subplot_tracer(tracer=tracer, grid=image_plane_grid)

# %%
"""
Just like for a plane, these quantities attributes can be computed by passing a `Grid` (converted to 2D ndarrays
the same dimensions as our input grid!).
"""

# %%
convergence = tracer.convergence_from_grid(grid=image_plane_grid)

print("Tracer - Convergence - `Grid` coordinate 1:")
print(convergence.in_2d[0, 0])
print("Tracer - Convergence - `Grid` coordinate 2:")
print(convergence.in_2d[0, 1])
print("Tracer - Convergence - `Grid` coordinate 3:")
print(convergence.in_2d[0, 2])
print("Tracer - Convergence - `Grid` coordinate 101:")
print(convergence.in_2d[1, 0])

# %%
"""
Of course, these convergences are identical to the image-plane convergences, as it`s only the lens galaxy that 
contributes to the overall mass of the ray-tracing system.
"""

# %%
image_plane_convergence = tracer.image_plane.convergence_from_grid(
    grid=image_plane_grid
)

print("Image-Plane - Convergence - `Grid` coordinate 1:")
print(image_plane_convergence.in_2d[0, 0])
print("Image-Plane - Convergence - `Grid` coordinate 2:")
print(image_plane_convergence.in_2d[0, 1])
print("Image-Plane - Convergence - `Grid` coordinate 3:")
print(image_plane_convergence.in_2d[0, 2])
print("Image-Plane - Convergence - `Grid` coordinate 101:")
print(image_plane_convergence.in_2d[1, 0])

# %%
"""
I`ve left the rest below commented to avoid too many print statements, but if you`re feeling adventurous go ahead 
and uncomment the lines below!
"""

# %%
# print(`Potential:`)
# print(tracer.potential_from_grid(grid=image_plane_grid))
# print(tracer.image_plane.potential_from_grid(grid=image_plane_grid))
# print(`Deflections:`)
# print(tracer.deflections_from_grid(grid=image_plane_grid))
# print(tracer.deflections_from_grid(grid=image_plane_grid))
# print(tracer.image_plane.deflections_from_grid(grid=image_plane_grid))
# print(tracer.image_plane.deflections_from_grid(grid=image_plane_grid))

# %%
"""
You can also plot the above attributes on individual figures, using appropriate ray-tracing `Plotter` (I`ve left most 
commented out again for convenience)
"""

# %%
aplt.Tracer.convergence(tracer=tracer, grid=image_plane_grid)

# aplt.Tracer.potential(tracer=tracer, grid=image_plane_grid)
# aplt.Tracer.deflections_y(tracer=tracer, grid=image_plane_grid)
# aplt.Tracer.deflections_x(tracer=tracer, grid=image_plane_grid)
# aplt.Tracer.image(tracer=tracer, grid=image_plane_grid)

# %%
"""
Before we finish, you might be wondering `why do both the image-plane and `Tracer` have the attributes convergence / 
potential / deflection angles, when the two are identical`. Afterall, only `MassProfile`'s contribute to these
quantities, and only the image-plane has galaxies with measureable  `MassProfile`'s! There are two reasons:

Convenience: 

 You could always write `tracer.image_plane.convergence` and `aplt.Plane.convergence(plane=tracer.image_plane). 
 However, code appears neater if you can just write `tracer.convergence` and `aplt.Tracer.convergence(tracer=tracer).

Multi-plane lensing:
 
 For now, we`re focused on the simplest lensing configuratio possible, an image-plane + source-plane configuration. 
 However, there are strong lens system where there are more than 2 planes! 

 In these instances, the  convergence, potential and deflections of each plane is different to the overall values 
 given by the `Tracer`.  This is beyond the scope of this chapter, but be reassured that what you`re learning now 
 will prepare you for the advanced chapters later on!

And with that, we`re done. You`ve performed your first ray-tracing with **PyAutoLens**! There are no exercises for this 
chapter, and we`re going to take a deeper look at ray-tracing in the next chapter.
"""
