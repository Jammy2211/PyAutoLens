from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.inversion import pixelizations as pix
from autolens.plotting import mapper_plotters

# In the previous example, we used a mapper to make a rectangular pixelization using a lensed source-plane grid.
# However, it isn't very clear why a mapper is called a mapper - up to now it hasn't done very much mapping at all!
# That's what we'll cover in this tutorial.

# Lets begin by setting up our grids, galaxies, tracers, pixelization, mapper etc, the same as before.
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05,
                                                                 sub_grid_size=2)
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, intensity=1.0,
                                                   effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)
rectangular = pix.Rectangular(shape=(20, 20))
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

print(image_plane_grids.image.grid_to_pixel)
print(tracer.image_plane.grids.image.grid_to_pixel)
print(mapper.grids.image.grid_to_pixel[0])