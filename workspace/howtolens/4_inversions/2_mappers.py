from autolens.imaging import image as im
from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.inversion import pixelizations as pix
from autolens.plotting import imaging_plotters
from autolens.plotting import mapper_plotters
from workspace.howtolens.simulations import inversions as simulate_image

# In the previous example, we used a mapper to make a rectangular pixelization using a lensed source-plane grid.
# However, it isn't very clear why a mapper is called a mapper - up to now it hasn't done very much mapping at all!
# That's what we'll cover in this tutorial.

# To begin, lets simulate and load an image - it'll be clear why we're doing this in a moment!
simulate_image.tutorial_2_image()
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/4_inversions'
image = im.load_imaging_from_path(image_path=path + '/data/2_mappers/image.fits',
                                  noise_map_path=path+'/data/2_mappers/noise_map.fits',
                                  psf_path=path + '/data/2_mappers/psf.fits', pixel_scale=0.05)
# imaging_plotters.plot_image_subplot(image=image)

# Lets begin by setting up our grids (using the image shape / pixel_scale).
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=image.shape, pixel_scale=image.pixel_scale,
                                                                 sub_grid_size=2)

# Our tracer will use the same lens galaxy and source galaxy as we used to simulate the image.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                    einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7, phi=135.0, intensity=0.2,
                                                   effective_radius=0.2, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# Finally, lets setup our pixelization and mapper, like we did before, using the tracer's source-plane grid.
rectangular = pix.Rectangular(shape=(10, 20))
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# mapper_plotters.plot_rectangular_mapper(mapper=mapper, plot_grid=True, source_pixels=[69, 89, 109])
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, should_plot_grid=True, source_pixels=[[69, 89], [109]])
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, should_plot_grid=True, image_pixels=[[range(0, 100)]])

# print(image_plane_grids.image.grid_to_pixel)
# print(tracer.image_plane.grids.image.grid_to_pixel)
# print(mapper.grids.image.grid_to_pixel[0])