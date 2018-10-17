from workspace.howtolens.simulations import inversions as simulate_image

from autolens.imaging import image as im
from autolens.imaging import mask as ma
from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.lensing import lensing_image as li
from autolens.inversion import pixelizations as pix
from autolens.plotting import imaging_plotters
from autolens.plotting import mapper_plotters

# In the previous example, we used a mapper to make a rectangular pixelization. However, wasn't clear what a mapper
# was actually mapping - it didn't do much mapping at all! That's what we'll cover in this tutorial.

# To begin, lets simulate and load an image - it'll be clear why we're doing this in a moment.
simulate_image.tutorial_2_image()
path = '/home/jammy/PyCharm/Projects/AutoLens/workspace/howtolens/4_inversions'
image = im.load_imaging_from_path(image_path=path + '/data/2_mappers/image.fits',
                                  noise_map_path=path+'/data/2_mappers/noise_map.fits',
                                  psf_path=path + '/data/2_mappers/psf.fits', pixel_scale=0.05)
imaging_plotters.plot_image_subplot(image=image)

# Lets begin by setting up our grids (using the image we loaded above).
image_plane_grids = ma.ImagingGrids.from_shape_and_pixel_scale(shape=image.shape, pixel_scale=image.pixel_scale,
                                                               sub_grid_size=2)

# Our tracer will use the same lens galaxy and source galaxy as we used to simulate the image.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                    einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7, phi=135.0, intensity=0.2,
                                                   effective_radius=0.2, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# Finally, lets setup our pixelization and mapper, like we did before, using the tracer's source-plane grid. We'll
# use a much higher resolution, 25 x 25, source-pixelization.
rectangular = pix.Rectangular(shape=(25, 25))
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# Again, we're going to plot our mapper, but we're also going to plot the image which was used to generate the grid we
# mapped to the source-plane.
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper)

# The pixels in the image map to the points in the source-plane. However, its not clear how - so lets highlight a set
# of image-pixels in both the image and source-plane
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, image_pixels=[[range(500, 600)], [range(900, 1000)]])

# That's nice, and we can see the mappings, but it isn't really what we want to know is it? We really want to go the
# other way, and see how our source-pixels map to the image. This is where mappers come into their own, as they let us
# map all the image-coordinates in a given source-pixel back t the image. Lets map source pixel 313, the central
# source-pixel,to the image.
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[312]])

# And there we see it - multiple imaging in all its glory. Try changing the source-pixel indexes of the line below.
# This will give you a feel for how different regions of the source-plane map to the image.
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[312, 318], [412]])

# Okay, so I think we can agree, mappers map things! More specifically, they map our source-plane pixelization to an
# observed image of a strong lens.
#
# Finally, lets do the same as above, but using a masked image. By applying a mask, the mapper will only map
# image-pixels inside the mask. This removes the (many) image pixels at the edge of the image, where the source clearly
# isn't present and which pad-out the size of the source-plane. Lets just have a quick look at these edges pixels:
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[0, 1, 2, 3, 4, 5, 6, 7],
                                                                                 [620, 621, 622, 623]])

# Lets use an annular mask.
mask = ma.Mask.annular(shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=1.0,
                       outer_radius_arcsec=2.2)

#I've checked that the annuli radii capture the source's light
imaging_plotters.plot_image(image=image, mask=mask)

# We setup our image and mask up as a lensing image and create a new tracer using the lensed image's (now masked) grids.
lensing_image = li.LensingImage(image=image, mask=mask)
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=lensing_image.grids)

# Finally, we use the tracer's (now masked) source-plane grid to setup a new mapper (using the same rectangular 25 x 25
# pixelization as before).
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# Lets have another look
mapper_plotters.plot_image_and_mapper(image=image, mask=mask, mapper=mapper)

# Woah! Look how much closer we are to the diamond of points in the centre of the source-plane (for those who have been
# reading up, this diamond is called the 'caustic'). This diamond defines when lensing moves from the quadruply imaged
# regime to doubly-imaged regime, and we can actually show this now using our mapper's source-pixels.
mapper_plotters.plot_image_and_mapper(image=image, mask=mask, mapper=mapper, source_pixels=[[312], [314], [316], [318]])

# Great - tutorial 2 down! We've learnt about mappers, and used them to understand how the image and source
# plane map to one another. Your exercises are:

# 1) Change the einstein radius of the lens galaxy in small increments (e.g. einstein radius 1.6" -> 1.55").
#    As the radius deviates from 1.6" (the input value of the simulated lens), what do you notice about where the
#    points maps in the centre of the source-plane (where the source-galaxy is simulated, e.g. (0.0", 0.0"))?

# 2) Now make the axis ratio of the lens's mass equal to 1.0. What happens to quadruple imaging?

# 3) Now, finally, think - how is all of this going to help us actually model lenses? We've said we're going to
#    reconstruct our source galaxy's on the pixel-grid. So, how does knowing how each pixel maps to the image
#    actually help us? If you've not got any bright ideas, then worry not - that exactly what we're going to cover
#    in the next tutorial.