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
imaging_plotters.plot_image_subplot(image=image)

# Lets begin by setting up our grids (using the image shape / pixel_scale).
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
# use a much higher resolution, 30 x 30, source-pixelization.
rectangular = pix.Rectangular(shape=(25, 25))
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# Now, we're going to plot our mapper, like we did before, but we're also going to plot the image which was used to
# generate the grid we mapped to the source-plane.
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper)

# However, you might not be convinced that the image-grid and source-grid map to one another, right? In chapter 1,
# we highlighted specific image pixels to do this, and we can do it again.
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, image_pixels=[[range(500, 600)], [range(900, 1000)]])

# This isn't really what we want to know right? We're not using the mapper to map these image-pixels, we're just
# shooting a random bunch of image pixels to the source-plane and seeing where they land. With mappers, where it gets
# really interesting, is that they allow us to map the other way. Lets map source pixel 313, the central source-pixel,
# to the image-plane (remembering we go from the top-left).
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[312]])

# Pretty great right? If you weren't convinced about multiple imaging before you must be now! Before moving on, try
# changing the source-pixel index. This will give you a feel for how different regions of the image and source plane
# map between one another. Bare in mind, you can set up multiple mappings of pixels.
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[312, 318], [412]])

# Okay, so hopefully you can see now that mappers map things! The final thing I want us to cover is that, if we
# mask our image (like you should be used to by now), the mapper only maps image-pixels inside the ma. This is nice,
# as it removes the large number of mapped image pixels in the exterior of the source-plane, which clearly don't
# map to the actual lensed source galaxy. Lets just have a quick look:
mapper_plotters.plot_image_and_mapper(image=image, mapper=mapper, source_pixels=[[0, 1, 2, 3, 4, 5, 6, 7],
                                                                                 [620, 621, 622, 623]])

# Lets generated our masked image, using an annular mask where I've made sure the radii trace the source's light.
mask = ma.Mask.annular(shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=1.0,
                       outer_radius_arcsec=2.2)

imaging_plotters.plot_image(image=image, mask=mask)

# We setup our image and mask up as a lensing image and use this to create a new tracer, using the lensed image's
# (now masked) grids.
lensing_image = li.LensingImage(image=image, mask=mask)
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=lensing_image.grids)

# Finally, we use these lensed and masked grids to setup a new mapper (using the same 25 x 25 pixelization as before).
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# Lets have another look
mapper_plotters.plot_image_and_mapper(image=image, mask=mask, mapper=mapper)

# Woah! Look how much closer we are to the diamond of points in the centre of the source-plane (for those who have been
# reading up, this diamond is called the 'caustic'). This diamond defines when lensing moves from the quadruply imaged
# regime to doubly-imaged regime, and we can actualyl show this now using our mapper's source-pixels.
mapper_plotters.plot_image_and_mapper(image=image, mask=mask, mapper=mapper, source_pixels=[[312], [314], [316], [318]])

# Great - tutorial 2 down! We've learnt about mappers, and used them to understand how the image and source
# plane map to one another. Your exercises are:

# 1) Change the einstein radius of the lens galaxy in small increments (e.g. einstein radius 1.6" -> 1.55"). As you
#    As the radius deviates from 1.6" (the input value of the simulated lens), what do you notice about where the
#    points maps in the centre of the source-plane (where the source-galaxy is simulated, e.g. (0.0", 0.0"))?

# 2) Now make the axis ratio of the lens's mass equal to 1.0. What happens to quadruple imaging?

# 3) Now, finally, think - how is all of this going to help us actually model lenses? We've said we're going to
#    reconstruct our source galaxy's on the pixel-grid. So, how does knowing how each pixel maps to the image
#    actually help us? If you've not got any bright ideas, then worry not - thats exactly what we're going to cover
#    next.