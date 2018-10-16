from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.inversion import pixelizations as pix
from autolens.plotting import mapper_plotters

# We'll start by learning about pixelization's. Typically, we'll apply these to a source-plane, to reconstruct its
# source galaxy, so we'll need a lensed source-plane grid. Lets quickly make one using the tool that should be familiar
# too you now!
image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05,
                                                                 sub_grid_size=2)

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, intensity=1.0,
                                                   effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# As we saw previously, the tracer's source-plane grid has been lensed by the lens galaxy. We'll use this source-plane
# grid to set up our pixelizations, afterall it's the source galaxy we ultimately want to reconstruct!
print(tracer.source_plane.grids)

# Now lets set up our pixelization, using a new module, 'pixelizations', which we've imported as 'pix'.
# There are multiple pixelizations available in PyAutoLens, but for now we'll keep things simple and use a uniforn
# rectangular grid. As usual, the grid's 'shape' defines its (y,x) dimensions.
rectangular = pix.Rectangular(shape=(10, 20))

# A pixelization by itself doesn't tell us much. Afterall, we've not passed it our grid of coordinates, or an image,
# or anything which tells it about the lens we're fitting. This information comes when we set up a 'mapper'. In this
# tutorial, we'll explore how the mapper uses the input source-grid to setup our rectangular pixelization (the reason
# its called a mapper will be covered in the next tutorial!)
mapper = rectangular.mapper_from_grids(grids=tracer.source_plane.grids)

# This mapper is a 'RectangularMapper' - every pixelization generates it owns mapper.
print(type(mapper))

# By plotting our mapper, we now see our pixelization. Its a fairly boring grid of rectangular pixels.
mapper_plotters.plot_rectangular_mapper(mapper=mapper)

# The mapper contains lots of information about our pixelization, for example its geometry attribute tells us where the
# pixel centers are located
print('Rectangular Grid Pixel Centre 1:')
print(mapper.geometry.pixel_centres[0])
print('Rectangular Grid Pixel Centre 2:')
print(mapper.geometry.pixel_centres[1])
print('Rectangular Grid Pixel Centre 3:')
print(mapper.geometry.pixel_centres[2])
print('etc.')

# Infact, we can plot these centre on our grid - to make it look slightly less boring!
mapper_plotters.plot_rectangular_mapper(mapper=mapper, plot_centres=True)

# The mapper also has the source-grid that we passed when we set it up. Lets check they're the same grids.
print('Source Grid Pixel 1')
print(tracer.source_plane.grids.image[0])
print(mapper.grids.image[0])
print('Source Grid Pixel 2')
print(tracer.source_plane.grids.image[1])
print(mapper.grids.image[1])
print('etc.')

# We can over-lay the grid on top. Its starting too look a bit less boring now!
mapper_plotters.plot_rectangular_mapper(mapper=mapper, plot_centres=True, plot_grid=True)

# Finally, the mapper and its geometry has lots more information about the pixelization, for example, the arc-second
# size and dimensions.
print(mapper.geometry.shape_arc_seconds)
print(mapper.geometry.arc_second_maxima)
print(mapper.geometry.arc_second_minima)

# And with that, we're done. This was a relatively gentle introduction in the world of pixelizations, but one that
# hopefully makes a lot of sense. Think about the following questions before moving on to the next tutorial:

# 1) Look at how the source-grid coordinates map to source-pixelization pixels. Is the distribution of points to pixels
#    even? Or do some pixels have a lot more grid-points inside of them? What might this means for our eventual source
#    reconstruction?

#  2) The rectangular pixelization's edges are perfectly aligned with the most exterior coordinates of the source-grid.
#     This is intentional - why do you think this is?