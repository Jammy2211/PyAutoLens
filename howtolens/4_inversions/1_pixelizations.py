from autolens.profiles import mass_profiles as mp
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import mask
from autolens.inversion import pixelizations as pix
from autolens.plotting import mapper_plotters

from astropy import cosmology

# To begin, we'll learn about how we pixelize our source-plane. To do this, we'll need a lensed source-plane, so lets
# quickly make one using all the tools we've learnt about up to now.

image_plane_grids = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05,
                                                                 sub_grid_size=2)

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, intensity=1.0,
                                                   effective_radius=1.0, sersic_index=2.5))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

# We'll use the source-plane grid with our pixelizations, afterall it's the source galaxy we want to reconstruct!
source_plane_grids = tracer.source_plane.grids

# Now lets set up our pixelization, using a new module, 'pixelizations', which we've imported as 'pix'. As the name
# suggests, there are multiple pixelizations available in PyAutoLens, but lets stick to a simple rectangular grid for
# now. The 'shape' defines its (y,x) dimensions, as we're used to with other grids in PyAutoLens.
rectangular = pix.Rectangular(shape=(20, 20))

# The rectangular pixelizatiion is just a 20 x 20 grid of pixels. It knows nothing of our observed image, the
# source-plane we intend to reconstruct the galaxy of or the arc-second coordinates of this plane. This information
# comes when we turn our pixelization into a 'mapper', using the grid of pixels in the plane we want to reconstruct
# the source. So lets make a 'mapper', by passing the pixelization our grid.
mapper = rectangular.mapper_from_grids(grids=source_plane_grids)

# By plotting our mapper, we can now see our pixelization, albeit its a fairly boring grid of rectangular pixels.
# mapper_plotters.plot_rectangular_mapper(mapper=mapper)

mapper_plotters.plot_rectangular_mapper(mapper=mapper, plot_grid=True)

# This mapper is a 'RectangularMapper' - every pixelization generates it owns mapper.
print(type(mapper))

# The mapper contains lots of information about our pixelization, for example its geometry tells us its pixel centers
# and arc-second shape
print(mapper.geometry.pixel_centres)
print()
print(mapper.geometry.shape_arc_seconds)
