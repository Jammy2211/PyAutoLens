import sys
sys.path.append("../")

from auto_lens.imaging import image
from auto_lens.imaging import mask
from auto_lens.imaging import grids
from auto_lens.profiles import light_profiles as lp
from auto_lens.analysis import ray_tracing
from auto_lens.analysis import galaxy

### Setup mask + grid of this image ###

ma = mask.Mask.unmasked(shape_arc_seconds=(5.0, 5.0), pixel_scale=0.1)
ma = mask.Mask(array=ma, pixel_scale=0.1)

image_grids = grids.GridCoordsCollection.from_mask(mask=ma, grid_size_sub=1, blurring_size=(3, 3))
mappers = grids.GridMapperCollection.from_mask(mask=ma)

### Setup the ray tracing model, and use to generate the 2D galaxy image ###

gal = galaxy.Galaxy(light_profiles=[lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
                                                        phi=90.0, intensity=0.5, effective_radius=1.0,
                                                        sersic_index=4.0)])

ray_trace = ray_tracing.Tracer(lens_galaxies=[gal], source_galaxies=[], image_plane_grids=image_grids)

grid_galaxy_image = ray_trace.generate_image_of_galaxy_light_profiles()
galaxy_image = mappers.data_to_pixel.map_to_2d(grid_galaxy_image)

# plt.imshow(galaxy_image)
# plt.show()

### Setup the image as an image.

sim_image = image.Image.simulate(array=galaxy_image)
