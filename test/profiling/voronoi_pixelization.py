from astropy.io import fits
import os
import time

from autolens.data.array import mask as ma
from autolens.data.imaging import image as im
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.data.array import grids
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing
from autolens.lensing import lensing_fitting

image_shape = (501, 501)
pixel_scale = 0.02
psf_shape = (21, 21)

psf = im.PSF.simulate_as_gaussian(shape=psf_shape, sigma=0.05, pixel_scale=pixel_scale)

image_plane_grids = grids.DataGrids.grids_for_simulation(shape=image_shape, sub_grid_size=4,
                                                         pixel_scale=pixel_scale, psf_shape=psf_shape)

lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                sersic_index=2.0),
                       mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                  sersic_index=1.5))

tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=[image_plane_grids])

image = im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=pixel_scale,
                          exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)


mask = ma.Mask.circular(shape=image_shape, pixel_scale=pixel_scale, radius_mask_arcsec=3.0)
lensing_image = li.LensingImage(image=image, mask=mask)

adaptive_pix = pix.AdaptiveMagnification(shape=(80, 80))

source_galaxy_voronoi = g.Galaxy(pixelization=adaptive_pix, regularization=reg.Constant(coefficients=(1.0,)))

# start = time.time()
image_plane_grids = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grids(
    galaxies=[source_galaxy_voronoi], grids=lensing_image.grids)
# diff = time.time() - start
# print("{}".format(diff))

tracer_fit = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy_voronoi],
                                             image_plane_grids=[image_plane_grids])

start = time.time()
relocated_grids = lensing_image.border.relocated_grids_from_grids(tracer_fit.source_plane.grids[0])
diff = time.time() - start
print("Border time = {}".format(diff))

start = time.time()
voronoi = adaptive_pix.voronoi_from_pixel_centers(relocated_grids.pix)
diff = time.time() - start
print("Voronoi time = {}".format(diff))

pixel_centres = relocated_grids.pix
pixels = pixel_centres.shape[0]

start = time.time()
pixel_neighbors, pixel_neighbors_size = adaptive_pix.neighbors_from_pixelization(pixels=pixels,
                                                                                 ridge_points=voronoi.ridge_points)
diff = time.time() - start
print("Neighbors time = {}".format(diff))

#
# start = time.time()
# adaptive_pix.geometry_from_grid(grid=relocated_grids.sub, pixel_centres=pixel_centres,
#                                            pixel_neighbors=pixel_neighbors)
# diff = time.time() - start
# print("{}".format(diff))

# start = time.time()
adaptive_mapper = adaptive_pix.mapper_from_grids_and_border(grids=tracer_fit.source_plane.grids[0],
                                                            border=lensing_image.border)
# diff = time.time() - start
# print("{}".format(diff))

start = time.time()
adaptive_mapper.sub_to_pix
diff = time.time() - start
print("Sub to Pix time = {}".format(diff))