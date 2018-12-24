import numpy as np
import time

from autolens.data.array import mask as ma
from autolens.data import ccd as im
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.data.array import grids
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.lens import lens_image as li
from autolens.lens import ray_tracing
from autolens.model.inversion.util import inversion_util
from autolens.model.inversion.util import regularization_util

image_shape = (501, 501)
pixel_scale = 0.02
psf_shape = (21, 21)

psf = im.PSF.simulate_as_gaussian(shape=psf_shape, sigma=0.05, pixel_scale=pixel_scale)

image_plane_grids = grids.GridStack.grid_stack_for_simulation(shape=image_shape, sub_grid_size=4,
                                                              pixel_scale=pixel_scale, psf_shape=psf_shape)

lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                sersic_index=2.0),
                       mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                  sersic_index=1.5))

tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=[image_plane_grids])

image = im.CCDData.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=pixel_scale,
                            exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)


mask = ma.Mask.circular(shape=image_shape, pixel_scale=pixel_scale, radius_arcsec=3.0)
lensing_image = li.LensData(ccd_data=image, mask=mask)

adaptive_pix = pix.AdaptiveMagnification(shape=(30, 30))

source_galaxy_voronoi = g.Galaxy(pixelization=adaptive_pix, regularization=reg.Constant(coefficients=(1.0,)))

# start = time.time()
image_plane_grids = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
    galaxies=[source_galaxy_voronoi], grids=lensing_image.grid_stack)
# diff = time.time() - start
# print("{}".format(diff))

tracer_fit = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy_voronoi],
                                                 image_plane_grid_stack=[image_plane_grids])

relocated_grids = lensing_image.border.relocated_grid_stack_from_grid_stack(tracer_fit.source_plane.grids[0])
voronoi = adaptive_pix.voronoi_from_pixel_centers(relocated_grids.pix)
pixel_centres = relocated_grids.pix
pixels = pixel_centres.shape[0]
pixel_neighbors, pixel_neighbors_size = adaptive_pix.neighbors_from_pixelization(pixels=pixels,
                                                                                 ridge_points=voronoi.ridge_points)
adaptive_pix.geometry_from_grid(grid=relocated_grids.sub, pixel_centres=pixel_centres,
                                pixel_neighbors=pixel_neighbors, pixel_neighbors_size=pixel_neighbors_size)
adaptive_mapper = adaptive_pix.mapper_from_grid_stack_and_border(grid_stack=tracer_fit.source_plane.grids[0],
                                                                 border=lensing_image.border)
mapping_matrix = adaptive_mapper.mapping_matrix
blurred_mapping_matrix = lensing_image.convolver_mapping_matrix.convolve_mapping_matrix(mapping_matrix=mapping_matrix)
data_vector = inversion_util.data_vector_from_blurred_mapping_matrix_and_data(
                blurred_mapping_matrix=blurred_mapping_matrix, image_1d=lensing_image, noise_map_1d=lensing_image.noise_map_1d)
curvature_matrix = inversion_util.curvature_matrix_from_blurred_mapping_matrix(
                blurred_mapping_matrix=blurred_mapping_matrix, noise_map_1d=lensing_image.noise_map_1d)
regularization_matrix = regularization_util.constant_regularization_matrix_from_pixel_neighbors(coefficients=(1.0,),
                        pixel_neighbors=adaptive_mapper.geometry.pixel_neighbors,
                        pixel_neighbors_size=pixel_neighbors_size)
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
solution_vector = np.linalg.solve(curvature_reg_matrix, data_vector)
inversion_util.reconstructed_data_vector_from_blurred_mapping_matrix_and_solution_vector(blurred_mapping_matrix,
                                                                                         solution_vector)


start_overall = time.time()

start = time.time()
relocated_grids = lensing_image.border.relocated_grid_stack_from_grid_stack(tracer_fit.source_plane.grids[0])
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


start = time.time()
adaptive_pix.geometry_from_grid(grid=relocated_grids.sub, pixel_centres=pixel_centres,
                                pixel_neighbors=pixel_neighbors, pixel_neighbors_size=pixel_neighbors_size)
diff = time.time() - start
print("Geometry Time = {}".format(diff))

start = time.time()
adaptive_mapper = adaptive_pix.mapper_from_grid_stack_and_border(grid_stack=tracer_fit.source_plane.grids[0],
                                                                 border=lensing_image.border)
diff = time.time() - start
print("Time to get mapper = {}".format(diff))

start = time.time()
mapping_matrix = adaptive_mapper.mapping_matrix
diff = time.time() - start
print("Mapping Matrix Time = {}".format(diff))

start = time.time()
blurred_mapping_matrix = lensing_image.convolver_mapping_matrix.convolve_mapping_matrix(mapping_matrix=mapping_matrix)
diff = time.time() - start
print("Blurred Mapping Matrix Time = {}".format(diff))

start = time.time()
data_vector = inversion_util.data_vector_from_blurred_mapping_matrix_and_data(
                blurred_mapping_matrix=blurred_mapping_matrix, image_1d=lensing_image, noise_map_1d=lensing_image.noise_map_1d)
diff = time.time() - start
print("Data Vector Time = {}".format(diff))

start = time.time()
curvature_matrix = inversion_util.curvature_matrix_from_blurred_mapping_matrix(
                blurred_mapping_matrix=blurred_mapping_matrix, noise_map_1d=lensing_image.noise_map_1d)
diff = time.time() - start
print("Curvature Matrix Time = {}".format(diff))

start = time.time()
regularization_matrix = regularization_util.constant_regularization_matrix_from_pixel_neighbors(coefficients=(1.0,),
                        pixel_neighbors=adaptive_mapper.geometry.pixel_neighbors,
                        pixel_neighbors_size=pixel_neighbors_size)
diff = time.time() - start
print("Reguarization Matrix Time = {}".format(diff))

start = time.time()
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
diff = time.time() - start
print("Curvature Reguarization Matrix Time = {}".format(diff))

start = time.time()
solution_vector = np.linalg.solve(curvature_reg_matrix, data_vector)
diff = time.time() - start
print("Inversion Time = {}".format(diff))

start = time.time()
inversion_util.reconstructed_data_vector_from_blurred_mapping_matrix_and_solution_vector(blurred_mapping_matrix,
                                                                                         solution_vector)
diff = time.time() - start
print("Reconstructed Vector Time = {}".format(diff))

diff_overall = time.time() - start_overall
print()
print("Overall Time  {}".format(diff_overall))

# lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=lensing_image, tracer=tracer_fit)
#
# start = time.time()
# for i in range(3):
#     lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=lensing_image, tracer=tracer_fit)
# diff = time.time() - start
# print("Time to perform fit = {}".format(diff))