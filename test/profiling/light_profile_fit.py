import time

from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens import lens_data as ld
from autolens.lens import ray_tracing
from autolens.lens import lens_fit
from autolens.lens.util import lens_fit_util

from test.simulation import simulation_util

repeats = 10

print('Number of repeats = ' + str(repeats))
print()

sub_grid_size = 4
radius_arcsec = 3.0
psf_shape = (21,21)

print('sub grid size = ' + str(sub_grid_size))
print('circular mask radius = ' + str(radius_arcsec) + '\n')
print('psf shape = ' + str(psf_shape) + '\n')

lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0,
                                                intensity=0.5, effective_radius=0.8, sersic_index=4.0),
                       mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6,
                                                    axis_ratio=0.7, phi=45.0))

source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=60.0,
                                                   intensity=0.4, effective_radius=0.5, sersic_index=1.0))

for data_resolution in ['LSST', 'Euclid', 'HST', 'HST_Up', 'AO']:

    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_and_source_smooth', data_resolution=data_resolution,
                                                  psf_shape=psf_shape)
    mask = msk.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=radius_arcsec)
    lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=sub_grid_size)

    print('Light profile fit run times for image type ' + data_resolution + '\n')
    print('Number of points = ' + str(lens_data.grid_stack.sub.shape[0]) + '\n')

    start_overall = time.time()

    start = time.time()
    for i in range(repeats):
        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=lens_data.grid_stack)
    diff = time.time() - start
    print("Time to Setup Tracer = {}".format(diff/repeats))

    start = time.time()
    for i in range(repeats):
        blurred_profile_image_1d = lens_fit_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.profile_image_plane_image_1d, blurring_image_1d=tracer.profile_image_plane_blurring_image_1d,
            convolver=lens_data.convolver_image)
        blurred_profile_image = lens_data.scaled_array_2d_from_array_1d(array_1d=blurred_profile_image_1d)
    diff = time.time() - start
    print("Time to perform PSF convolution = {}".format(diff/repeats))

    start = time.time()
    for i in range(repeats):
        lens_fit.LensDataFit(image_1d=lens_data.image_1d, noise_map_1d=lens_data.noise_map_1d, mask_1d=lens_data.mask_1d,
                             model_image_1d=blurred_profile_image_1d)
    diff = time.time() - start
    print("Time to perform fit (1D) = {}".format(diff/repeats))

    start = time.time()
    for i in range(repeats):
        lens_fit.LensDataFit(image_1d=lens_data.unmasked_image, noise_map_1d=lens_data.unmasked_noise_map, mask_1d=lens_data.mask_2d,
                             model_image_1d=blurred_profile_image)
    diff = time.time() - start
    print("Time to perform fit (2D) = {}".format(diff/repeats))

    start = time.time()
    for i in range(repeats):
        lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer)
    diff = time.time() - start
    print("Time to perform complete fit = {}".format(diff/repeats))

    print()