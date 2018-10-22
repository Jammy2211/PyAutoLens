import os
import shutil

import numpy as np
import pytest

from autolens.imaging import image as im
from autolens.imaging import imaging_util
from autolens.imaging import mask as msk
from autolens.lensing import lensing_fitting
from autolens.galaxy import galaxy as g
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp


def test__simulate_lensed_source_and_fit__no_psf_blurring__chi_squared_is_0__noise_term_correct():
    psf = im.PSF(array=np.array([[0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0]]), pixel_scale=1.0)

    imaging_grids = msk.ImagingGrids.grids_for_simulation(shape=(11, 11), pixel_scale=0.2, psf_shape=psf.shape)

    lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
                           mass=mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8))
    source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=imaging_grids)

    image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.2,
                                                   exposure_time=300.0, psf=psf, background_sky_level=None,
                                                   add_noise=False)
    image_simulated.noise_map = np.ones(image_simulated.shape)

    path = "{}/data".format(
        os.path.dirname(os.path.realpath(__file__)))  # Setup path so we can output the simulated data.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) == False:
        os.makedirs(path)

    imaging_util.numpy_array_to_fits(array=image_simulated, path=path + '/_data.fits')
    imaging_util.numpy_array_to_fits(array=image_simulated.noise_map, path=path + '/noise_map.fits')
    imaging_util.numpy_array_to_fits(array=psf, path=path + '/psf.fits')

    image = im.load_imaging_from_path(image_path=path + '/_data.fits',
                                      noise_map_path=path + '/noise_map.fits',
                                      psf_path=path + '/psf.fits', pixel_scale=0.2)

    mask = msk.Mask.circular(shape=image.shape, pixel_scale=0.2, radius_mask_arcsec=0.8)

    lensing_image = li.LensingImage(image=image, mask=mask, sub_grid_size=1)

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=lensing_image.grids)

    fitter = lensing_fitting.LensingProfileFit(lensing_image=lensing_image, tracer=tracer)

    assert fitter.chi_squared_term == 0.0


def test__simulate_lensed_source_and_fit__include_psf_blurring__chi_squared_is_0__noise_term_correct():
    psf = im.PSF.simulate_as_gaussian(shape=(3, 3), pixel_scale=1.0, sigma=0.75)

    imaging_grids = msk.ImagingGrids.grids_for_simulation(shape=(11, 11), pixel_scale=0.2, psf_shape=psf.shape)

    lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
                           mass=mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8))
    source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=imaging_grids)

    image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.2,
                                                   exposure_time=300.0, psf=psf, background_sky_level=None,
                                                   add_noise=False)
    image_simulated.noise_map = np.ones(image_simulated.shape)

    path = "{}/data".format(
        os.path.dirname(os.path.realpath(__file__)))  # Setup path so we can output the simulated data.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) == False:
        os.makedirs(path)

    imaging_util.numpy_array_to_fits(array=image_simulated, path=path + '/_data.fits')
    imaging_util.numpy_array_to_fits(array=image_simulated.noise_map, path=path + '/noise_map.fits')
    imaging_util.numpy_array_to_fits(array=psf, path=path + '/psf.fits')

    image = im.load_imaging_from_path(image_path=path + '/_data.fits',
                                      noise_map_path=path + '/noise_map.fits',
                                      psf_path=path + '/psf.fits', pixel_scale=0.2)

    mask = msk.Mask.circular(shape=image.shape, pixel_scale=0.2, radius_mask_arcsec=0.8)

    lensing_image = li.LensingImage(image=image, mask=mask, sub_grid_size=1)

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=lensing_image.grids)

    fitter = lensing_fitting.LensingProfileFit(lensing_image=lensing_image, tracer=tracer)

    assert fitter.chi_squared_term == pytest.approx(0.0, 1e-4)
