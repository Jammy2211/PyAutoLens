import os
import shutil

import numpy as np
import pytest

from autolens.data.imaging import image as im
from autolens.data.array.util import array_util
from autolens.data.array import grids, mask as msk
from autolens.lensing import lensing_fitting
from autolens.model.galaxy import galaxy as g
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


def test__simulate_lensed_source_and_fit__no_psf_blurring__chi_squared_is_0__noise_term_correct():

    psf = im.PSF(array=np.array([[0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0]]), pixel_scale=1.0)

    imaging_grids = grids.DataGrids.grids_for_simulation(shape=(11, 11), pixel_scale=0.2, psf_shape=psf.shape)

    lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
                           mass=mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8))
    source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=[imaging_grids])

    image_simulated = im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.2,
                                                   exposure_time=300.0, psf=psf, background_sky_level=None,
                                                   add_noise=False)
    image_simulated.noise_map = np.ones(image_simulated.shape)

    path = "{}/datas".format(
        os.path.dirname(os.path.realpath(__file__)))  # Setup path so we can output the simulated datas.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) == False:
        os.makedirs(path)

    array_util.numpy_array_to_fits(array=image_simulated, path=path + '/data.fits')
    array_util.numpy_array_to_fits(array=image_simulated.noise_map, path=path + '/noise_map_.fits')
    array_util.numpy_array_to_fits(array=psf, path=path + '/psf.fits')

    image = im.load_imaging_from_fits(image_path=path + '/data.fits',
                                      noise_map_path=path + '/noise_map_.fits',
                                      psf_path=path + '/psf.fits', pixel_scale=0.2)

    mask = msk.Mask.circular(shape=image.shape, pixel_scale=0.2, radius_arcsec=0.8)

    lensing_image = li.LensingImage(image=image, mask=mask, sub_grid_size=1)

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=[lensing_image.grids])

    fitter = lensing_fitting.LensingProfileFit(lensing_images=[lensing_image], tracer=tracer)

    assert fitter.chi_squared_term == 0.0


def test__simulate_lensed_source_and_fit__include_psf_blurring__chi_squared_is_0__noise_term_correct():

    psf = im.PSF.simulate_as_gaussian(shape=(3, 3), pixel_scale=1.0, sigma=0.75)

    imaging_grids = grids.DataGrids.grids_for_simulation(shape=(11, 11), pixel_scale=0.2, psf_shape=psf.shape)

    lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
                           mass=mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8))
    source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=[imaging_grids])

    image_simulated = im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.2,
                                                   exposure_time=300.0, psf=psf, background_sky_level=None,
                                                   add_noise=False)
    image_simulated.noise_map = np.ones(image_simulated.shape)

    path = "{}/datas".format(
        os.path.dirname(os.path.realpath(__file__)))  # Setup path so we can output the simulated datas.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) == False:
        os.makedirs(path)

    array_util.numpy_array_to_fits(array=image_simulated, path=path + '/data.fits')
    array_util.numpy_array_to_fits(array=image_simulated.noise_map, path=path + '/noise_map_.fits')
    array_util.numpy_array_to_fits(array=psf, path=path + '/psf.fits')

    image = im.load_imaging_from_fits(image_path=path + '/data.fits',
                                      noise_map_path=path + '/noise_map_.fits',
                                      psf_path=path + '/psf.fits', pixel_scale=0.2)

    mask = msk.Mask.circular(shape=image.shape, pixel_scale=0.2, radius_arcsec=0.8)

    lensing_image = li.LensingImage(image=image, mask=mask, sub_grid_size=1)

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=[lensing_image.grids])

    fitter = lensing_fitting.LensingProfileFit(lensing_images=[lensing_image], tracer=tracer)

    assert fitter.chi_squared_term == pytest.approx(0.0, 1e-4)
