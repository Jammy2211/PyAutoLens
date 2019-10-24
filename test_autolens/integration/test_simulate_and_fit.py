import os

import numpy as np
import pytest
import shutil

import autoarray as aa
import autolens as al

def test__simulate_lensed_source_and_fit__no_psf_blurring__chi_squared_is_0__noise_normalization_correct():
    psf = aa.kernel(
        array_1d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=0.2,
    )

    grid = aa.grid.uniform(
        shape_2d=(11, 11), pixel_scales=0.2, sub_size=2
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.1, 0.1), einstein_radius=1.8
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    imaging_simulated = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scales=0.2,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.0,
        add_noise=False,
    )

    imaging_simulated.noise_map = np.ones(imaging_simulated.image.shape)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    aa.util.array.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.image, file_path=path + "/image.fits"
    )
    aa.util.array.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.noise_map, file_path=path + "/noise_map.fits"
    )
    aa.util.array.numpy_array_2d_to_fits(array_2d=psf, file_path=path + "/psf.fits")

    imaging = aa.imaging.from_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
        pixel_scales=0.2,
    )

    mask = aa.mask.circular(
        shape=imaging.image.shape, pixel_scales=0.2, sub_size=2, radius_arcsec=0.8
    )

    lens_data = al.MaskedImaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.ImagingFit.from_masked_data_and_tracer(
        lens_data=lens_data, tracer=tracer
    )

    assert fit.chi_squared == 0.0

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_lensed_source_and_fit__include_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    psf = aa.kernel.from_gaussian(shape_2d=(3, 3), pixel_scales=0.2, sigma=0.75)

    grid = aa.grid.uniform(
        shape_2d=(11, 11), pixel_scales=0.2, sub_size=1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.1, 0.1), einstein_radius=1.8
        ),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    imaging_simulated = al.SimulatedImagingData.simulate(
        real_space_image=tracer.padded_profile_image_2d_from_grid_and_psf_shape(
            grid=grid, psf_shape=psf.shape_2d
        ),
        real_space_pixel_scales=0.2,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.0,
        add_noise=False,
    )
    imaging_simulated.noise_map = np.ones(imaging_simulated.image.shape)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    aa.util.array.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.image, file_path=path + "/image.fits"
    )
    aa.util.array.numpy_array_2d_to_fits(
        array_2d=imaging_simulated.noise_map, file_path=path + "/noise_map.fits"
    )
    aa.util.array.numpy_array_2d_to_fits(array_2d=psf, file_path=path + "/psf.fits")

    imaging = aa.imaging.from_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
        pixel_scales=0.2,
    )

    mask = aa.mask.circular(
        shape=imaging.image.shape, pixel_scales=0.2, sub_size=1, radius_arcsec=0.8
    )

    lens_data = al.MaskedImaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fitter = al.ImagingFit.from_masked_data_and_tracer(
        lens_data=lens_data, tracer=tracer
    )

    assert fitter.chi_squared == pytest.approx(0.0, 1e-4)

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)
