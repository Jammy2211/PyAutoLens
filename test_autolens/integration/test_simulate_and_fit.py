import os

import numpy as np
import pytest
import shutil

import autolens as al
from autolens.simulator import simulator


def test__simulate_imaging_data_and_fit__no_psf_blurring__chi_squared_is_0__noise_normalization_correct():
    psf = al.kernel.manual_2d(
        array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=0.2,
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    imaging_simulator = simulator.ImagingSimulator(
        shape_2d=(11, 11),
        pixel_scales=0.2,
        sub_size=2,
        exposure_time=300.0,
        psf=psf,
        background_level=0.0,
        add_noise=False,
    )

    imaging_simulated = imaging_simulator.from_tracer(tracer=tracer)

    imaging_simulated.noise_map = al.array.ones(
        shape_2d=imaging_simulated.image.shape_2d
    )

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    imaging_simulated.output_to_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
    )

    imaging = al.imaging.from_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
        pixel_scales=0.2,
    )

    mask = al.mask.circular(
        shape_2d=imaging.image.shape_2d, pixel_scales=0.2, sub_size=2, radius=0.8
    )

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

    assert fit.chi_squared == 0.0

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_imaging_data_and_fit__include_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    psf = al.kernel.from_gaussian(shape_2d=(3, 3), pixel_scales=0.2, sigma=0.75)

    grid = al.grid.uniform(shape_2d=(11, 11), pixel_scales=0.2, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    imaging_simulator = simulator.ImagingSimulator(
        shape_2d=(11, 11),
        pixel_scales=0.2,
        sub_size=1,
        exposure_time=300.0,
        psf=psf,
        background_level=0.0,
        add_noise=False,
    )

    imaging_simulated = imaging_simulator.from_image(
        image=tracer.padded_profile_image_from_grid_and_psf_shape(
            grid=grid, psf_shape_2d=psf.shape_2d
        )
    )
    imaging_simulated.noise_map = al.array.ones(
        shape_2d=imaging_simulated.image.shape_2d
    )

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    imaging_simulated.output_to_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
    )

    imaging = al.imaging.from_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
        pixel_scales=0.2,
    )

    mask = al.mask.circular(
        shape_2d=imaging.image.shape_2d, pixel_scales=0.2, sub_size=1, radius=0.8
    )

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_interferometer_data_and_fit__chi_squared_is_0__noise_normalization_correct():

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    interferometer_simulator = simulator.InterferometerSimulator(
        real_space_shape_2d=(51, 51),
        real_space_pixel_scales=0.1,
        uv_wavelengths=np.ones(shape=(7, 2)),
        sub_size=2,
        exposure_time=300.0,
        background_level=0.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer_simulated = interferometer_simulator.from_tracer(tracer=tracer)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    interferometer_simulated.output_to_fits(
        visibilities_path=path + "/visibilities.fits",
        noise_map_path=path + "/noise_map.fits",
        uv_wavelengths_path=path + "/uv_wavelengths.fits",
    )

    interferometer = al.interferometer.from_fits(
        visibilities_path=path + "/visibilities.fits",
        noise_map_path=path + "/noise_map.fits",
        uv_wavelengths_path=path + "/uv_wavelengths.fits",
    )

    mask = al.mask.unmasked(shape_2d=(51, 51), pixel_scales=0.1, sub_size=2)

    masked_interferometer = al.masked.interferometer(
        interferometer=interferometer, real_space_mask=mask, inversion_uses_border=False
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.fit(masked_dataset=masked_interferometer, tracer=tracer)

    assert fit.chi_squared == 0.0

    pix = al.pix.Rectangular(shape=(7, 7))

    reg = al.reg.Constant(coefficient=0.0001)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.fit(masked_dataset=masked_interferometer, tracer=tracer)
    assert abs(fit.chi_squared) < 1.0e-4

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)
