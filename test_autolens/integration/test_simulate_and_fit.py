import os

import numpy as np
import pytest
import shutil

import autolens as al


def test__simulate_imaging_data_and_fit__no_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    grid = al.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.2, sub_size=2)

    psf = al.Kernel.manual_2d(
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

    simulator = al.SimulatorImaging(
        exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        psf=psf,
        background_sky_map=al.Array.zeros(shape_2d=grid.shape_2d),
        add_noise=False,
    )

    imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

    imaging.noise_map = al.Array.ones(shape_2d=imaging.image.shape_2d)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    imaging.output_to_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
    )

    imaging = al.Imaging.from_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
        pixel_scales=0.2,
    )

    mask = al.Mask.circular(
        shape_2d=imaging.image.shape_2d, pixel_scales=0.2, sub_size=2, radius=0.8
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

    assert fit.chi_squared == 0.0

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_imaging_data_and_fit__include_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    grid = al.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = al.Kernel.from_gaussian(
        shape_2d=(3, 3), pixel_scales=0.2, sigma=0.75, renormalize=True
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

    simulator = al.SimulatorImaging(
        exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        psf=psf,
        background_sky_map=al.Array.zeros(shape_2d=grid.shape_2d),
        add_noise=False,
    )

    imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)
    imaging.noise_map = al.Array.ones(shape_2d=imaging.image.shape_2d)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    imaging.output_to_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
    )

    simulator = al.Imaging.from_fits(
        image_path=path + "/image.fits",
        noise_map_path=path + "/noise_map.fits",
        psf_path=path + "/psf.fits",
        pixel_scales=0.2,
    )

    mask = al.Mask.circular(
        shape_2d=simulator.image.shape_2d, pixel_scales=0.2, sub_size=1, radius=0.8
    )

    masked_imaging = al.MaskedImaging(imaging=simulator, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_interferometer_data_and_fit__chi_squared_is_0__noise_normalization_correct():

    grid = al.Grid.uniform(shape_2d=(51, 51), pixel_scales=0.1, sub_size=2)

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

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        background_sky_map=al.Array.zeros(shape_2d=grid.shape_2d),
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    interferometer.output_to_fits(
        visibilities_path=path + "/visibilities.fits",
        noise_map_path=path + "/noise_map.fits",
        uv_wavelengths_path=path + "/uv_wavelengths.fits",
    )

    interferometer = al.Interferometer.from_fits(
        visibilities_path=path + "/visibilities.fits",
        noise_map_path=path + "/noise_map.fits",
        uv_wavelengths_path=path + "/uv_wavelengths.fits",
    )

    visibilities_mask = np.full(fill_value=False, shape=(7, 2))

    real_space_mask = al.Mask.unmasked(shape_2d=(51, 51), pixel_scales=0.1, sub_size=2)

    masked_interferometer = al.MaskedInterferometer(
        interferometer=interferometer,
        visibilities_mask=visibilities_mask,
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
        inversion_uses_border=False,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer, tracer=tracer
    )

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

    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer, tracer=tracer
    )
    assert abs(fit.chi_squared) < 1.0e-4

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)
