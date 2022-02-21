import os
from os import path
import shutil

import autolens as al
import numpy as np
import pytest


def test__simulate_imaging_data_and_fit__no_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    grid = al.Grid2DIterate.uniform(shape_native=(11, 11), pixel_scales=0.2)

    psf = al.Kernel2D.manual_native(
        array=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        pixel_scales=grid.pixel_scales,
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.EllExponential(centre=(0.1, 0.1), intensity=0.5)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise=False
    )

    imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

    imaging.noise_map = al.Array2D.ones(
        shape_native=imaging.image.shape_native, pixel_scales=grid.pixel_scales
    )

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    imaging.output_to_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    imaging = al.Imaging.from_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = al.Mask2D.circular(
        shape_native=imaging.image.shape_native,
        pixel_scales=0.2,
        sub_size=2,
        radius=0.8,
    )

    masked_imaging = imaging.apply_mask(mask=mask)
    masked_imaging = masked_imaging.apply_settings(
        settings=al.SettingsImaging(grid_class=al.Grid2DIterate)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_imaging, tracer=tracer)

    assert fit.chi_squared >= 0.0
    assert fit.chi_squared < 1.0e-8

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )
    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__simulate_imaging_data_and_fit__include_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.EllExponential(centre=(0.1, 0.1), intensity=0.5)
    )
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    imaging = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise=False)

    imaging = imaging.via_tracer_from(tracer=tracer, grid=grid)
    imaging.noise_map = al.Array2D.ones(
        shape_native=imaging.image.shape_native, pixel_scales=0.2
    )

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    imaging.output_to_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    imaging = al.Imaging.from_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = al.Mask2D.circular(
        shape_native=imaging.image.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_imaging = imaging.apply_mask(mask=mask)
    masked_imaging = masked_imaging.apply_settings(
        settings=al.SettingsImaging(sub_size=1)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_imaging, tracer=tracer)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__simulate_imaging_data_and_fit__known_likelihood():

    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2, sub_size=1)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_0 = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=(1.0)),
    )
    source_galaxy_1 = al.Galaxy(
        redshift=2.0,
        pixelization=al.pix.Rectangular(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=(1.0)),
    )
    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    imaging = al.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    imaging = imaging.via_tracer_from(tracer=tracer, grid=grid)

    mask = al.Mask2D.circular(
        shape_native=imaging.image.shape_native, pixel_scales=0.2, radius=2.0
    )

    masked_imaging = imaging.apply_mask(mask=mask)

    fit = al.FitImaging(dataset=masked_imaging, tracer=tracer)

    assert fit.figure_of_merit == pytest.approx(609.0653285500165, 1.0e-2)


def test__simulate_interferometer_data_and_fit__chi_squared_is_0__noise_normalization_correct():

    grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.EllExponential(centre=(0.1, 0.1), intensity=0.5)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer = simulator.via_tracer_from(tracer=tracer, grid=grid)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    interferometer.output_to_fits(
        visibilities_path=path.join(file_path, "visibilities.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
    )

    real_space_mask = al.Mask2D.unmasked(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=2
    )

    interferometer = al.Interferometer.from_fits(
        visibilities_path=path.join(file_path, "visibilities.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
    )
    interferometer = interferometer.apply_settings(
        settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=interferometer,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(use_border=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pix = al.pix.Rectangular(shape=(7, 7))

    reg = al.reg.Constant(coefficient=0.0001)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=interferometer,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(use_border=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )
    assert abs(fit.chi_squared) < 1.0e-4

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__simulate_interferometer_data_and_fit__known_likelihood():

    mask = al.Mask2D.circular(
        radius=3.0, shape_native=(31, 31), pixel_scales=0.2, sub_size=1
    )

    grid = al.Grid2D.from_mask(mask=mask)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_0 = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=(1.0)),
    )
    source_galaxy_1 = al.Galaxy(
        redshift=2.0,
        pixelization=al.pix.Rectangular(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=(1.0)),
    )
    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_seed=1,
    )

    interferometer = simulator.via_tracer_from(tracer=tracer, grid=grid)

    interferometer = interferometer.apply_settings(
        settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
    )

    fit = al.FitInterferometer(
        dataset=interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.figure_of_merit == pytest.approx(-5.433894158056919, 1.0e-2)
