import os
from os import path
import shutil

import autolens as al
import numpy as np
import pytest


def test__perfect_fit__chi_squared_0():
    grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

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

    dataset.output_to_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
    )

    real_space_mask = al.Mask2D.all_false(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=2
    )

    dataset = al.Interferometer.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
    )
    dataset = dataset.apply_settings(
        settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(7, 7)),
        regularization=al.reg.Constant(coefficient=0.0001),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
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

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    source_galaxy_1 = al.Galaxy(redshift=2.0, pixelization=pixelization)
    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_seed=1,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset = dataset.apply_settings(
        settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
    )

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.figure_of_merit == pytest.approx(-5.433894158056919, 1.0e-2)


def test__simulate_interferometer_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():
    grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.array(
            [
                [0.04, 200.0, 0.3, 400000.0, 60000000.0],
                [0.00003, 500.0, 600000.0, 0.1, 75555555],
            ]
        ),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset = dataset.apply_settings(
        settings=al.SettingsInterferometer(
            grid_class=al.Grid2D, transformer_class=al.TransformerDFT, sub_size=1
        )
    )

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy_linear = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
    )

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_linear]
    )

    fit_linear = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.1, 0.2]), 1.0e-4
    )
    assert fit_linear.linear_light_profile_intensity_dict[
        lens_galaxy_linear.light
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        source_galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        source_galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)
    assert fit.log_likelihood == pytest.approx(fit_linear.log_likelihood)

    lens_galaxy_image = lens_galaxy.image_2d_from(grid=dataset.grid)

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image, 1.0e-4
    )

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(grid=dataset.grid)

    source_galaxy_image = source_galaxy.image_2d_from(grid=traced_grid_2d_list[1])

    assert fit_linear.galaxy_model_image_dict[source_galaxy_linear] == pytest.approx(
        source_galaxy_image, 1.0e-4
    )

    lens_galaxy_visibilities = lens_galaxy.visibilities_from(
        grid=dataset.grid, transformer=dataset.transformer
    )

    assert fit_linear.galaxy_model_visibilities_dict[
        lens_galaxy_linear
    ] == pytest.approx(lens_galaxy_visibilities, 1.0e-4)

    source_galaxy_visibilities = source_galaxy.visibilities_from(
        grid=traced_grid_2d_list[1], transformer=dataset.transformer
    )

    assert fit_linear.galaxy_model_visibilities_dict[
        source_galaxy_linear
    ] == pytest.approx(source_galaxy_visibilities, 1.0e-4)


def test__simulate_interferometer_data_and_fit__linear_light_profiles_and_pixelization():
    grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.array(
            [
                [0.04, 200.0, 0.3, 400000.0, 60000000.0],
                [0.00003, 500.0, 600000.0, 0.1, 75555555],
            ]
        ),
        transformer_class=al.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset = dataset.apply_settings(
        settings=al.SettingsInterferometer(
            grid_class=al.Grid2D, transformer_class=al.TransformerDFT, sub_size=1
        )
    )

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                1.00338472e02,
                9.55074606e-02,
                9.24767167e-02,
                9.45392540e-02,
                1.41969109e-01,
                1.41828976e-01,
                1.41521130e-01,
                1.84257307e-01,
                1.85507562e-01,
                1.83726575e-01,
            ]
        ),
        1.0e-2,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-29.20551989, 1.0e-4)

    lens_galaxy_image = lens_galaxy.image_2d_from(grid=dataset.grid)

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(grid=dataset.grid)

    source_galaxy_image = source_galaxy.image_2d_from(grid=traced_grid_2d_list[1])

    # assert fit_linear.galaxy_model_image_dict[source_galaxy_pix] == pytest.approx(
    #     source_galaxy_image, 1.0e-1
    # )
