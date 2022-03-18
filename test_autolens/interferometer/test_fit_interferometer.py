import numpy as np
import pytest

import autolens as al


def test__model_visibilities(interferometer_7):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d=np.ones(2))
    )
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.model_visibilities.slim[0] == pytest.approx(
        np.array([1.2933 + 0.2829j]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-27.06284, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(interferometer_7):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d=np.ones(2))
    )
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()

    hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-30.24288, 1.0e-4)


def test__fit_figure_of_merit(interferometer_7):
    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-21709493.32, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-21709493.32, 1.0e-4)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=0.01)

    g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-66.90612, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-66.90612, 1.0e-4)

    galaxy_light = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-1570173.14216, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-1570173.14216, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(interferometer_7):

    hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-9648681.9168, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-9648681.9168, 1.0e-4)

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        use_hyper_scaling=False,
    )

    assert fit.noise_map == pytest.approx(interferometer_7.noise_map, 1.0e-4)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=0.01)

    g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-68.63380, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-68.63380, 1.0e-4)

    galaxy_light = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-892439.04665, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-892439.04665, 1.0e-4)


def test___fit_figure_of_merit__different_settings(
    interferometer_7, interferometer_7_lop
):

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=0.01)

    g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_w_tilde=True, use_linear_operators=False
        ),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-66.90612, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-66.90612, 1.0e-4)

    fit = al.FitInterferometer(
        dataset=interferometer_7_lop,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_w_tilde=False, use_linear_operators=True
        ),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-71.5177, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-71.5177, 1.0e-4)


def test___galaxy_model_image_dict(interferometer_7, interferometer_7_grid):
    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitInterferometer(dataset=interferometer_7_grid, tracer=tracer)

    traced_grids_of_planes = tracer.traced_grid_list_from(
        grid=interferometer_7_grid.grid
    )

    g0_image = g0.image_2d_from(grid=traced_grids_of_planes[0])
    g1_image = g1.image_2d_from(grid=traced_grids_of_planes[1])

    assert fit.galaxy_model_image_dict[g0].slim == pytest.approx(g0_image, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1].slim == pytest.approx(g1_image, 1.0e-4)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(redshift=0.5)
    g1 = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7.grid, source_pixelization_grid=None
    )

    inversion = al.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_image_dict[g0].native == np.zeros((7, 7))).all()

    assert fit.galaxy_model_image_dict[g1].slim == pytest.approx(
        inversion.mapped_reconstructed_image.slim, 1.0e-4
    )
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
    g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
    g2 = al.Galaxy(redshift=0.5)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, galaxy_pix])

    fit = al.FitInterferometer(
        dataset=interferometer_7_grid,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    traced_grids = tracer.traced_grid_list_from(grid=interferometer_7_grid.grid)

    g0_visibilities = g0.visibilities_via_transformer_from(
        grid=traced_grids[0], transformer=interferometer_7_grid.transformer
    )

    g1_visibilities = g1.visibilities_via_transformer_from(
        grid=traced_grids[1], transformer=interferometer_7_grid.transformer
    )

    profile_visibilities = g0_visibilities + g1_visibilities

    profile_subtracted_visibilities = (
        interferometer_7_grid.visibilities - profile_visibilities
    )
    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7_grid.grid,
        settings=al.SettingsPixelization(use_border=False),
    )

    inversion = al.InversionInterferometer(
        visibilities=profile_subtracted_visibilities,
        noise_map=interferometer_7_grid.noise_map,
        transformer=interferometer_7_grid.transformer,
        w_tilde=interferometer_7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=al.SettingsInversion(use_w_tilde=False),
    )

    g0_image = g0.image_2d_from(grid=traced_grids[0])

    g1_image = g1.image_2d_from(grid=traced_grids[1])

    assert (fit.galaxy_model_image_dict[g2].native == np.zeros((7, 7))).all()

    assert fit.galaxy_model_image_dict[g0].slim == pytest.approx(g0_image.slim, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1].slim == pytest.approx(g1_image.slim, 1.0e-4)
    assert fit.galaxy_model_image_dict[galaxy_pix].slim == pytest.approx(
        inversion.mapped_reconstructed_image.slim, 1.0e-4
    )


def test___galaxy_model_visibilities_dict(interferometer_7, interferometer_7_grid):

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitInterferometer(dataset=interferometer_7_grid, tracer=tracer)

    traced_grids_of_planes = tracer.traced_grid_list_from(
        grid=interferometer_7_grid.grid
    )

    g0_profile_visibilities = g0.visibilities_via_transformer_from(
        grid=traced_grids_of_planes[0], transformer=interferometer_7_grid.transformer
    )

    g1_profile_visibilities = g1.visibilities_via_transformer_from(
        grid=traced_grids_of_planes[1], transformer=interferometer_7_grid.transformer
    )

    assert fit.galaxy_model_visibilities_dict[g0].slim == pytest.approx(
        g0_profile_visibilities, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].slim == pytest.approx(
        g1_profile_visibilities, 1.0e-4
    )
    assert (
        fit.galaxy_model_visibilities_dict[g2].slim == (0.0 + 0.0j) * np.zeros((7,))
    ).all()

    assert fit.model_visibilities.slim == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0].slim
        + fit.galaxy_model_visibilities_dict[g1].slim,
        1.0e-4,
    )
    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(redshift=0.5)
    g1 = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7.grid, source_pixelization_grid=None
    )

    inversion = al.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=al.SettingsInversion(use_w_tilde=False),
    )

    assert (
        fit.galaxy_model_visibilities_dict[g0] == (0.0 + 0.0j) * np.zeros((7,))
    ).all()

    assert fit.galaxy_model_visibilities_dict[g1].slim == pytest.approx(
        inversion.mapped_reconstructed_data.slim, 1.0e-4
    )

    assert fit.model_visibilities.slim == pytest.approx(
        fit.galaxy_model_visibilities_dict[g1].slim, 1.0e-4
    )
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
    g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
    g2 = al.Galaxy(redshift=0.5)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, galaxy_pix])

    fit = al.FitInterferometer(
        dataset=interferometer_7_grid,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    traced_grids = tracer.traced_grid_list_from(grid=interferometer_7_grid.grid)

    g0_visibilities = g0.visibilities_via_transformer_from(
        grid=traced_grids[0], transformer=interferometer_7_grid.transformer
    )

    g1_visibilities = g1.visibilities_via_transformer_from(
        grid=traced_grids[1], transformer=interferometer_7_grid.transformer
    )

    profile_visibilities = g0_visibilities + g1_visibilities

    profile_subtracted_visibilities = (
        interferometer_7_grid.visibilities - profile_visibilities
    )
    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7_grid.grid,
        settings=al.SettingsPixelization(use_border=False),
    )

    inversion = al.InversionInterferometer(
        visibilities=profile_subtracted_visibilities,
        noise_map=interferometer_7_grid.noise_map,
        transformer=interferometer_7_grid.transformer,
        w_tilde=interferometer_7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=al.SettingsInversion(use_w_tilde=False),
    )

    assert (
        fit.galaxy_model_visibilities_dict[g2] == (0.0 + 0.0j) * np.zeros((7,))
    ).all()

    assert fit.galaxy_model_visibilities_dict[g0].slim == pytest.approx(
        g0_visibilities.slim, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].slim == pytest.approx(
        g1_visibilities.slim, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix].slim == pytest.approx(
        inversion.mapped_reconstructed_data.slim, 1.0e-4
    )

    assert fit.model_visibilities.slim == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0].slim
        + fit.galaxy_model_visibilities_dict[g1].slim
        + inversion.mapped_reconstructed_data.slim,
        1.0e-4,
    )


def test___stochastic_mode__gives_different_log_likelihood_list(interferometer_7):

    pix = al.pix.VoronoiBrightnessImage(pixels=5)
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        pixelization=pix,
        regularization=reg,
        hyper_model_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_galaxy_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit_0 = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )
    fit_1 = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_0.log_evidence == fit_1.log_evidence

    fit_0 = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=True),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )
    fit_1 = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=True),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_0.log_evidence != fit_1.log_evidence


def test__total_mappers(interferometer_7):
    g0 = al.Galaxy(redshift=0.5)

    g1 = al.Galaxy(redshift=1.0)

    g2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.total_mappers == 0

    g2 = al.Galaxy(
        redshift=2.0,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.total_mappers == 1
