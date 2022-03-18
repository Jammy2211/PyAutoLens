import numpy as np
import pytest

import autolens as al


def test__model_image__with_and_without_psf_blurring(
    masked_imaging_7x7_no_blur, masked_imaging_7x7
):

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.m.MockLightProfile(
            image_2d_value=1.0, image_2d_first_value=2.0
        ),
    )
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.model_image.slim == pytest.approx(
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-14.6337, 1.0e-4)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.model_image.slim == pytest.approx(
        np.array([1.33, 1.16, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-14.52960, 1.0e-4)


def test__noise_map__with_and_without_hyper_galaxy(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d_value=1.0)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=al.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        ),
        hyper_model_image=hyper_image,
        hyper_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=4.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-20.7470, 1.0e-4)


def test__noise_map__with_hyper_galaxy_reaches_upper_limit(masked_imaging_7x7_no_blur):

    hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=al.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0e9, noise_power=1.0
        ),
        hyper_model_image=hyper_image,
        hyper_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=1.0e8, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-174.0565, 1.0e-4)


def test__image__with_and_without_hyper_background_sky(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d_value=1.0)
    )
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.image.slim == pytest.approx(np.full(fill_value=1.0, shape=(9,)), 1.0e-1)

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d_value=1.0)
    )
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)

    fit = al.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
    )

    assert fit.image.slim == pytest.approx(np.full(fill_value=2.0, shape=(9,)), 1.0e-1)
    assert fit.log_likelihood == pytest.approx(-15.6337, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d_value=1.0)
    )
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = al.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=3.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-18.1579, 1.0e-4)


def test__fit_figure_of_merit(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.image == np.full(fill_value=1.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=2.0, shape=(9,))).all()
    assert fit.log_likelihood == pytest.approx(-1168351.9731, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-1168351.9731, 1.0e-4)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.image == np.full(fill_value=1.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=2.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-22.90055, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-22.90055, 1.0e-4)

    galaxy_light = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.image == np.full(fill_value=1.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=2.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-37667.0303, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-37667.0303, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(masked_imaging_7x7):

    hyper_galaxy = al.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        use_hyper_scaling=True,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.log_likelihood == pytest.approx(-186617.89365, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-186617.89365, 1.0e-4)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        pixelization=pix,
        regularization=reg,
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-30.14482, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-30.14482, 1.0e-4)

    galaxy_light = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_galaxy_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_minimum_value=0.0,
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-6106.6402, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-6106.6402, 1.0e-4)


def test__galaxy_model_image_dict(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    traced_grids_of_planes = tracer.traced_grid_list_from(grid=masked_imaging_7x7.grid)
    traced_blurring_grids_of_planes = tracer.traced_grid_list_from(
        grid=masked_imaging_7x7.blurring_grid
    )

    g0_image = g0.image_2d_from(grid=traced_grids_of_planes[0])
    g0_blurring_image = g0.image_2d_from(grid=traced_blurring_grids_of_planes[0])

    g0_blurred_image = masked_imaging_7x7.convolver.convolve_image(
        image=g0_image, blurring_image=g0_blurring_image
    )

    g1_image = g1.image_2d_from(grid=traced_grids_of_planes[1])
    g1_blurring_image = g1.image_2d_from(grid=traced_blurring_grids_of_planes[1])

    g1_blurred_image = masked_imaging_7x7.convolver.convolve_image(
        image=g1_image, blurring_image=g1_blurring_image
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(g0_blurred_image, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_blurred_image, 1.0e-4)
    assert (fit.galaxy_model_image_dict[g2].slim == np.zeros(9)).all()

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native + fit.galaxy_model_image_dict[g1].native,
        1.0e-4,
    )

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(redshift=0.5)
    g1 = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid, source_pixelization_grid=None
    )

    inversion = al.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_image_dict[g0] == np.zeros(9)).all()

    assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
        inversion.mapped_reconstructed_image.native, 1.0e-4
    )

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g1].native, 1.0e-4
    )

    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
    g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=2.0))
    g2 = al.Galaxy(redshift=0.5)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)
    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2, galaxy_pix])

    masked_imaging_7x7.image[0] = 3.0

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    traced_grids = tracer.traced_grid_list_from(grid=masked_imaging_7x7.grid)
    traced_blurring_grids = tracer.traced_grid_list_from(
        grid=masked_imaging_7x7.blurring_grid
    )

    g0_image = g0.image_2d_from(grid=traced_grids[0])
    g0_blurring_image = g0.image_2d_from(grid=traced_blurring_grids[0])

    g0_blurred_image = masked_imaging_7x7.convolver.convolve_image(
        image=g0_image, blurring_image=g0_blurring_image
    )

    g1_image = g1.image_2d_from(grid=traced_grids[1])
    g1_blurring_image = g1.image_2d_from(grid=traced_blurring_grids[1])

    g1_blurred_image = masked_imaging_7x7.convolver.convolve_image(
        image=g1_image, blurring_image=g1_blurring_image
    )

    blurred_image = g0_blurred_image + g1_blurred_image

    profile_subtracted_image = masked_imaging_7x7.image - blurred_image

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid,
        settings=al.SettingsPixelization(use_border=False),
    )

    inversion = al.InversionImaging(
        image=profile_subtracted_image,
        noise_map=masked_imaging_7x7.noise_map,
        convolver=masked_imaging_7x7.convolver,
        w_tilde=masked_imaging_7x7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
    )

    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

    assert fit.galaxy_model_image_dict[g0].native == pytest.approx(
        g0_blurred_image.native, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
        g1_blurred_image.native, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[galaxy_pix].native == pytest.approx(
        inversion.mapped_reconstructed_image.native, 1.0e-4
    )

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native
        + fit.galaxy_model_image_dict[g1].native
        + inversion.mapped_reconstructed_image.native,
        1.0e-4,
    )


def test___blurred_and_model_image_properties(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_images_of_planes = tracer.blurred_image_2d_list_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    assert blurred_images_of_planes[0].native == pytest.approx(
        fit.model_images_of_planes_list[0].native, 1.0e-4
    )

    assert blurred_images_of_planes[1].native == pytest.approx(
        fit.model_images_of_planes_list[1].native, 1.0e-4
    )

    unmasked_blurred_image = tracer.unmasked_blurred_image_2d_via_psf_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (unmasked_blurred_image == fit.unmasked_blurred_image).all()

    unmasked_blurred_image_of_planes_list = tracer.unmasked_blurred_image_2d_list_via_psf_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (
        unmasked_blurred_image_of_planes_list[0]
        == fit.unmasked_blurred_image_of_planes_list[0]
    ).all()
    assert (
        unmasked_blurred_image_of_planes_list[1]
        == fit.unmasked_blurred_image_of_planes_list[1]
    ).all()

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=1.0)

    g0 = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid,
        settings=al.SettingsPixelization(use_border=False),
    )

    inversion = al.Inversion(
        dataset=masked_imaging_7x7, linear_obj_list=[mapper], regularization_list=[reg]
    )

    assert (fit.model_images_of_planes_list[0].native == np.zeros((7, 7))).all()
    assert inversion.mapped_reconstructed_image.native == pytest.approx(
        fit.model_images_of_planes_list[1].native, 1.0e-4
    )

    galaxy_light = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image = tracer.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    profile_subtracted_image = masked_imaging_7x7.image - blurred_image

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid,
        settings=al.SettingsPixelization(use_border=False),
    )

    inversion = al.InversionImaging(
        image=profile_subtracted_image,
        noise_map=masked_imaging_7x7.noise_map,
        convolver=masked_imaging_7x7.convolver,
        w_tilde=masked_imaging_7x7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
    )

    assert blurred_image.native == pytest.approx(
        fit.model_images_of_planes_list[0].native, 1.0e-4
    )
    assert inversion.mapped_reconstructed_image.native == pytest.approx(
        fit.model_images_of_planes_list[1].native, 1.0e-4
    )


def test__subtracted_images_of_planes_list(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d=np.ones(1))
    )

    g1 = al.Galaxy(
        redshift=0.75, light_profile=al.m.MockLightProfile(image_2d=2.0 * np.ones(1))
    )

    g2 = al.Galaxy(
        redshift=1.0, light_profile=al.m.MockLightProfile(image_2d=3.0 * np.ones(1))
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == -4.0
    assert fit.subtracted_images_of_planes_list[1].slim[0] == -3.0
    assert fit.subtracted_images_of_planes_list[2].slim[0] == -2.0

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d=np.ones(1))
    )

    g1 = al.Galaxy(
        redshift=1.0, light_profile=al.m.MockLightProfile(image_2d=2.0 * np.ones(1))
    )

    g2 = al.Galaxy(
        redshift=1.0, light_profile=al.m.MockLightProfile(image_2d=3.0 * np.ones(1))
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == -4.0
    assert fit.subtracted_images_of_planes_list[1].slim[0] == -0.0


def test___stochastic_mode__gives_different_log_likelihoods(masked_imaging_7x7):

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

    fit_0 = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=False),
    )
    fit_1 = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=False),
    )

    assert fit_0.log_evidence == fit_1.log_evidence

    fit_0 = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=True),
    )
    fit_1 = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_pixelization=al.SettingsPixelization(is_stochastic=True),
    )

    assert fit_0.log_evidence != fit_1.log_evidence


def test__preloads__refit_with_new_preloads(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    refit = fit.refit_with_new_preloads(preloads=al.Preloads())

    assert fit.figure_of_merit == refit.figure_of_merit

    refit = fit.refit_with_new_preloads(
        preloads=al.Preloads(blurred_image=fit.blurred_image + 1.0)
    )

    assert fit.figure_of_merit != refit.figure_of_merit


def test__preloads__blurred_image_uses_preload_when_passed(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(
        redshift=0.5, light_profile=al.m.MockLightProfile(image_2d=np.ones(1))
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert (fit.blurred_image == np.array([1.0])).all()

    blurred_image = np.array([2.0])
    preloads = al.Preloads(blurred_image=blurred_image)

    fit = al.FitImaging(
        dataset=masked_imaging_7x7_no_blur, tracer=tracer, preloads=preloads
    )

    assert (fit.blurred_image == np.array([2.0])).all()


def test__total_mappers(masked_imaging_7x7):

    g0 = al.Galaxy(redshift=0.5)

    g1 = al.Galaxy(redshift=1.0)

    g2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 0

    g2 = al.Galaxy(
        redshift=2.0,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 1

    g0 = al.Galaxy(
        redshift=0.5,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    g1 = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    g2 = al.Galaxy(
        redshift=2.0,
        pixelization=al.pix.Rectangular(),
        regularization=al.reg.Constant(),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.total_mappers == 3
