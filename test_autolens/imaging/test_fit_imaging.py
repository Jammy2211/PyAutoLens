import numpy as np
import pytest

import autolens as al


def test__model_image__with_and_without_psf_blurring(
    masked_imaging_7x7_no_blur, masked_imaging_7x7
):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
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

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0),
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
        bulge=al.m.MockLightProfile(image_2d_value=1.0),
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

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))
    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.image.slim == pytest.approx(np.full(fill_value=1.0, shape=(9,)), 1.0e-1)

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))
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

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))
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


def test__fit_figure_of_merit(masked_imaging_7x7, masked_imaging_covariance_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        disk=al.lp.Sersic(intensity=2.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-2859741.44762, 1.0e-4)

    basis = al.lp_basis.Basis(
        light_profile_list=[
            al.lp.Sersic(intensity=1.0),
            al.lp.Sersic(intensity=2.0),
        ]
    )

    g0 = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-2859741.44762, 1.0e-4)

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.90055, 1.0e-4)

    galaxy_light = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-37667.0303, 1.0e-4)

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0_linear, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-6741.83381, 1.0e-4)

    basis = al.lp_basis.Basis(
        light_profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ]
    )

    g0_linear = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0_linear, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-6741.83381, 1.0e-4)

    basis = al.lp_basis.Basis(
        light_profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ],
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0_basis = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0_basis, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-208205.2074336, 1.0e-4)

    tracer = al.Tracer.from_galaxies(galaxies=[g0_linear, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.79906, 1.0e-4)

    g0_operated = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_operated = al.Galaxy(redshift=1.0, bulge=al.lp_operated.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0_operated, g1_operated])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-2657889.4489, 1.0e-4)

    g0_linear_operated = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear_operated.Sersic(sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_linear_operated = al.Galaxy(
        redshift=1.0, bulge=al.lp_linear_operated.Sersic(sersic_index=4.0)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0_linear_operated, g1_linear_operated])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.9881985, 1.0e-4)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        disk=al.lp.Sersic(intensity=2.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_covariance_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-3688191.0841, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(masked_imaging_7x7):

    hyper_galaxy = al.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

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

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0 = al.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
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
        bulge=al.lp.Sersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_galaxy_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_minimum_value=0.0,
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

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

    # Normal Light Profiles Only

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image_2d_list = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(
        blurred_image_2d_list[0], 1.0e-4
    )
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(
        blurred_image_2d_list[1], 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native + fit.galaxy_model_image_dict[g1].native,
        1.0e-4,
    )

    # Linear Light Profiles Only

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic())

    tracer = al.Tracer.from_galaxies(galaxies=[g0_linear, g1_linear, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -4.99645959e-01, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[g1_linear][4] == pytest.approx(
        1.9986499980, 1.0e-2
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

    assert fit.model_image == pytest.approx(
        fit.galaxy_model_image_dict[g0_linear] + fit.galaxy_model_image_dict[g1_linear],
        1.0e-4,
    )

    # Pixelization + Regularizaiton only

    g0_no_light = al.Galaxy(
        redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(galaxies=[g0_no_light, galaxy_pix_0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.galaxy_model_image_dict[g0_no_light] == np.zeros(9)).all()
    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(
        1.259965886, 1.0e-4
    )

    assert fit.model_image == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix_0], 1.0e-4
    )

    # Normal light + Linear Light PRofiles + Pixelization + Regularizaiton

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(
        galaxies=[g0, g0_linear, g2, galaxy_pix_0, galaxy_pix_1]
    )

    masked_imaging_7x7.image[0] = 3.0

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer, settings_inversion=al.SettingsInversion(use_w_tilde=False))

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(
        blurred_image_2d_list[0], 1.0e-4
    )

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -650.736682, 1.0e-4
    )

    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(
        1.08219997, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[galaxy_pix_1][4] == pytest.approx(
        1.0822004, 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()


def test__model_images_of_planes_list(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_linear = al.Galaxy(redshift=0.75, bulge=al.lp_linear.Sersic())

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(
        galaxies=[g0, g1_linear, galaxy_pix_0, galaxy_pix_1]
    )

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer, settings_inversion=al.SettingsInversion(use_w_tilde=False))

    assert fit.model_images_of_planes_list[0] == pytest.approx(
        fit.galaxy_model_image_dict[g0], 1.0e-4
    )
    assert fit.model_images_of_planes_list[1] == pytest.approx(
        fit.galaxy_model_image_dict[g1_linear], 1.0e-4
    )
    assert fit.model_images_of_planes_list[2] == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix_0]
        + fit.galaxy_model_image_dict[galaxy_pix_1],
        1.0e-4,
    )


def test___unmasked_blurred_images(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_images_of_planes = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    unmasked_blurred_image = tracer.unmasked_blurred_image_2d_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (fit.unmasked_blurred_image == unmasked_blurred_image).all()

    unmasked_blurred_image_of_planes_list = tracer.unmasked_blurred_image_2d_list_from(
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


def test__subtracted_images_of_planes_list(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    g1 = al.Galaxy(redshift=0.75, bulge=al.lp.Sersic(intensity=2.0))

    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == pytest.approx(0.200638, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[1].slim[0] == pytest.approx(0.360511, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[2].slim[0] == pytest.approx(0.520383, 1.0e-4)

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))

    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == pytest.approx(0.200638, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[1].slim[0] == pytest.approx(0.840127, 1.0e-4)


def test__tracer_linear_light_profiles_to_light_profiles(masked_imaging_7x7):

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic(sersic_index=4.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g0_linear, g1_linear])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.tracer.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)

    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    assert tracer.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)
    assert tracer.galaxies[1].bulge.intensity == pytest.approx(-371.061130, 1.0e-4)
    assert tracer.galaxies[2].bulge.intensity == pytest.approx(0.08393533428, 1.0e-4)


def _test___stochastic_mode__gives_different_log_likelihoods(masked_imaging_7x7):

    pixelization = al.Pixelization(
        mesh=al.mesh.VoronoiBrightnessImage(pixels=7),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0 = al.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
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

    # Sum 5 stochastic likelihoods to avoid random chance of identical
    # pixelizations and therefore likelihoods.

    log_evidence_x5_0 = sum([fit_0.log_evidence for i in range(5)])
    log_evidence_x5_1 = sum([fit_1.log_evidence for i in range(5)])

    assert log_evidence_x5_0 != log_evidence_x5_1


def test__preloads__refit_with_new_preloads(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    refit = fit.refit_with_new_preloads(preloads=al.Preloads())

    assert fit.figure_of_merit == refit.figure_of_merit

    refit = fit.refit_with_new_preloads(
        preloads=al.Preloads(blurred_image=fit.blurred_image + 1.0)
    )

    assert fit.figure_of_merit != refit.figure_of_merit


def test__preloads__blurred_image_uses_preload_when_passed(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer.from_galaxies(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.blurred_image[0] == pytest.approx(0.15987, 1.0e-4)

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

    pixelization = al.Pixelization(mesh=al.mesh.Rectangular())

    g2 = al.Galaxy(redshift=2.0, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 1

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    g1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    g2 = al.Galaxy(redshift=2.0, pixelization=pixelization)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.total_mappers == 3
