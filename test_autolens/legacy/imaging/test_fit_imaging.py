import numpy as np
import pytest

import autolens as al


def test__noise_map__with_and_without_hyper_galaxy(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    fit = al.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=al.legacy.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        ),
        adapt_model_image=hyper_image,
        adapt_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    fit = al.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=4.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-20.7470, 1.0e-4)


def test__noise_map__with_hyper_galaxy_reaches_upper_limit(masked_imaging_7x7_no_blur):

    hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=al.legacy.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0e9, noise_power=1.0
        ),
        adapt_model_image=hyper_image,
        adapt_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    fit = al.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=1.0e8, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-174.0565, 1.0e-4)


def test__image__with_and_without_hyper_background_sky(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))
    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    fit = al.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.image.slim == pytest.approx(np.full(fill_value=1.0, shape=(9,)), 1.0e-1)

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))
    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    hyper_image_sky = al.legacy.hyper_data.HyperImageSky(sky_scale=1.0)

    fit = al.legacy.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
    )

    assert fit.image.slim == pytest.approx(np.full(fill_value=2.0, shape=(9,)), 1.0e-1)
    assert fit.log_likelihood == pytest.approx(-15.6337, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(masked_imaging_7x7_no_blur):

    g0 = al.Galaxy(redshift=0.5, bulge=al.m.MockLightProfile(image_2d_value=1.0))
    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    fit = al.legacy.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_background_noise = al.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = al.legacy.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=3.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-18.1579, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(masked_imaging_7x7):

    hyper_galaxy = al.legacy.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    hyper_image_sky = al.legacy.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = al.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
        hyper_galaxy=hyper_galaxy,
        adapt_model_image=np.ones(9),
        adapt_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.legacy.FitImaging(
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
        adapt_model_image=np.ones(9),
        adapt_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.legacy.FitImaging(
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
        adapt_model_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        adapt_galaxy_image=al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_minimum_value=0.0,
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.legacy.FitImaging(
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
