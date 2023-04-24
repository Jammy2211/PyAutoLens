import numpy as np
import pytest

import autolens as al


def test__noise_map__with_and_without_hyper_background(interferometer_7):

    g0 = al.legacy.Galaxy(
        redshift=0.5, bulge=al.m.MockLightProfile(image_2d=np.ones(9))
    )
    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0])

    fit = al.legacy.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()

    hyper_background_noise = al.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = al.legacy.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-33.400998, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(interferometer_7):

    hyper_background_noise = al.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = al.legacy.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.legacy.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.legacy.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-9648681.9168, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-9648681.9168, 1.0e-4)

    fit = al.legacy.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        use_hyper_scaling=False,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map == pytest.approx(interferometer_7.noise_map, 1.0e-4)

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    g0 = al.legacy.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.legacy.Tracer.from_galaxies(
        galaxies=[al.legacy.Galaxy(redshift=0.5), g0]
    )

    fit = al.legacy.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-68.63380, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-68.63380, 1.0e-4)

    galaxy_light = al.legacy.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = al.legacy.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[galaxy_light, galaxy_pix])

    fit = al.legacy.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-892439.04665, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-892439.04665, 1.0e-4)
