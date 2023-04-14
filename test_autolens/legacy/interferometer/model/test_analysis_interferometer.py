from os import path

import autofit as af
import autolens as al

directory = path.dirname(path.realpath(__file__))


def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
    interferometer_7,
):
    hyper_background_noise = al.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    lens_galaxy = al.legacy.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(
        galaxies=af.Collection(lens=lens_galaxy),
        hyper_background_noise=hyper_background_noise,
    )

    analysis = al.legacy.AnalysisInterferometer(dataset=interferometer_7)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.legacy.FitInterferometer(
        dataset=interferometer_7,
        tracer=tracer,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.log_likelihood == analysis_log_likelihood

