import numpy as np
from os import path
import pytest

import autofit as af

import autolens as al

from autolens.imaging.model.result import ResultImaging

from autolens import exc


directory = path.dirname(path.realpath(__file__))


def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
    masked_imaging_7x7
):

    hyper_image_sky = al.legacy.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = al.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    lens = al.legacy.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        galaxies=af.Collection(lens=lens),
    )

    analysis = al.legacy.AnalysisImaging(dataset=masked_imaging_7x7)
    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)
    fit = al.legacy.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.log_likelihood == analysis_log_likelihood


def test__uses_hyper_fit_correctly(masked_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.lens = al.legacy.Galaxy(
        redshift=0.5, light=al.lp.Sersic(intensity=1.0), mass=al.mp.IsothermalSph
    )
    galaxies.source = al.legacy.Galaxy(redshift=1.0, light=al.lp.Sersic())

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
    lens_hyper_image[4] = 10.0
    hyper_model_image = al.Array2D.full(
        fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
    )

    hyper_galaxy_image_path_dict = {("galaxies", "lens"): lens_hyper_image}

    result = al.m.MockResult(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        hyper_model_image=hyper_model_image,
    )

    analysis = al.legacy.AnalysisImaging(
        dataset=masked_imaging_7x7, hyper_dataset_result=result
    )

    hyper_galaxy = al.legacy.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    instance.galaxies.lens.hyper_galaxy = hyper_galaxy

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    g0 = al.legacy.Galaxy(
        redshift=0.5,
        light_profile=instance.galaxies.lens.light,
        mass_profile=instance.galaxies.lens.mass,
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=hyper_model_image,
        hyper_galaxy_image=lens_hyper_image,
        hyper_minimum_value=0.0,
    )
    g1 = al.legacy.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.legacy.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.tracer.galaxies[0].hyper_galaxy_image == lens_hyper_image).all()
    assert analysis_log_likelihood == fit.log_likelihood



