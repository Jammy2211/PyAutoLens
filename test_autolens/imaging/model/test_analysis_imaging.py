import numpy as np
from os import path
import pytest

import autofit as af

import autolens as al

from autolens.imaging.model.result import ResultImaging

from autolens import exc


directory = path.dirname(path.realpath(__file__))


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):

    model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

    instance = model.instance_from_prior_medians()

    samples = al.m.MockSamples(max_log_likelihood_instance=instance)

    search = al.m.MockSearch(name="test_search", samples=samples)

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    def modify_after_fit(
        paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ):
        pass

    analysis.modify_after_fit = modify_after_fit

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImaging)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    masked_imaging_7x7
):
    lens = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
    masked_imaging_7x7
):

    hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    lens = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

    model = af.Collection(
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        galaxies=af.Collection(lens=lens),
    )

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.log_likelihood == analysis_log_likelihood


def test__positions__resample__raises_exception(masked_imaging_7x7):

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal()),
            source=al.Galaxy(redshift=1.0),
        )
    )

    positions_likelihood = al.PositionsLHResample(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood=positions_likelihood
    )

    instance = model.instance_from_unit_vector([])

    with pytest.raises(exc.RayTracingException):
        analysis.log_likelihood_function(instance=instance)


def test__positions__likelihood_overwrites__changes_likelihood(masked_imaging_7x7):

    lens = al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal())
    source = al.Galaxy(redshift=1.0, light=al.lp.SphSersic())

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == pytest.approx(analysis_log_likelihood, 1.0e-4)
    assert analysis_log_likelihood == pytest.approx(-6258.043397009, 1.0e-4)

    positions_likelihood = al.PositionsLHPenalty(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood=positions_likelihood
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    log_likelihood_penalty_base = positions_likelihood.log_likelihood_penalty_base_from(
        dataset=masked_imaging_7x7
    )
    log_likelihood_penalty = positions_likelihood.log_likelihood_penalty_from(
        tracer=tracer
    )

    assert analysis_log_likelihood == pytest.approx(
        log_likelihood_penalty_base - log_likelihood_penalty, 1.0e-4
    )
    assert analysis_log_likelihood == pytest.approx(-22048700558.9052, 1.0e-4)


def test__uses_hyper_fit_correctly(masked_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.lens = al.Galaxy(
        redshift=0.5, light=al.lp.EllSersic(intensity=1.0), mass=al.mp.SphIsothermal
    )
    galaxies.source = al.Galaxy(redshift=1.0, light=al.lp.EllSersic())

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

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, hyper_dataset_result=result
    )

    hyper_galaxy = al.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    instance.galaxies.lens.hyper_galaxy = hyper_galaxy

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=instance.galaxies.lens.light,
        mass_profile=instance.galaxies.lens.mass,
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=hyper_model_image,
        hyper_galaxy_image=lens_hyper_image,
        hyper_minimum_value=0.0,
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.tracer.galaxies[0].hyper_galaxy_image == lens_hyper_image).all()
    assert analysis_log_likelihood == fit.log_likelihood


def test__sets_up_hyper_galaxy_images__froms(masked_imaging_7x7):

    hyper_galaxy_image_path_dict = {
        ("galaxies", "lens"): al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        ("galaxies", "source"): al.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    result = al.m.MockResult(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        hyper_model_image=al.Array2D.full(
            fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, hyper_dataset_result=result
    )

    assert (
        analysis.hyper_galaxy_image_path_dict[("galaxies", "lens")].native
        == np.ones((3, 3))
    ).all()

    assert (
        analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].native
        == 2.0 * np.ones((3, 3))
    ).all()

    assert (analysis.hyper_model_image.native == 3.0 * np.ones((3, 3))).all()


def test__stochastic_log_likelihoods_for_instance(masked_imaging_7x7):

    lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
    lens_hyper_image[4] = 10.0
    source_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
    source_hyper_image[4] = 10.0
    hyper_model_image = al.Array2D.full(
        fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
    )

    hyper_galaxy_image_path_dict = {
        ("galaxies", "lens"): lens_hyper_image,
        ("galaxies", "source"): source_hyper_image,
    }

    result = al.m.MockResult(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        hyper_model_image=hyper_model_image,
    )

    galaxies = af.ModelInstance()
    galaxies.lens = al.Galaxy(
        redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0)
    )
    galaxies.source = al.Galaxy(
        redshift=1.0,
        pixelization=al.mesh.VoronoiMagnification(shape=(3, 3)),
        regularization=al.reg.Constant(),
    )

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        hyper_dataset_result=result,
        settings_lens=al.SettingsLens(stochastic_samples=10),
    )

    stochastic_log_likelihoods = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert stochastic_log_likelihoods is None

    galaxies.source = al.Galaxy(
        redshift=1.0,
        pixelization=al.mesh.VoronoiBrightnessImage(pixels=5),
        regularization=al.reg.Constant(),
    )

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    stochastic_log_likelihoods = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert sum(stochastic_log_likelihoods[0:5]) != pytest.approx(
        sum(stochastic_log_likelihoods[5:10], 1.0e-4)
    )

    galaxies.source = al.Galaxy(
        redshift=1.0,
        pixelization=al.mesh.DelaunayBrightnessImage(pixels=5),
        regularization=al.reg.Constant(),
    )

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    stochastic_log_likelihoods = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert sum(stochastic_log_likelihoods[0:5]) != pytest.approx(
        sum(stochastic_log_likelihoods[5:10], 1.0e-4)
    )


def test__profile_log_likelihood_function(masked_imaging_7x7):

    lens = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))
    source = al.Galaxy(
        redshift=1.0,
        regularization=al.reg.Constant(coefficient=1.0),
        pixelization=al.mesh.Rectangular(shape=(3, 3)),
    )

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    profiling_dict = analysis.profile_log_likelihood_function(instance=instance)

    assert "regularization_term_0" in profiling_dict
    assert "log_det_regularization_matrix_term_0" in profiling_dict
