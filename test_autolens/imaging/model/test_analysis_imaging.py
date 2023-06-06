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
    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__positions__resample__raises_exception(masked_imaging_7x7):

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph()),
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

    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())
    source = al.Galaxy(redshift=1.0, light=al.lp.SersicSph())

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


def test__sets_up_adapt_galaxy_images__froms(masked_imaging_7x7):

    adapt_galaxy_image_path_dict = {
        ("galaxies", "lens"): al.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        ("galaxies", "source"): al.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    result = al.m.MockResult(
        adapt_galaxy_image_path_dict=adapt_galaxy_image_path_dict,
        adapt_model_image=al.Array2D.full(
            fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, adapt_result=result
    )

    assert (
        analysis.adapt_galaxy_image_path_dict[("galaxies", "lens")].native
        == np.ones((3, 3))
    ).all()

    assert (
        analysis.adapt_galaxy_image_path_dict[("galaxies", "source")].native
        == 2.0 * np.ones((3, 3))
    ).all()

    assert (analysis.adapt_model_image.native == 3.0 * np.ones((3, 3))).all()


def test__stochastic_log_likelihoods_for_instance(masked_imaging_7x7):

    lens_adapt_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
    lens_adapt_image[4] = 10.0
    source_adapt_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
    source_adapt_image[4] = 10.0
    adapt_model_image = al.Array2D.full(
        fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
    )

    adapt_galaxy_image_path_dict = {
        ("galaxies", "lens"): lens_adapt_image,
        ("galaxies", "source"): source_adapt_image,
    }

    result = al.m.MockResult(
        adapt_galaxy_image_path_dict=adapt_galaxy_image_path_dict,
        adapt_model_image=adapt_model_image,
    )

    pixelization = al.Pixelization(mesh=al.mesh.VoronoiMagnification(shape=(3, 3)))

    galaxies = af.ModelInstance()
    galaxies.lens = al.Galaxy(
        redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0)
    )
    galaxies.source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        adapt_result=result,
        settings_lens=al.SettingsLens(stochastic_samples=10),
    )

    stochastic_log_likelihoods = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert stochastic_log_likelihoods is None

    pixelization = al.Pixelization(mesh=al.mesh.VoronoiBrightnessImage(pixels=7))

    galaxies.source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    stochastic_log_likelihoods = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert sum(stochastic_log_likelihoods[0:5]) != pytest.approx(
        sum(stochastic_log_likelihoods[5:10], 1.0e-4)
    )

    pixelization = al.Pixelization(mesh=al.mesh.DelaunayBrightnessImage(pixels=5))

    galaxies.source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    stochastic_log_likelihoods = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert sum(stochastic_log_likelihoods[0:5]) != pytest.approx(
        sum(stochastic_log_likelihoods[5:10], 1.0e-4)
    )


def test__profile_log_likelihood_function(masked_imaging_7x7):

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    profiling_dict = analysis.profile_log_likelihood_function(instance=instance)

    assert "regularization_term_0" in profiling_dict
    assert "log_det_regularization_matrix_term_0" in profiling_dict
