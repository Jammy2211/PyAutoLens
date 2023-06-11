from os import path
import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens import exc

from autolens.interferometer.model.result import ResultInterferometer

directory = path.dirname(path.realpath(__file__))


def test__make_result__result_interferometer_is_returned(interferometer_7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

    instance = model.instance_from_prior_medians()

    samples = al.m.MockSamples(max_log_likelihood_instance=instance)

    search = al.m.MockSearch(name="test_search", samples=samples)

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    def modify_after_fit(
        paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ):
        pass

    analysis.modify_after_fit = modify_after_fit

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultInterferometer)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(interferometer_7):
    lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens_galaxy))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__positions__resample__raises_exception(interferometer_7, mask_2d_7x7):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph()),
            source=al.Galaxy(redshift=1.0),
        )
    )

    positions_likelihood = al.PositionsLHResample(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisInterferometer(
        dataset=interferometer_7, positions_likelihood=positions_likelihood
    )

    instance = model.instance_from_unit_vector([])

    with pytest.raises(exc.RayTracingException):
        analysis.log_likelihood_function(instance=instance)


def test__positions__likelihood_overwrite__changes_likelihood(
    interferometer_7, mask_2d_7x7
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())
    source = al.Galaxy(redshift=1.0, light=al.lp.SersicSph())

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood
    assert analysis_log_likelihood == pytest.approx(-127914.36273, 1.0e-4)

    positions_likelihood = al.PositionsLHPenalty(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisInterferometer(
        dataset=interferometer_7, positions_likelihood=positions_likelihood
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    log_likelihood_penalty_base = positions_likelihood.log_likelihood_penalty_base_from(
        dataset=interferometer_7
    )
    log_likelihood_penalty = positions_likelihood.log_likelihood_penalty_from(
        tracer=tracer
    )

    assert analysis_log_likelihood == pytest.approx(
        log_likelihood_penalty_base - log_likelihood_penalty, 1.0e-4
    )
    assert analysis_log_likelihood == pytest.approx(-22048700567.590656, 1.0e-4)


def test__sets_up_adapt_galaxy_images(interferometer_7):
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

    analysis = al.AnalysisInterferometer(dataset=interferometer_7, adapt_result=result)

    analysis.set_adapt_dataset(result=result)

    assert (
        analysis.adapt_galaxy_image_path_dict[("galaxies", "lens")].native
        == np.ones((3, 3))
    ).all()

    assert (
        analysis.adapt_galaxy_image_path_dict[("galaxies", "source")].native
        == 2.0 * np.ones((3, 3))
    ).all()

    assert (analysis.adapt_model_image.native == 3.0 * np.ones((3, 3))).all()


def test__stochastic_log_likelihoods_for_instance(interferometer_7):
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

    analysis = al.AnalysisInterferometer(
        dataset=interferometer_7,
        settings_lens=al.SettingsLens(stochastic_samples=2),
        adapt_result=result,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    pixelization = al.Pixelization(mesh=al.mesh.VoronoiBrightnessImage(pixels=5))

    galaxies = af.ModelInstance()
    galaxies.source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    log_evidences = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert len(log_evidences) == 2
    assert log_evidences[0] != log_evidences[1]

    galaxies.source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    log_evidences = analysis.stochastic_log_likelihoods_via_instance_from(
        instance=instance
    )

    assert len(log_evidences) == 2
    assert log_evidences[0] != log_evidences[1]


def test__profile_log_likelihood_function(interferometer_7):
    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    run_time_dict, info_dict = analysis.profile_log_likelihood_function(
        instance=instance
    )

    assert "regularization_term_0" in run_time_dict
    assert "log_det_regularization_matrix_term_0" in run_time_dict
