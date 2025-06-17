from os import path
import pytest

import autofit as af
import autolens as al
from autolens import exc

from autolens.interferometer.model.result import ResultInterferometer

directory = path.dirname(path.realpath(__file__))


def test__make_result__result_interferometer_is_returned(interferometer_7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    search = al.m.MockSearch(name="test_search")

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


def test__positions__likelihood_overwrite__changes_likelihood(
    interferometer_7, mask_2d_7x7
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.05, 0.05)))
    source = al.Galaxy(redshift=1.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisInterferometer(dataset=interferometer_7)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitInterferometer(dataset=interferometer_7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood
    assert analysis_log_likelihood == pytest.approx(-62.463179940, 1.0e-4)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisInterferometer(
        dataset=interferometer_7, positions_likelihood_list=[positions_likelihood]
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    assert analysis_log_likelihood == pytest.approx(44097289444.30784, 1.0e-4)


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
