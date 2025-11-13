from os import path
import pytest

import autofit as af

import autolens as al

from autolens.imaging.model.result import ResultImaging

from autolens import exc


directory = path.dirname(path.realpath(__file__))


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):

    model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    search = al.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImaging)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    masked_imaging_7x7,
):
    lens = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == analysis_log_likelihood


def test__positions__likelihood_overwrites__changes_likelihood(masked_imaging_7x7):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.05, 0.05)))
    source = al.Galaxy(redshift=1.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_unit_vector([])

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.log_likelihood == pytest.approx(analysis_log_likelihood, 1.0e-4)
    assert analysis_log_likelihood == pytest.approx(-14.79034680979, 1.0e-4)

    positions_likelihood = al.PositionsLH(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood_list=[positions_likelihood], use_jax=False
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    assert analysis_log_likelihood == pytest.approx(-44097289521.734665, 1.0e-4)


def test__positions__likelihood_overwrites__changes_likelihood__double_source_plane_example(masked_imaging_7x7):

    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.05, 0.05)))
    source_0 = al.Galaxy(redshift=1.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))
    source_1 = al.Galaxy(redshift=2.0, light=al.lp.SersicSph(centre=(0.05, 0.05)))

    model = af.Collection(galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1))

    instance = model.instance_from_unit_vector([])

    positions_likelihood_0 = al.PositionsLH(
        plane_redshift=1.0, positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )
    positions_likelihood_1 = al.PositionsLH(
        plane_redshift=2.0, positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood_list=[positions_likelihood_0, positions_likelihood_1], use_jax=False
    )
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    assert analysis_log_likelihood == pytest.approx(-44097289521.734665, 1.0e-4)

