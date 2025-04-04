from os import path

import autofit as af
import autolens as al

from autolens.point.model.result import ResultPoint

directory = path.dirname(path.realpath(__file__))


def _test__make_result__result_imaging_is_returned(point_dataset):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
        )
    )

    search = al.m.MockSearch(name="test_search")

    solver = al.m.MockPointSolver(model_positions=point_dataset.positions)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver)

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultPoint)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    positions_x2, positions_x2_noise_map
):
    point_dataset = al.PointDataset(
        name="point_0",
        positions=positions_x2,
        positions_noise_map=positions_x2_noise_map,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
        )
    )

    solver = al.m.MockPointSolver(model_positions=positions_x2)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit_positions.chi_squared == 0.0
    assert fit_positions.log_likelihood == analysis_log_likelihood

    model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
    solver = al.m.MockPointSolver(model_positions=model_positions)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver)

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    assert fit_positions.residual_map.in_list == [1.0, 1.0]
    assert fit_positions.chi_squared == 2.0
    assert fit_positions.log_likelihood == analysis_log_likelihood


def test__figure_of_merit__includes_fit_fluxes(
    positions_x2, positions_x2_noise_map, fluxes_x2, fluxes_x2_noise_map
):
    point_dataset = al.PointDataset(
        name="point_0",
        positions=positions_x2,
        positions_noise_map=positions_x2_noise_map,
        fluxes=fluxes_x2,
        fluxes_noise_map=fluxes_x2_noise_map,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                sis=al.mp.IsothermalSph(einstein_radius=1.0),
                point_0=al.ps.PointFlux(flux=1.0),
            )
        )
    )

    solver = al.m.MockPointSolver(model_positions=positions_x2)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver)

    instance = model.instance_from_unit_vector([])

    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    tracer = analysis.tracer_via_instance_from(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    fit_fluxes = al.FitFluxes(
        name="point_0",
        data=fluxes_x2,
        noise_map=fluxes_x2_noise_map,
        positions=positions_x2,
        tracer=tracer,
    )

    assert (
        fit_positions.log_likelihood + fit_fluxes.log_likelihood
        == analysis_log_likelihood
    )

    model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
    solver = al.m.MockPointSolver(model_positions=model_positions)

    analysis = al.AnalysisPoint(dataset=point_dataset, solver=solver)

    instance = model.instance_from_unit_vector([])
    analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

    fit_positions = al.FitPositionsImagePairRepeat(
        name="point_0",
        data=positions_x2,
        noise_map=positions_x2_noise_map,
        tracer=tracer,
        solver=solver,
    )

    fit_fluxes = al.FitFluxes(
        name="point_0",
        data=fluxes_x2,
        noise_map=fluxes_x2_noise_map,
        positions=positions_x2,
        tracer=tracer,
    )

    assert fit_positions.residual_map.in_list == [1.0, 1.0]
    assert fit_positions.chi_squared == 2.0
    assert (
        fit_positions.log_likelihood + fit_fluxes.log_likelihood
        == analysis_log_likelihood
    )
