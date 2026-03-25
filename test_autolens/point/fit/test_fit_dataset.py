import pytest

import autolens as al


@pytest.fixture
def point_source_tracer():
    point_source = al.ps.Point(centre=(0.1, 0.1))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    return al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])


@pytest.fixture
def positions_and_noise():
    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    return positions, noise_map


@pytest.fixture
def mock_solver():
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])
    return al.m.MockPointSolver(model_positions=model_positions)


def test__fit_dataset__matching_point_name__positions_log_likelihood_correct(
    point_source_tracer, positions_and_noise, mock_solver
):
    positions, noise_map = positions_and_noise
    dataset = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise_map
    )

    fit = al.FitPointDataset(
        dataset=dataset, tracer=point_source_tracer, solver=mock_solver
    )

    assert fit.positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit.flux is None


def test__fit_dataset__nonmatching_point_name__positions_and_flux_are_none(
    point_source_tracer, positions_and_noise, mock_solver
):
    positions, noise_map = positions_and_noise
    dataset = al.PointDataset(
        name="point_1", positions=positions, positions_noise_map=noise_map
    )

    fit = al.FitPointDataset(
        dataset=dataset, tracer=point_source_tracer, solver=mock_solver
    )

    assert fit.positions is None
    assert fit.flux is None


@pytest.mark.parametrize(
    "fit_positions_cls, expected_log_likelihood",
    [
        (al.FitPositionsImagePair,       -22.14472),
        (al.FitPositionsImagePairRepeat, -22.14472),
        (al.FitPositionsImagePairAll,    -24.6435280294),
        (al.FitPositionsSource,          -12.9947298),
    ],
    ids=[
        "FitPositionsImagePair",
        "FitPositionsImagePairRepeat",
        "FitPositionsImagePairAll",
        "FitPositionsSource",
    ],
)
def test__fit_dataset__log_likelihood_correct_for_each_position_fitting_class(
    point_source_tracer, positions_and_noise, mock_solver,
    fit_positions_cls, expected_log_likelihood,
):
    positions, noise_map = positions_and_noise
    dataset = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise_map
    )

    fit = al.FitPointDataset(
        dataset=dataset,
        tracer=point_source_tracer,
        solver=mock_solver,
        fit_positions_cls=fit_positions_cls,
    )

    assert fit.positions.log_likelihood == pytest.approx(expected_log_likelihood, 1.0e-4)


def test__fit_dataset__positions_and_flux__both_log_likelihoods_correct_and_sum():
    point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])
    fluxes = al.ArrayIrregular([1.0, 2.0])
    flux_noise_map = al.ArrayIrregular([3.0, 1.0])

    solver = al.m.MockPointSolver(model_positions=model_positions)

    dataset = al.PointDataset(
        name="point_0",
        positions=positions,
        positions_noise_map=noise_map,
        fluxes=fluxes,
        fluxes_noise_map=flux_noise_map,
    )

    fit = al.FitPointDataset(dataset=dataset, tracer=tracer, solver=solver)

    assert fit.positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit.flux.log_likelihood == pytest.approx(-2.9920449, 1.0e-4)
    assert fit.log_likelihood == fit.positions.log_likelihood + fit.flux.log_likelihood
