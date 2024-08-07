import pytest

import autolens as al


def test__fit_dataset__positions_only():
    point_source = al.ps.Point(centre=(0.1, 0.1))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    solver = al.m.MockPointSolver(model_positions=model_positions)

    dataset_0 = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise_map
    )

    fit = al.FitPointDataset(dataset=dataset_0, tracer=tracer, solver=solver)

    assert fit.positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit.flux == None

    dataset_1 = al.PointDataset(
        name="point_1", positions=positions, positions_noise_map=noise_map
    )

    fit = al.FitPointDataset(dataset=dataset_1, tracer=tracer, solver=solver)

    assert fit.flux == None
    assert fit.positions == None
    assert fit.flux == None


def test__fit_dataset__all_positions_classes():
    point_source = al.ps.Point(centre=(0.1, 0.1))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    solver = al.m.MockPointSolver(model_positions=model_positions)

    dataset_0 = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise_map
    )

    fit = al.FitPointDataset(
        dataset=dataset_0,
        tracer=tracer,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePair,
    )

    assert fit.positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)

    fit = al.FitPointDataset(
        dataset=dataset_0,
        tracer=tracer,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairRepeat,
    )

    assert fit.positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)

    fit = al.FitPointDataset(
        dataset=dataset_0,
        tracer=tracer,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairAll,
    )

    assert fit.positions.log_likelihood == pytest.approx(-47.78945977, 1.0e-4)

    fit = al.FitPointDataset(
        dataset=dataset_0,
        tracer=tracer,
        solver=solver,
        fit_positions_cls=al.FitPositionsSource,
    )

    assert fit.positions.log_likelihood == pytest.approx(-12.9947298, 1.0e-4)


def test__fit_dataset__positions_and_flux():
    point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source])

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    fluxes = al.ArrayIrregular([1.0, 2.0])
    flux_noise_map = al.ArrayIrregular([3.0, 1.0])

    solver = al.m.MockPointSolver(model_positions=model_positions)

    dataset_0 = al.PointDataset(
        name="point_0",
        positions=positions,
        positions_noise_map=noise_map,
        fluxes=fluxes,
        fluxes_noise_map=flux_noise_map,
    )

    fit = al.FitPointDataset(dataset=dataset_0, tracer=tracer, solver=solver)

    assert fit.positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit.flux.log_likelihood == pytest.approx(-2.9920449, 1.0e-4)
    assert fit.log_likelihood == fit.positions.log_likelihood + fit.flux.log_likelihood
