import pytest

import autolens as al


def test__fits_dataset__positions_only():

    point_source = al.ps.Point(centre=(0.1, 0.1))
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer.from_galaxies(
        galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
    )

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ValuesIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    point_solver = al.m.MockPointSolver(model_positions=model_positions)

    point_dataset_0 = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise_map
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

    fit = al.FitPointDict(
        point_dict=point_dict, tracer=tracer, point_solver=point_solver
    )

    assert fit["point_0"].positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit["point_0"].flux == None
    assert fit.log_likelihood == fit["point_0"].positions.log_likelihood

    point_dataset_1 = al.PointDataset(
        name="point_1", positions=positions, positions_noise_map=noise_map
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

    fit = al.FitPointDict(
        point_dict=point_dict, tracer=tracer, point_solver=point_solver
    )

    assert fit["point_0"].positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit["point_0"].flux == None
    assert fit["point_1"].positions == None
    assert fit["point_1"].flux == None
    assert fit.log_likelihood == fit["point_0"].positions.log_likelihood


def test__fits_dataset__positions_and_flux():

    point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
    galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer.from_galaxies(
        galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
    )

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ValuesIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    fluxes = al.ValuesIrregular([1.0, 2.0])
    flux_noise_map = al.ValuesIrregular([3.0, 1.0])

    point_solver = al.m.MockPointSolver(model_positions=model_positions)

    point_dataset_0 = al.PointDataset(
        name="point_0",
        positions=positions,
        positions_noise_map=noise_map,
        fluxes=fluxes,
        fluxes_noise_map=flux_noise_map,
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

    fit = al.FitPointDict(
        point_dict=point_dict, tracer=tracer, point_solver=point_solver
    )

    assert fit["point_0"].positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit["point_0"].flux.log_likelihood == pytest.approx(-2.9920449, 1.0e-4)
    assert (
        fit.log_likelihood
        == fit["point_0"].positions.log_likelihood + fit["point_0"].flux.log_likelihood
    )

    point_dataset_1 = al.PointDataset(
        name="point_1",
        positions=positions,
        positions_noise_map=noise_map,
        fluxes=fluxes,
        fluxes_noise_map=flux_noise_map,
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

    fit = al.FitPointDict(
        point_dict=point_dict, tracer=tracer, point_solver=point_solver
    )

    assert fit["point_0"].positions.log_likelihood == pytest.approx(-22.14472, 1.0e-4)
    assert fit["point_0"].flux.log_likelihood == pytest.approx(-2.9920449, 1.0e-4)
    assert fit["point_1"].positions == None
    assert fit["point_1"].flux == None
    assert (
        fit.log_likelihood
        == fit["point_0"].flux.log_likelihood + fit["point_0"].positions.log_likelihood
    )


def test__model_has_image_and_source_chi_squared__fits_both_correctly():

    galaxy_point_image = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))

    galaxy_point_source = al.Galaxy(
        redshift=1.0, point_1=al.ps.PointSourceChi(centre=(0.1, 0.1))
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[al.Galaxy(redshift=0.5), galaxy_point_image, galaxy_point_source]
    )

    positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ValuesIrregular([0.5, 1.0])
    model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

    point_solver = al.m.MockPointSolver(model_positions=model_positions)

    point_dataset_0 = al.PointDataset(
        name="point_0", positions=positions, positions_noise_map=noise_map
    )

    point_dataset_1 = al.PointDataset(
        name="point_1", positions=positions, positions_noise_map=noise_map
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

    fit = al.FitPointDict(
        point_dict=point_dict, tracer=tracer, point_solver=point_solver
    )

    assert isinstance(fit["point_0"].positions, al.FitPositionsImage)
    assert isinstance(fit["point_1"].positions, al.FitPositionsSource)

    assert fit["point_0"].positions.model_positions.in_list == model_positions.in_list
    assert fit["point_1"].positions.model_positions.in_list == positions.in_list
