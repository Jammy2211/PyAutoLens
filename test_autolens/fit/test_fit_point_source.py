import autolens as al
import numpy as np
import pytest
from autolens.mock import mock

from functools import partial


class TestAbstractFitPositionsSourcePlane:
    def test__furthest_separation_of_source_plane_positions(self):

        positions = al.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0)])
        noise_map = al.ValuesIrregular([[1.0, 1.0]])

        tracer = mock.MockTracer(traced_grid=positions)
        fit = al.FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit.furthest_separations_of_source_plane_positions.in_list == [1.0, 1.0]
        assert fit.max_separation_of_source_plane_positions == 1.0
        assert fit.max_separation_within_threshold(threshold=2.0) == True
        assert fit.max_separation_within_threshold(threshold=0.5) == False

        positions = al.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)])
        noise_map = al.ValuesIrregular([1.0, 1.0, 1.0])

        tracer = mock.MockTracer(traced_grid=positions)
        fit = al.FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit.furthest_separations_of_source_plane_positions.in_list == [
            3.0,
            2.0,
            3.0,
        ]
        assert fit.max_separation_of_source_plane_positions == 3.0
        assert fit.max_separation_within_threshold(threshold=3.5) == True
        assert fit.max_separation_within_threshold(threshold=2.0) == False
        assert fit.max_separation_within_threshold(threshold=0.5) == False

    def test__same_as_above_with_real_tracer(self):

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0)),
                al.Galaxy(redshift=1.0),
            ]
        )

        noise_map = al.ValuesIrregular([1.0, 1.0])

        positions = al.Grid2DIrregular([(1.0, 0.0), (-1.0, 0.0)])
        fit = al.FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )
        assert fit.max_separation_within_threshold(threshold=0.01)

        positions = al.Grid2DIrregular([(1.2, 0.0), (-1.0, 0.0)])
        fit = al.FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )
        assert fit.max_separation_within_threshold(threshold=0.3)
        assert not fit.max_separation_within_threshold(threshold=0.15)


class TestFitPositionsImage:
    def test__two_sets_of_positions__residuals_likelihood_correct(self):

        point_source = al.ps.Point(centre=(0.1, 0.1))
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
        )

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])
        model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        fit = al.FitPositionsImage(
            name="point_0",
            positions=positions,
            noise_map=noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit.model_positions.in_list == [(3.0, 1.0), (2.0, 3.0)]

        assert fit.model_positions.in_list == [(3.0, 1.0), (2.0, 3.0)]

        assert fit.noise_map.in_list == [0.5, 1.0]
        assert fit.residual_map.in_list == [np.sqrt(10.0), np.sqrt(2.0)]
        assert fit.normalized_residual_map.in_list == [
            np.sqrt(10.0) / 0.5,
            np.sqrt(2.0) / 1.0,
        ]
        assert fit.chi_squared_map.in_list == [
            (np.sqrt(10.0) / 0.5) ** 2,
            np.sqrt(2.0) ** 2.0,
        ]
        assert fit.chi_squared == pytest.approx(42.0, 1.0e-4)
        assert fit.noise_normalization == pytest.approx(2.28945, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-22.14472, 1.0e-4)

    def test__more_model_positions_than_data_positions__pairs_closest_positions(self):

        g0 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))

        tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])
        model_positions = al.Grid2DIrregular(
            [(3.0, 1.0), (2.0, 3.0), (1.0, 0.0), (0.0, 1.0)]
        )

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        fit = al.FitPositionsImage(
            name="point_0",
            positions=positions,
            noise_map=noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit.model_positions.in_list == [(1.0, 0.0), (2.0, 3.0)]
        assert fit.noise_map.in_list == [0.5, 1.0]
        assert fit.residual_map.in_list == [1.0, np.sqrt(2.0)]
        assert fit.normalized_residual_map.in_list == [2.0, np.sqrt(2.0) / 1.0]
        assert fit.chi_squared_map.in_list == [4.0, np.sqrt(2.0) ** 2.0]
        assert fit.chi_squared == pytest.approx(6.0, 1.0e-4)
        assert fit.noise_normalization == pytest.approx(2.289459, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-4.144729, 1.0e-4)

    def test__multi_plane_position_solving(self):

        grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=1)

        g0 = al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0))
        g1 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))
        g2 = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.1, 0.1)))

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])

        positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.01)

        fit_0 = al.FitPositionsImage(
            name="point_0",
            positions=positions,
            noise_map=noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        fit_1 = al.FitPositionsImage(
            name="point_1",
            positions=positions,
            noise_map=noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        scaling_factor = al.util.cosmology.scaling_factor_between_redshifts_from(
            redshift_0=0.5,
            redshift_1=1.0,
            redshift_final=2.0,
            cosmology=tracer.cosmology,
        )

        assert fit_0.model_positions[0, 0] == pytest.approx(
            scaling_factor * fit_1.model_positions[0, 0], 1.0e-1
        )
        assert fit_0.model_positions[0, 1] == pytest.approx(
            scaling_factor * fit_1.model_positions[0, 1], 1.0e-1
        )


class TestFitPositionsSource:
    def test__two_sets_of_positions__residuals_likelihood_correct(self):

        point_source = al.ps.PointSourceChi(centre=(0.0, 0.0))
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
        )

        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])

        fit = al.FitPositionsSource(
            name="point_0", positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit.model_positions.in_list == [(0.0, 1.0), (0.0, 2.0)]
        assert fit.noise_map.in_list == [0.5, 1.0]
        assert fit.residual_map.in_list == [1.0, 2.0]
        assert fit.normalized_residual_map.in_list == [1.0 / 0.5, 2.0 / 1.0]
        assert fit.chi_squared_map.in_list == [(1.0 / 0.5) ** 2.0, 2.0 ** 2.0]
        assert fit.chi_squared == pytest.approx(8.0, 1.0e-4)
        assert fit.noise_normalization == pytest.approx(2.28945, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-5.14472988, 1.0e-4)

        galaxy_mass = al.Galaxy(
            redshift=0.5,
            mass=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[galaxy_mass, galaxy_point_source])

        fit = al.FitPositionsSource(
            name="point_0", positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit.model_positions.in_list == [(0.0, 0.0), (0.0, 1.0)]
        assert fit.log_likelihood == pytest.approx(-1.6447298, 1.0e-4)

    def test__multi_plane_position_solving(self):

        g0 = al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0))
        g1 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))
        g2 = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.1, 0.1)))

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

        positions = al.Grid2DIrregular([(0.0, 1.0), (0.0, 2.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])

        traced_grids = tracer.traced_grids_of_planes_from_grid(grid=positions)

        fit_0 = al.FitPositionsSource(
            name="point_0", positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit_0.model_positions[0, 1] == pytest.approx(0.326054, 1.0e-1)
        assert fit_0.model_positions[1, 1] == pytest.approx(1.326054, 1.0e-1)

        assert (fit_0.model_positions == traced_grids[1]).all()

        fit_1 = al.FitPositionsSource(
            name="point_1", positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert (fit_1.model_positions == traced_grids[2]).all()


class TestFitFluxes:
    def test__one_set_of_fluxes__residuals_likelihood_correct(self):

        tracer = mock.MockTracer(
            magnification=al.ValuesIrregular([2.0, 2.0]),
            profile=al.ps.PointFlux(flux=2.0),
        )

        fluxes = al.ValuesIrregular([1.0, 2.0])
        noise_map = al.ValuesIrregular([3.0, 1.0])
        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

        fit = al.FitFluxes(
            name="point_0",
            fluxes=fluxes,
            noise_map=noise_map,
            positions=positions,
            tracer=tracer,
        )

        assert fit.fluxes.in_list == [1.0, 2.0]
        assert fit.noise_map.in_list == [3.0, 1.0]
        assert fit.model_fluxes.in_list == [4.0, 4.0]
        assert fit.residual_map.in_list == [-3.0, -2.0]
        assert fit.normalized_residual_map.in_list == [-1.0, -2.0]
        assert fit.chi_squared_map.in_list == [1.0, 4.0]
        assert fit.chi_squared == pytest.approx(5.0, 1.0e-4)
        assert fit.noise_normalization == pytest.approx(5.87297, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-5.43648, 1.0e-4)

    def test__use_real_tracer(self, gal_x1_mp):

        point_source = al.ps.PointFlux(centre=(0.1, 0.1), flux=2.0)
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_mp, galaxy_point_source])

        fluxes = al.ValuesIrregular([1.0, 2.0])
        noise_map = al.ValuesIrregular([3.0, 1.0])
        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

        fit = al.FitFluxes(
            name="point_0",
            fluxes=fluxes,
            noise_map=noise_map,
            positions=positions,
            tracer=tracer,
        )

        assert fit.model_fluxes.in_list[1] == pytest.approx(2.5, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-3.11702, 1.0e-4)

    def test__multi_plane_calculation(self, gal_x1_mp):

        g0 = al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0))
        g1 = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux(flux=1.0))
        g2 = al.Galaxy(redshift=2.0, point_1=al.ps.PointFlux(flux=2.0))

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1, g2])

        fluxes = al.ValuesIrregular([1.0])
        noise_map = al.ValuesIrregular([3.0])
        positions = al.Grid2DIrregular([(2.0, 0.0)])

        fit_0 = al.FitFluxes(
            name="point_0",
            fluxes=fluxes,
            noise_map=noise_map,
            positions=positions,
            tracer=tracer,
        )

        deflections_func = partial(
            tracer.deflections_between_planes_from_grid, plane_i=0, plane_j=1
        )

        magnification_0 = tracer.magnification_via_hessian_from_grid(
            grid=positions, deflections_func=deflections_func
        )

        assert fit_0.magnifications[0] == magnification_0

        fit_1 = al.FitFluxes(
            name="point_1",
            fluxes=fluxes,
            noise_map=noise_map,
            positions=positions,
            tracer=tracer,
        )

        deflections_func = partial(
            tracer.deflections_between_planes_from_grid, plane_i=0, plane_j=2
        )

        magnification_1 = tracer.magnification_via_hessian_from_grid(
            grid=positions, deflections_func=deflections_func
        )

        assert fit_1.magnifications[0] == magnification_1

        assert fit_0.magnifications[0] != pytest.approx(fit_1.magnifications[0], 1.0e-1)


class TestFitPointDict:
    def test__fits_dataset__positions_only(self):

        point_source = al.ps.Point(centre=(0.1, 0.1))
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
        )

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])
        model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        point_dataset_0 = al.PointDataset(
            name="point_0", positions=positions, positions_noise_map=noise_map
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

        fit = al.FitPointDict(
            point_dict=point_dict, tracer=tracer, positions_solver=positions_solver
        )

        assert fit["point_0"].positions.log_likelihood == pytest.approx(
            -22.14472, 1.0e-4
        )
        assert fit["point_0"].flux == None
        assert fit.log_likelihood == fit["point_0"].positions.log_likelihood

        point_dataset_1 = al.PointDataset(
            name="point_1", positions=positions, positions_noise_map=noise_map
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

        fit = al.FitPointDict(
            point_dict=point_dict, tracer=tracer, positions_solver=positions_solver
        )

        assert fit["point_0"].positions.log_likelihood == pytest.approx(
            -22.14472, 1.0e-4
        )
        assert fit["point_0"].flux == None
        assert fit["point_1"].positions == None
        assert fit["point_1"].flux == None
        assert fit.log_likelihood == fit["point_0"].positions.log_likelihood

    def test__fits_dataset__positions_and_flux(self):

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

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        point_dataset_0 = al.PointDataset(
            name="point_0",
            positions=positions,
            positions_noise_map=noise_map,
            fluxes=fluxes,
            fluxes_noise_map=flux_noise_map,
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

        fit = al.FitPointDict(
            point_dict=point_dict, tracer=tracer, positions_solver=positions_solver
        )

        assert fit["point_0"].positions.log_likelihood == pytest.approx(
            -22.14472, 1.0e-4
        )
        assert fit["point_0"].flux.log_likelihood == pytest.approx(-2.9920449, 1.0e-4)
        assert (
            fit.log_likelihood
            == fit["point_0"].positions.log_likelihood
            + fit["point_0"].flux.log_likelihood
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
            point_dict=point_dict, tracer=tracer, positions_solver=positions_solver
        )

        assert fit["point_0"].positions.log_likelihood == pytest.approx(
            -22.14472, 1.0e-4
        )
        assert fit["point_0"].flux.log_likelihood == pytest.approx(-2.9920449, 1.0e-4)
        assert fit["point_1"].positions == None
        assert fit["point_1"].flux == None
        assert (
            fit.log_likelihood
            == fit["point_0"].flux.log_likelihood
            + fit["point_0"].positions.log_likelihood
        )

    def test__model_has_image_and_source_chi_squared__fits_both_correctly(self):

        galaxy_point_image = al.Galaxy(
            redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1))
        )

        galaxy_point_source = al.Galaxy(
            redshift=1.0, point_1=al.ps.PointSourceChi(centre=(0.1, 0.1))
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), galaxy_point_image, galaxy_point_source]
        )

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])
        model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        point_dataset_0 = al.PointDataset(
            name="point_0", positions=positions, positions_noise_map=noise_map
        )

        point_dataset_1 = al.PointDataset(
            name="point_1", positions=positions, positions_noise_map=noise_map
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

        fit = al.FitPointDict(
            point_dict=point_dict, tracer=tracer, positions_solver=positions_solver
        )

        assert isinstance(fit["point_0"].positions, al.FitPositionsImage)
        assert isinstance(fit["point_1"].positions, al.FitPositionsSource)

        assert (
            fit["point_0"].positions.model_positions.in_list == model_positions.in_list
        )
        assert fit["point_1"].positions.model_positions.in_list == positions.in_list
