import autolens as al
import numpy as np
import pytest
from autolens.mock import mock


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
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.0)
                ),
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


# class TestFitPositionsSourcePlane:
#     def test__likelihood__is_sum_of_separations_divided_by_noise(self):
#
#         positions = al.Grid2DIrregular(
#             [
#                 [(0.0, 0.0), (0.0, 1.0), (0.0, 0.5)],
#                 [(0.0, 0.0), (0.0, 0.0), (3.0, 3.0)],
#                 [(0.0, 0.0), (1.0, 1.0), (3.0, 3.0)],
#             ]
#         )
#
#         noise_map = al.ValuesIrregular(
#             [
#                 [1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#             ]
#         )
#
#         tracer = MockTracerPositions(positions=positions)
#
#         fit = al.FitPositionsSourceMaxSeparation(
#             positions=positions, noise_map=noise_map, tracer=tracer,
#         )
#         assert fit.chi_squared_map[0] == 1.0
#         assert fit.chi_squared_map[1] == pytest.approx(18.0, 1e-4)
#         assert fit.chi_squared_map[2] == pytest.approx(18.0, 1e-4)
#         assert fit.figure_of_merit == pytest.approx(-0.5 * (1.0 + 18 + 18), 1e-4)
#
#         fit = al.FitPositionsSourceMaxSeparation(
#             positions=positions, noise_map=noise_map, tracer=tracer,
#         )
#         assert fit.chi_squared_map[0] == (1.0 / 2.0) ** 2.0
#         assert fit.chi_squared_map[1] == pytest.approx(
#             (np.sqrt(18.0) / 2.0) ** 2.0, 1e-4
#         )
#         assert fit.chi_squared_map[2] == pytest.approx(
#             (np.sqrt(18.0) / 2.0) ** 2.0, 1e-4
#         )
#         assert fit.figure_of_merit == pytest.approx(
#             -0.5
#             * (
#                 (1.0 / 2.0) ** 2.0
#                 + (np.sqrt(18.0) / 2.0) ** 2.0
#                 + (np.sqrt(18.0) / 2.0) ** 2.0
#             ),
#             1e-4,
#         )


class TestFitPositionsImage:
    def test__two_sets_of_positions__residuals_likelihood_correct(self):

        point_source = al.ps.PointSource(centre=(0.1, 0.1))
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
        )

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])
        model_positions = al.Grid2DIrregular([(3.0, 1.0), (2.0, 3.0)])

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        fit = al.FitPositionsImage(
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

        point_source = al.ps.PointSource(centre=(0.1, 0.1))
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), galaxy_point_source]
        )

        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
        noise_map = al.ValuesIrregular([0.5, 1.0])
        model_positions = al.Grid2DIrregular(
            [(3.0, 1.0), (2.0, 3.0), (1.0, 0.0), (0.0, 1.0)]
        )

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        fit = al.FitPositionsImage(
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


class TestFitFluxes:
    def test__one_set_of_fluxes__residuals_likelihood_correct(self):

        tracer = mock.MockTracer(
            magnification=al.ValuesIrregular([2.0, 2.0]), attribute=2.0
        )

        fluxes = al.ValuesIrregular([1.0, 2.0])
        noise_map = al.ValuesIrregular([3.0, 1.0])
        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

        fit = al.FitFluxes(
            fluxes=fluxes, noise_map=noise_map, positions=positions, tracer=tracer
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

        point_source = al.ps.PointSourceFlux(centre=(0.1, 0.1), flux=2.0)
        galaxy_point_source = al.Galaxy(redshift=1.0, point_0=point_source)
        tracer = al.Tracer.from_galaxies(galaxies=[gal_x1_mp, galaxy_point_source])

        fluxes = al.ValuesIrregular([1.0, 2.0])
        noise_map = al.ValuesIrregular([3.0, 1.0])
        positions = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])

        fit = al.FitFluxes(
            fluxes=fluxes, noise_map=noise_map, positions=positions, tracer=tracer
        )

        assert fit.model_fluxes.in_list[1] == pytest.approx(2.5, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-3.11702, 1.0e-4)
