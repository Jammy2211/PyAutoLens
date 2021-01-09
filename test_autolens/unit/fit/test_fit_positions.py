import autolens as al
import numpy as np
import pytest
from autolens.mock import mock


class MockTracerPositions:
    def __init__(self, positions, noise=None):
        self.positions = positions
        self.noise = noise

    def traced_grids_of_planes_from_grid(self, grid, plane_index_limit=None):
        return [self.positions]


class TestAbstractFitPositionsSourcePlane:
    def test__furthest_separation_of_source_plane_positions(self):

        positions = al.GridIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 1.0)]])
        noise_map = al.ValuesIrregularGrouped([[1.0, 1.0]])

        tracer = MockTracerPositions(positions=positions)
        fit = al.FitPositionsSourcePlaneMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit.furthest_separations_of_source_plane_positions.in_grouped_list == [
            [1.0, 1.0]
        ]
        assert fit.max_separation_of_source_plane_positions == 1.0
        assert fit.max_separation_within_threshold(threshold=2.0) == True
        assert fit.max_separation_within_threshold(threshold=0.5) == False

        positions = al.GridIrregularGrouped(
            grid=[[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)], [(0.0, 0.0)]]
        )
        noise_map = al.ValuesIrregularGrouped([[1.0, 1.0], [1.0]])

        tracer = MockTracerPositions(positions=positions)
        fit = al.FitPositionsSourcePlaneMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )

        assert fit.furthest_separations_of_source_plane_positions.in_grouped_list == [
            [3.0, 2.0, 3.0],
            [0.0],
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

        noise_map = al.ValuesIrregularGrouped([[1.0, 1.0]])

        positions = al.GridIrregularGrouped([[(1.0, 0.0), (-1.0, 0.0)]])
        fit = al.FitPositionsSourcePlaneMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )
        assert fit.max_separation_within_threshold(threshold=0.01)

        positions = al.GridIrregularGrouped([[(1.2, 0.0), (-1.0, 0.0)]])
        fit = al.FitPositionsSourcePlaneMaxSeparation(
            positions=positions, noise_map=noise_map, tracer=tracer
        )
        assert fit.max_separation_within_threshold(threshold=0.3)
        assert not fit.max_separation_within_threshold(threshold=0.15)


# class TestFitPositionsSourcePlane:
#     def test__likelihood__is_sum_of_separations_divided_by_noise(self):
#
#         positions = al.GridIrregularGrouped(
#             [
#                 [(0.0, 0.0), (0.0, 1.0), (0.0, 0.5)],
#                 [(0.0, 0.0), (0.0, 0.0), (3.0, 3.0)],
#                 [(0.0, 0.0), (1.0, 1.0), (3.0, 3.0)],
#             ]
#         )
#
#         noise_map = al.ValuesIrregularGrouped(
#             [
#                 [1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#             ]
#         )
#
#         tracer = MockTracerPositions(positions=positions)
#
#         fit = al.FitPositionsSourcePlaneMaxSeparation(
#             positions=positions, noise_map=noise_map, tracer=tracer,
#         )
#         assert fit.chi_squared_map[0] == 1.0
#         assert fit.chi_squared_map[1] == pytest.approx(18.0, 1e-4)
#         assert fit.chi_squared_map[2] == pytest.approx(18.0, 1e-4)
#         assert fit.figure_of_merit == pytest.approx(-0.5 * (1.0 + 18 + 18), 1e-4)
#
#         fit = al.FitPositionsSourcePlaneMaxSeparation(
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


class TestFitPositionsImagePlane:
    def test__two_sets_of_positions__residuals_likelihood_correct(self):

        tracer = MockTracerPositions(positions=None)

        positions = al.GridIrregularGrouped([[(0.0, 0.0), (3.0, 4.0)], [(3.0, 3.0)]])

        noise_map = al.ValuesIrregularGrouped([[0.5, 1.0], [1.0]])

        model_positions = al.GridIrregularGrouped(
            [[(3.0, 1.0), (2.0, 3.0)], [(3.0, 3.0)]]
        )

        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        fit = al.FitPositionsImagePlane(
            positions=positions,
            noise_map=noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit.noise_map.in_grouped_list == [[0.5, 1.0], [1.0]]
        assert fit.residual_map.in_grouped_list == [
            [np.sqrt(10.0), np.sqrt(2.0)],
            [0.0],
        ]
        assert fit.normalized_residual_map.in_grouped_list == [
            [np.sqrt(10.0) / 0.5, np.sqrt(2.0) / 1.0],
            [0.0],
        ]
        assert fit.chi_squared_map.in_grouped_list == [
            [(np.sqrt(10.0) / 0.5) ** 2, np.sqrt(2.0) ** 2.0],
            [0.0],
        ]
        assert fit.chi_squared == pytest.approx(42.0, 1.0e-4)
        assert fit.noise_normalization == pytest.approx(4.12733, 1.0e-4)
        assert fit.log_likelihood == pytest.approx(-23.06366, 1.0e-4)
