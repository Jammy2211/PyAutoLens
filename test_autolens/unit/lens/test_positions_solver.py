import autolens as al
from autolens.lens import positions_solver as pos

import numpy as np

import pytest


class TestPositionSolver:
    def test__positions_for_simple_mass_profiles(self):

        grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05)

        sis = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.01)

        positions = solver.solve(lensing_obj=sis, source_plane_coordinate=(0.0, 0.11))

    def test__positions_for_tracer(self):

        grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

        g0 = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.001, 0.001),
                einstein_radius=1.0,
                elliptical_comps=(0.0, 0.111111),
            ),
        )

        g1 = al.Galaxy(redshift=1.0)

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        solver = pos.PositionsSolver(grid=grid, pixel_scale_precision=0.01)

        coordinates = solver.solve(
            lensing_obj=tracer, source_plane_coordinate=(0.0, 0.0)
        )

        assert coordinates[0] == pytest.approx((1.028125, -0.003125), 1.0e-4)
        assert coordinates[1] == pytest.approx((0.009375, -0.95312), 1.0e-4)
        assert coordinates[2] == pytest.approx((0.009375, 0.95312), 1.0e-4)
        assert coordinates[3] == pytest.approx((-1.028125, -0.003125), 1.0e-4)

    def test__multiple_image_coordinate_of_light_profile_centres_of_source_plane(self):

        grid = al.Grid.uniform(shape_2d=(50, 50), pixel_scales=0.05, sub_size=4)

        g0 = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0, elliptical_comps=(0.0, 0.055555)
            ),
        )

        g1 = al.Galaxy(
            redshift=1.0,
            light=al.lp.SphericalGaussian(centre=(0.0, 0.0)),
            light0=al.lp.SphericalGaussian(centre=(0.1, 0.1)),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        coordinates_manual_0 = tracer.image_plane_multiple_image_positions(
            grid=grid, source_plane_coordinate=(0.0, 0.0)
        )

        coordinates_manual_1 = tracer.image_plane_multiple_image_positions(
            grid=grid, source_plane_coordinate=(0.1, 0.1)
        )

        positions_of_galaxies = tracer.image_plane_multiple_image_positions_of_galaxies(
            grid=grid
        )

        solver = pos.PositionsSolver(grid=grid, pixel_scale_precision=0.01)

        coordinates = solver.solve(
            lensing_obj=tracer, source_plane_coordinate=(0.0, 0.0)
        )

        assert coordinates_manual_0 == positions_of_galaxies[0]
        assert coordinates_manual_1 == positions_of_galaxies[1]


class TestGridNeighbors1d:
    def test__creates_numpy_array_with_correct_neighbors(self):

        grid_1d = np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )

        grid_neighbors_1d, grid_has_neighbors = pos.grid_neighbors_1d_from(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_neighbors_1d
            == np.array(
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 2, 3, 5, 6, 7, 8],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                ]
            )
        ).all()

        grid_1d = np.array(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [2.0, 3.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ]
        )

        grid_neighbors_1d, grid_has_neighbors = pos.grid_neighbors_1d_from(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_neighbors_1d
            == np.array(
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 2, 3, 5, 6, 7, 8],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                ]
            )
        ).all()

        assert (
            grid_has_neighbors
            == np.array([False, False, False, False, True, False, False, False, False])
        ).all()

        grid_1d = np.array(
            [
                [1.5, -1.5],
                [1.5, -0.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, -1.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [0.5, 1.5],
                [-0.5, -1.5],
                [-0.5, -0.5],
                [-0.5, 0.5],
                [-0.5, 1.5],
                [-1.5, -1.5],
                [-1.5, -0.5],
                [-1.5, 0.5],
                [-1.5, 1.5],
            ]
        )

        grid_neighbors_1d, grid_has_neighbors = pos.grid_neighbors_1d_from(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_neighbors_1d
            == np.array(
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 2, 4, 6, 8, 9, 10],
                    [1, 2, 3, 5, 7, 9, 10, 11],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 5, 6, 8, 10, 12, 13, 14],
                    [5, 6, 7, 9, 11, 13, 14, 15],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                ]
            )
        ).all()

        assert (
            grid_has_neighbors
            == np.array(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            )
        ).all()

        grid_1d = np.array(
            [
                [1.0, -4.0],
                [1.0, -3.0],
                [1.0, -2.0],
                [0.0, -4.0],
                [0.0, -3.0],
                [0.0, -2.0],
                [-1.0, -4.0],
                [-1.0, -3.0],
                [-1.0, -2.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [0.0, 4.0],
                [-1.0, 2.0],
                [-1.0, 3.0],
                [-1.0, 4.0],
            ]
        )

        grid_neighbors_1d, grid_has_neighbors = pos.grid_neighbors_1d_from(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_neighbors_1d
            == np.array(
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 2, 6, 8, 12, 13, 14],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 4, 5, 9, 11, 15, 16, 17],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                ]
            )
        ).all()

        assert (
            grid_has_neighbors
            == np.array(
                [
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ]
            )
        ).all()


class TestTroughCoordinates:
    def test__simple_arrays(self):

        distance_1d = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        grid_1d = np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )

        grid_neighbors_1d, grid_has_neighbors = pos.grid_neighbors_1d_from(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        trough_coordinates = pos.grid_trough_from(
            distance_1d=distance_1d,
            grid_1d=grid_1d,
            neighbors=grid_neighbors_1d.astype("int"),
            has_neighbors=grid_has_neighbors,
        )

        assert (np.asarray(trough_coordinates) == np.array([[0.0, 0.0]])).all()

    # def test__simple_arrays_with_mask(self):
    #
    #     array = al.Array.manual_2d(
    #         array=[
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #             [9.0, 1.0, 9.0, 1.0, 9.0],
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #         ]
    #     )
    #
    #     mask = al.Mask.manual(
    #         mask=[
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #             [True, True, True, False, True],
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #         ]
    #     )
    #
    #     trough_coordinates = pos.grid_trough_from(array_2d=array.in_2d, mask=mask)
    #
    #     assert trough_coordinates == [[2, 3]]
    #
    #     array = al.Array.manual_2d(
    #         array=[
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #             [9.0, 9.0, 9.0, 1.0, 9.0],
    #             [9.0, 1.0, 9.0, 1.0, 9.0],
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #         ]
    #     )
    #
    #     mask = al.Mask.manual(
    #         mask=[
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #         ]
    #     )
    #
    #     trough_coordinates = pos.grid_trough_from(array_2d=array.in_2d, mask=mask)
    #
    #     assert trough_coordinates == []
    #
    #     array = al.Array.manual_2d(
    #         array=[
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #             [2.0, 8.0, 7.0, 6.0, 8.0],
    #             [4.0, 9.0, 4.0, 1.0, 8.0],
    #             [1.0, 0.5, 7.0, 0.1, 8.0],
    #             [9.0, 9.0, 9.0, 9.0, 9.0],
    #         ]
    #     )
    #
    #     mask = al.Mask.manual(
    #         mask=[
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #             [True, True, True, True, True],
    #             [True, True, True, False, True],
    #             [True, True, True, True, True],
    #         ]
    #     )
    #
    #     trough_coordinates = pos.grid_trough_from(array_2d=array.in_2d, mask=mask)
    #
    #     assert trough_coordinates == [[3, 3]]
