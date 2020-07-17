import autolens as al
from autolens.lens import positions_solver as pos

import numpy as np

import pytest


class TestPositionSolver:
    def test__grid_within_circle__finds_all_image_pixels_within_circle_in_source_plane__returns_as_grid(
        self
    ):

        grid = al.Grid.uniform(shape_2d=(5, 5), pixel_scales=0.1)

        sis = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.0)

        solver = al.PositionsSolver()

        grid_within_circle, distances_within_circle = solver.grid_within_circle_from(
            lensing_obj=sis, grid=grid, source_plane_coordinate=(0.0, 0.0), radius=0.05
        )

        # [True, True, True, False, True]
        # [True, True, True, True, False]
        # [True, True, False True, True]
        # [True, True, True, True, True]
        # [True, True, True, True, True]

        assert (grid_within_circle == np.array([[0.0, 0.0]])).all()
        assert (distances_within_circle == np.array([[0.0]])).all()

        grid_within_circle, distances_within_circle = solver.grid_within_circle_from(
            lensing_obj=sis, grid=grid, source_plane_coordinate=(0.1, 0.1), radius=0.11
        )

        # Mask witihin circle:

        # [True, True, True, False, True]
        # [True, True, False, False, False]
        # [True, True, True, False, True]
        # [True, True, True, True, True]
        # [True, True, True, True, True]

        print(distances_within_circle)

        assert grid_within_circle == pytest.approx(
            np.array([[0.2, 0.1], [0.1, 0.0], [0.1, 0.1], [0.1, 0.2], [0.0, 0.1]]),
            1.0e-4,
        )
        assert distances_within_circle == pytest.approx(
            np.array([0.01, 0.01, 0.0, 0.01, 0.01]), 1.0e-4
        )

    def test__positions_for_simple_mass_profiles(self):

        grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05)

        sis = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        solver = al.PositionsSolver()

        positions = solver.solve(
            lensing_obj=sis, grid=grid, source_plane_coordinate=(0.0, 0.11)
        )


class TestGridPixelCentres1dViaOverlay:
    def test__overlays_grid_using_pixel_scale(self):

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

        grid_pixel_centres_1d, y_shape, x_shape = pos.grid_pixel_centres_1d_via_grid_1d_overlap(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_pixel_centres_1d
            == np.array(
                [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
            )
        ).all()
        assert (y_shape, x_shape) == (5, 5)

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

        grid_pixel_centres_1d, y_shape, x_shape = pos.grid_pixel_centres_1d_via_grid_1d_overlap(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_pixel_centres_1d
            == np.array(
                [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
            )
        ).all()
        assert (y_shape, x_shape) == (5, 5)

        grid_1d = np.array([[3.0, 3.0], [3.0, 1.0], [0.0, 3.0], [2.0, 2.0]])

        grid_pixel_centres_1d, y_shape, x_shape = pos.grid_pixel_centres_1d_via_grid_1d_overlap(
            grid_1d=grid_1d, pixel_scale=1.0
        )

        assert (
            grid_pixel_centres_1d
            == np.array(
                [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
            )
        ).all()
        assert (y_shape, x_shape) == (5, 5)


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

        trough_coordinates = pos.trough_coordinates_from(
            distance_1d=distance_1d,
            grid_1d=grid_1d,
            neighbors=grid_neighbors_1d.astype("int"),
            has_neighbors=grid_has_neighbors,
        )

        assert (np.asarray(trough_coordinates) == np.array([[0.0, 0.0]])).all()

    def test__simple_arrays_with_mask(self):

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        trough_coordinates = pos.trough_coordinates_from(
            array_2d=array.in_2d, mask=mask
        )

        assert trough_coordinates == [[2, 3]]

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        trough_coordinates = pos.trough_coordinates_from(
            array_2d=array.in_2d, mask=mask
        )

        assert trough_coordinates == []

        array = al.Array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 0.1, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask = al.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        trough_coordinates = pos.trough_coordinates_from(
            array_2d=array.in_2d, mask=mask
        )

        assert trough_coordinates == [[3, 3]]


class TestPositionsAtCoordinate:
    def test__uniform_grid__locates_pixels_correctly(self):

        grid = al.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0)

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(0.3, 0.3)
        )

        assert pixels_at_coordinate == [(1, 2), (1, 3), (2, 2), (2, 3)]

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(-0.3, 0.3)
        )

        assert pixels_at_coordinate == [(2, 2), (2, 3), (3, 2), (3, 3)]

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(0.6, 0.6)
        )

        assert pixels_at_coordinate == [(1, 2), (1, 3), (2, 2), (2, 3)]

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(1.1, 1.1)
        )

        assert pixels_at_coordinate == [(1, 3)]

    def test__uniform_grid__mask_remove_points(self):

        grid = al.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0)

        mask = al.Mask.manual(
            mask=[
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(0.3, 0.3), mask=mask
        )

        assert pixels_at_coordinate == [(1, 3)]

    def test__non_uniform_grid__locates_multiple_pixels_correctly(self):

        grid = al.Grid.manual_2d(
            grid=[
                [
                    [3.0, 1.0],
                    [0.0, 0.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 3.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [3.0, 1.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
            pixel_scales=1.0,
        )

        pixels_at_coordinate = pos.positions_at_coordinate_from(
            grid_2d=grid.in_2d, coordinate=(2.0, 2.0)
        )

        assert pixels_at_coordinate == [(1, 1), (4, 5)]
