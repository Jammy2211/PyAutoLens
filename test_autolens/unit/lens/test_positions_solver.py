import autolens as al
from autolens.lens import positions_solver as pos

import numpy as np

import pytest


class TestAbstractPositionsSolver:
    def test__solver_with_remove_distance_from_mass_profile_centre__remove_pixels_from_initial_grid(
        self,
    ):

        grid = al.Grid.manual_1d(
            grid=[[0.0, -0.1], [0.0, 0.0], [0.0, 0.1]],
            shape_2d=(1, 3),
            pixel_scales=0.1,
        )

        sis = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        solver = pos.AbstractPositionsSolver(distance_from_mass_profile_centre=0.01)

        grid = solver.grid_with_coordinates_from_mass_profile_centre_removed(
            grid=grid, lensing_obj=sis
        )

        assert grid == pytest.approx(np.array([[0.0, -0.1], [0.0, 0.1]]), 1.0e-4)

    def test__solver_create_buffed_and_updated_grid_from_input_coordinate(self):

        solver = pos.AbstractPositionsSolver(use_upscaling=True, upscale_factor=1)

        grid = solver.grid_buffed_and_upscaled_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=1, upscale_factor=1
        )

        grid_util = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=1, upscale_factor=1
        )

        assert (grid == grid_util).all()

        solver = pos.AbstractPositionsSolver(use_upscaling=False, upscale_factor=3)

        grid = solver.grid_buffed_and_upscaled_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=1, upscale_factor=1
        )

        grid_util = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=1, upscale_factor=1
        )

        assert (grid == grid_util).all()

        solver = pos.AbstractPositionsSolver(use_upscaling=True, upscale_factor=1)

        grid = solver.grid_buffed_and_upscaled_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=3, upscale_factor=1
        )

        grid_util = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=3, upscale_factor=1
        )

        assert (grid == grid_util).all()

        solver = pos.AbstractPositionsSolver(use_upscaling=True, upscale_factor=2)

        grid = solver.grid_buffed_and_upscaled_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=2, upscale_factor=2
        )

        grid_util = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 1.0), pixel_scales=(1.0, 1.0), buffer=2, upscale_factor=2
        )

        assert (grid == grid_util).all()


class TestPositionFinder:
    def test__positions_found_for_simple_mass_profiles(self):

        grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05)

        sis = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        solver = al.PositionsFinder(grid=grid, pixel_scale_precision=0.01)

        positions = solver.solve(lensing_obj=sis, source_plane_coordinate=(0.0, 0.11))

        assert positions[0] == pytest.approx(np.array([0.003125, -0.89062]), 1.0e-4)
        assert positions[1] == pytest.approx(np.array([-0.003125, -0.89062]), 1.0e-4)

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

        solver = pos.PositionsFinder(grid=grid, pixel_scale_precision=0.01)

        coordinates = solver.solve(
            lensing_obj=tracer, source_plane_coordinate=(0.0, 0.0)
        )

        assert coordinates.in_list[0][0] == pytest.approx((1.028125, -0.003125), 1.0e-4)
        assert coordinates.in_list[0][1] == pytest.approx((0.009375, -0.95312), 1.0e-4)
        assert coordinates.in_list[0][2] == pytest.approx((0.009375, 0.95312), 1.0e-4)
        assert coordinates.in_list[0][3] == pytest.approx(
            (-1.028125, -0.003125), 1.0e-4
        )

    def test__same_as_above_using_solver_for_tracer_method(self):

        grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

        g0 = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.001, 0.001),
                einstein_radius=1.0,
                elliptical_comps=(0.0, 0.111111),
            ),
        )

        g1 = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalLightProfile(centre=(0.0, 0.0))
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        solver = pos.PositionsFinder(grid=grid, pixel_scale_precision=0.01)

        coordinates = solver.solve_from_tracer(tracer=tracer)

        assert coordinates.in_list[0][0] == pytest.approx((1.028125, -0.003125), 1.0e-4)
        assert coordinates.in_list[0][1] == pytest.approx((0.009375, -0.95312), 1.0e-4)
        assert coordinates.in_list[0][2] == pytest.approx((0.009375, 0.95312), 1.0e-4)
        assert coordinates.in_list[0][3] == pytest.approx(
            (-1.028125, -0.003125), 1.0e-4
        )

    def test__solver_for_tracer_method__multiple_source_planes_or_galaxies(self):

        grid = al.Grid.uniform(shape_2d=(50, 50), pixel_scales=0.05, sub_size=4)

        g0 = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0, elliptical_comps=(0.0, 0.055555)
            ),
        )

        g1 = al.Galaxy(
            redshift=1.0,
            light_0=al.lp.EllipticalLightProfile(centre=(0.0, 0.0)),
            light_1=al.lp.EllipticalLightProfile(centre=(0.1, 0.1)),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        solver = pos.PositionsFinder(grid=grid, pixel_scale_precision=0.01)

        position_manual_0 = solver.solve(
            lensing_obj=tracer, source_plane_coordinate=(0.0, 0.0)
        )

        position_manual_1 = solver.solve(
            lensing_obj=tracer, source_plane_coordinate=(0.1, 0.1)
        )

        positions = solver.solve_from_tracer(tracer=tracer)

        assert position_manual_0.in_list[0] == positions.in_list[0]
        assert position_manual_1.in_list[0] == positions.in_list[1]

        g2 = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalLightProfile(centre=(0.0, 0.0))
        )

        g3 = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalLightProfile(centre=(0.1, 0.1))
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g2, g3])

        solver = pos.PositionsFinder(grid=grid, pixel_scale_precision=0.01)

        positions = solver.solve_from_tracer(tracer=tracer)

        assert position_manual_0.in_list[0] == positions.in_list[0]
        assert position_manual_1.in_list[0] == positions.in_list[1]


class TestGridRemoveDuplicates:
    def test__remove_duplicates_from_grid_within_tolerance(self):

        grid = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

        grid = pos.grid_remove_duplicates(grid=np.asarray(grid))

        assert grid == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

        grid = [(1.0, 1.0), (1.0, 1.0), (3.0, 3.0)]

        grid = pos.grid_remove_duplicates(grid=np.asarray(grid))

        assert grid == [(1.0, 1.0), (3.0, 3.0)]

        grid = [(1.0, 1.0), (1.0001, 1.0001), (3.0, 3.0)]

        grid = pos.grid_remove_duplicates(grid=np.asarray(grid))

        assert grid == [(1.0, 1.0), (1.0001, 1.0001), (3.0, 3.0)]

        grid = [
            (1.0, 1.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (2.0, 2.0),
            (4.0, 4.0),
            (5.0, 5.0),
            (3.0, 3.0),
        ]

        grid = pos.grid_remove_duplicates(grid=np.asarray(grid))

        assert grid == [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (5.0, 5.0), (3.0, 3.0)]


class TestGridBuffedAroundCoordinate:
    def test__single_point_grid_buffed_correctly__upscale_factor_1(self):
        grid_buffed_1d = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 0.0), pixel_scales=(1.0, 1.0), buffer=1
        )

        assert (
            grid_buffed_1d
            == np.array(
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
        ).all()

        grid_buffed_1d = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.5, 1.2), pixel_scales=(1.0, 1.0), buffer=1
        )

        assert grid_buffed_1d == pytest.approx(
            np.array(
                [
                    [1.5, 0.2],
                    [1.5, 1.2],
                    [1.5, 2.2],
                    [0.5, 0.2],
                    [0.5, 1.2],
                    [0.5, 2.2],
                    [-0.5, 0.2],
                    [-0.5, 1.2],
                    [-0.5, 2.2],
                ]
            ),
            1.0e-4,
        )

        grid_buffed_1d = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 0.0), pixel_scales=(1.0, 1.0), buffer=2
        )

        assert (
            grid_buffed_1d
            == np.array(
                [
                    [2.0, -2.0],
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [2.0, 2.0],
                    [1.0, -2.0],
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [0.0, -2.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [-1.0, -2.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                    [-1.0, 2.0],
                    [-2.0, -2.0],
                    [-2.0, -1.0],
                    [-2.0, 0.0],
                    [-2.0, 1.0],
                    [-2.0, 2.0],
                ]
            )
        ).all()

    def test__single_point_grid_buffed_correctly__upscale_factor_above_1(self):
        grid_buffed_1d = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 0.0), pixel_scales=(1.0, 1.0), buffer=1, upscale_factor=2
        )

        assert grid_buffed_1d == pytest.approx(
            np.array(
                [
                    [1.25, -1.25],
                    [1.25, -0.75],
                    [1.25, -0.25],
                    [1.25, 0.25],
                    [1.25, 0.75],
                    [1.25, 1.25],
                    [0.75, -1.25],
                    [0.75, -0.75],
                    [0.75, -0.25],
                    [0.75, 0.25],
                    [0.75, 0.75],
                    [0.75, 1.25],
                    [0.25, -1.25],
                    [0.25, -0.75],
                    [0.25, -0.25],
                    [0.25, 0.25],
                    [0.25, 0.75],
                    [0.25, 1.25],
                    [-0.25, -1.25],
                    [-0.25, -0.75],
                    [-0.25, -0.25],
                    [-0.25, 0.25],
                    [-0.25, 0.75],
                    [-0.25, 1.25],
                    [-0.75, -1.25],
                    [-0.75, -0.75],
                    [-0.75, -0.25],
                    [-0.75, 0.25],
                    [-0.75, 0.75],
                    [-0.75, 1.25],
                    [-1.25, -1.25],
                    [-1.25, -0.75],
                    [-1.25, -0.25],
                    [-1.25, 0.25],
                    [-1.25, 0.75],
                    [-1.25, 1.25],
                ]
            ),
            1.0e-4,
        )

        grid_buffed_1d = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 0.0), pixel_scales=(1.0, 1.0), buffer=2, upscale_factor=2
        )

        assert grid_buffed_1d == pytest.approx(
            np.array(
                [
                    [2.25, -2.25],
                    [2.25, -1.75],
                    [2.25, -1.25],
                    [2.25, -0.75],
                    [2.25, -0.25],
                    [2.25, 0.25],
                    [2.25, 0.75],
                    [2.25, 1.25],
                    [2.25, 1.75],
                    [2.25, 2.25],
                    [1.75, -2.25],
                    [1.75, -1.75],
                    [1.75, -1.25],
                    [1.75, -0.75],
                    [1.75, -0.25],
                    [1.75, 0.25],
                    [1.75, 0.75],
                    [1.75, 1.25],
                    [1.75, 1.75],
                    [1.75, 2.25],
                    [1.25, -2.25],
                    [1.25, -1.75],
                    [1.25, -1.25],
                    [1.25, -0.75],
                    [1.25, -0.25],
                    [1.25, 0.25],
                    [1.25, 0.75],
                    [1.25, 1.25],
                    [1.25, 1.75],
                    [1.25, 2.25],
                    [0.75, -2.25],
                    [0.75, -1.75],
                    [0.75, -1.25],
                    [0.75, -0.75],
                    [0.75, -0.25],
                    [0.75, 0.25],
                    [0.75, 0.75],
                    [0.75, 1.25],
                    [0.75, 1.75],
                    [0.75, 2.25],
                    [0.25, -2.25],
                    [0.25, -1.75],
                    [0.25, -1.25],
                    [0.25, -0.75],
                    [0.25, -0.25],
                    [0.25, 0.25],
                    [0.25, 0.75],
                    [0.25, 1.25],
                    [0.25, 1.75],
                    [0.25, 2.25],
                    [-0.25, -2.25],
                    [-0.25, -1.75],
                    [-0.25, -1.25],
                    [-0.25, -0.75],
                    [-0.25, -0.25],
                    [-0.25, 0.25],
                    [-0.25, 0.75],
                    [-0.25, 1.25],
                    [-0.25, 1.75],
                    [-0.25, 2.25],
                    [-0.75, -2.25],
                    [-0.75, -1.75],
                    [-0.75, -1.25],
                    [-0.75, -0.75],
                    [-0.75, -0.25],
                    [-0.75, 0.25],
                    [-0.75, 0.75],
                    [-0.75, 1.25],
                    [-0.75, 1.75],
                    [-0.75, 2.25],
                    [-1.25, -2.25],
                    [-1.25, -1.75],
                    [-1.25, -1.25],
                    [-1.25, -0.75],
                    [-1.25, -0.25],
                    [-1.25, 0.25],
                    [-1.25, 0.75],
                    [-1.25, 1.25],
                    [-1.25, 1.75],
                    [-1.25, 2.25],
                    [-1.75, -2.25],
                    [-1.75, -1.75],
                    [-1.75, -1.25],
                    [-1.75, -0.75],
                    [-1.75, -0.25],
                    [-1.75, 0.25],
                    [-1.75, 0.75],
                    [-1.75, 1.25],
                    [-1.75, 1.75],
                    [-1.75, 2.25],
                    [-2.25, -2.25],
                    [-2.25, -1.75],
                    [-2.25, -1.25],
                    [-2.25, -0.75],
                    [-2.25, -0.25],
                    [-2.25, 0.25],
                    [-2.25, 0.75],
                    [-2.25, 1.25],
                    [-2.25, 1.75],
                    [-2.25, 2.25],
                ]
            ),
            1.0e-4,
        )

        grid_buffed_1d = pos.grid_buffed_around_coordinate_from(
            coordinate=(0.0, 0.0), pixel_scales=(1.0, 1.0), buffer=1, upscale_factor=3
        )

        assert grid_buffed_1d == pytest.approx(
            np.array(
                [
                    [1.33, -1.33],
                    [1.33, -1.0],
                    [1.33, -0.66],
                    [1.33, -0.33],
                    [1.33, 0.0],
                    [1.33, 0.33],
                    [1.33, 0.66],
                    [1.33, 1.0],
                    [1.33, 1.33],
                    [1.0, -1.33],
                    [1.0, -1.0],
                    [1.0, -0.66],
                    [1.0, -0.33],
                    [1.0, 0.0],
                    [1.0, 0.33],
                    [1.0, 0.66],
                    [1.0, 1.0],
                    [1.0, 1.33],
                    [0.66, -1.33],
                    [0.66, -1.0],
                    [0.66, -0.66],
                    [0.66, -0.33],
                    [0.66, 0.0],
                    [0.66, 0.33],
                    [0.66, 0.66],
                    [0.66, 1.0],
                    [0.66, 1.33],
                    [0.33, -1.33],
                    [0.33, -1.0],
                    [0.33, -0.66],
                    [0.33, -0.33],
                    [0.33, 0.0],
                    [0.33, 0.33],
                    [0.33, 0.66],
                    [0.33, 1.0],
                    [0.33, 1.33],
                    [0.0, -1.33],
                    [0.0, -1.0],
                    [0.0, -0.66],
                    [0.0, -0.33],
                    [0.0, 0.0],
                    [0.0, 0.33],
                    [0.0, 0.66],
                    [0.0, 1.0],
                    [0.0, 1.33],
                    [-0.33, -1.33],
                    [-0.33, -1.0],
                    [-0.33, -0.66],
                    [-0.33, -0.33],
                    [-0.33, 0.0],
                    [-0.33, 0.33],
                    [-0.33, 0.66],
                    [-0.33, 1.0],
                    [-0.33, 1.33],
                    [-0.66, -1.33],
                    [-0.66, -1.0],
                    [-0.66, -0.66],
                    [-0.66, -0.33],
                    [-0.66, 0.0],
                    [-0.66, 0.33],
                    [-0.66, 0.66],
                    [-0.66, 1.0],
                    [-0.66, 1.33],
                    [-1.0, -1.33],
                    [-1.0, -1.0],
                    [-1.0, -0.66],
                    [-1.0, -0.33],
                    [-1.0, 0.0],
                    [-1.0, 0.33],
                    [-1.0, 0.66],
                    [-1.0, 1.0],
                    [-1.0, 1.33],
                    [-1.33, -1.33],
                    [-1.33, -1.0],
                    [-1.33, -0.66],
                    [-1.33, -0.33],
                    [-1.33, 0.0],
                    [-1.33, 0.33],
                    [-1.33, 0.66],
                    [-1.33, 1.0],
                    [-1.33, 1.33],
                ]
            ),
            1.0e-1,
        )


class TestGridNeighbors1d:
    def test__creates_numpy_array_with_correct_neighbors(self):

        neighbors_1d, has_neighbors = pos.grid_square_neighbors_1d_from(shape_1d=9)

        assert (
            neighbors_1d
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
            has_neighbors
            == np.array([False, False, False, False, True, False, False, False, False])
        ).all()

        neighbors_1d, has_neighbors = pos.grid_square_neighbors_1d_from(shape_1d=16)

        assert (
            neighbors_1d
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
            has_neighbors
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


class TestPairCoordinateToGrid:
    def test__coordinate_paired_to_closest_pixel_on_grid(self):

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

        index = pos.pair_coordinate_to_closest_pixel_on_grid(
            coordinate=(1.0, -1.0), grid_1d=grid_1d
        )

        assert index == 0

        index = pos.pair_coordinate_to_closest_pixel_on_grid(
            coordinate=(1.0, 1.0), grid_1d=grid_1d
        )

        assert index == 2

        index = pos.pair_coordinate_to_closest_pixel_on_grid(
            coordinate=(1.01, 1.10), grid_1d=grid_1d
        )

        assert index == 2

        index = pos.pair_coordinate_to_closest_pixel_on_grid(
            coordinate=(10.0, 10.0), grid_1d=grid_1d
        )

        assert index == 2


class TestGridPeaks:
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

        neighbors_1d, has_neighbors = pos.grid_square_neighbors_1d_from(shape_1d=9)

        peaks_coordinates = pos.grid_peaks_from(
            distance_1d=distance_1d,
            grid_1d=grid_1d,
            neighbors=neighbors_1d.astype("int"),
            has_neighbors=has_neighbors,
        )

        assert (np.asarray(peaks_coordinates) == np.array([[0.0, 0.0]])).all()

        distance_1d = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

        grid_1d = al.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0)

        neighbors_1d, has_neighbors = pos.grid_square_neighbors_1d_from(shape_1d=25)

        peaks_coordinates = pos.grid_peaks_from(
            distance_1d=distance_1d,
            grid_1d=grid_1d,
            neighbors=neighbors_1d.astype("int"),
            has_neighbors=has_neighbors,
        )

        assert (
            np.asarray(peaks_coordinates) == np.array([[0.0, -1.0], [0.0, 1.0]])
        ).all()


class TestWithinDistance:
    def test__grid_keeps_only_points_within_distance(self):

        grid_1d = np.array([[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]])

        distances_1d = np.array([2.0, 1.0, 3.0])

        new_grid = pos.grid_within_distance(
            distances_1d=distances_1d, grid_1d=grid_1d, within_distance=10.0
        )

        assert (new_grid == grid_1d).all()

        new_grid = pos.grid_within_distance(
            distances_1d=distances_1d, grid_1d=grid_1d, within_distance=2.5
        )

        assert (new_grid == np.array([[2.0, 2.0], [1.0, 1.0]])).all()

        new_grid = pos.grid_within_distance(
            distances_1d=distances_1d, grid_1d=grid_1d, within_distance=1.5
        )

        assert (new_grid == np.array([[1.0, 1.0]])).all()
