import pytest
import numpy as np
import analysis
import math

class TestSourcePlane(object):

    class TestInit(object):

        def test__sets_correct_values(self):
            coordinates = [(1.0, 1.0), (0.0, 0.5)]

            source_plane = analysis.SourcePlane(coordinates)

            assert source_plane.coordinates == [(1.0, 1.0), (0.0, 0.5)]

    class TestCoordinatesToCentre(object):

        def test__source_plane_centre_zeros_by_default__no_shift(self):

            coordinates = (0.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates)

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__source_plane_centre_x_shift__x_shifts(self):
            coordinates = (0.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.5, 0.0))

            coordinates_shift = source_plane.coordinates_to_centre(source_plane.coordinates)

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__source_plane_centre_y_shift__y_shifts(self):
            coordinates = (0.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.5))

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__source_plane_centre_x_and_y_shift__x_and_y_both_shift(self):
            coordinates = (0.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.5, 0.5))

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__source_plane_centre_and_coordinates__correct_shifts(self):
            coordinates = (0.2, 0.4)

            source_plane = analysis.SourcePlane(coordinates, centre=(1.0, 0.5))

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_source_plane_analysis__r_is_zero(self):
            coordinates = (0.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            assert source_plane.coordinates_to_radius(coordinates) == 0.0

        def test__x_coordinates_is_one__r_is_one(self):
            coordinates = (1.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            assert source_plane.coordinates_to_radius(coordinates) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self):
            coordinates = (1.0, 1.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            assert source_plane.coordinates_to_radius(coordinates) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinate_first__r_includes_shift(self):
            coordinates = (1.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(-1.0, 0.0))

            assert source_plane.coordinates_to_radius(coordinates) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self):
            coordinates = (3.0, 3.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(2.0, 2.0))

            assert source_plane.coordinates_to_radius(coordinates) == pytest.approx(math.sqrt(2.0), 1e-5)

    class TestCoordinatesAngleFromX(object):
        def test__angle_is_zero__angles_follow_trig(self):
            coordinates = (1.0, 0.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self):
            coordinates = (1.0, 1.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self):
            coordinates = (1.0, 1.7320)

            source_plane = analysis.SourcePlane(coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self):
            coordinates = (-1.0, 1.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_continues_above_180(self):
            coordinates = (-1.0, -1.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 225.0

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self):
            coordinates = (1.0, -1.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 315.0

        def test__include_source_plane_centre__angle_takes_into_accounts(self):

            coordinates = (2.0, 2.0)

            source_plane = analysis.SourcePlane(coordinates, centre=(1.0, 1.0))

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 45.0

    class TestComputeBoarderFunction(object):
        def test__compute_boarder_function__four_coordinates_in_circle__computes_correct_function(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.1, 0.0), (0.0, -1.1), (0.0, 0.0)]
            edge_mask = [True, True, True, True, False]

            source_plane = analysis.SourcePlane(coordinates)
            function = source_plane.compute_boarder_function(edge_mask)
