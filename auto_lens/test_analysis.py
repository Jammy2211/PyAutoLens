import pytest
import numpy as np
import analysis
import math


# TODO : Work out some more test cases, particularly for the border / move factors / relocate routines
# TODO : Need to add functionality for sub-coordinates.

class TestSourcePlane(object):
    class TestInit(object):
        def test__sets_correct_values(self):
            coordinates = [(1.0, 1.0), (0.0, 0.5)]

            source_plane = analysis.SourcePlane(coordinates)

            assert source_plane.coordinates == [(1.0, 1.0), (0.0, 0.5)]

        def test__four_coordinates__correct_source_plane(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))

            assert source_plane.coordinates == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

        def test__four_coordinates_and_offset_centre__doesnt_change_coordinate_values(self):

            # The centre is used by SourcePlaneGeomtry, but doesn't change the input coordinate values
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

            source_plane = analysis.SourcePlane(coordinates, centre=(0.5, 0.5))

            assert source_plane.coordinates == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

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


class TestSorucePlaneBorder(object):

    class TestSetupBorder(object):

        def test__four_coordinates_in_circle__correct_border(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

            border = analysis.SourcePlaneBorder(coordinates, 3, centre=(0.0, 0.0))

            assert border.coordinates == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

        def test__six_coordinates_two_masked__correct_border(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

            border = analysis.SourcePlaneBorder(coordinates, 3, centre=(0.0, 0.0))

            assert border.coordinates == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

        def test__test_other_thetas_radii(self):
            coordinates = [(2.0, 0.0),  (2.0, 2.0), (-1.0, -1.0), (0.0, -3.0)]

            border = analysis.SourcePlaneBorder(coordinates, 3, centre=(0.0, 0.0))

            assert border.coordinates == [(2.0, 0.0), (2.0, 2.0), (-1.0, -1.0), (0.0, -3.0)]
            assert border.radii == [2.0, 2.0 * math.sqrt(2), math.sqrt(2.0), 3.0]
            assert border.thetas == [0.0, 45.0, 225.0, 270.0]

        def test__source_plane_centre_offset__coordinates_same_r_and_theta_shifted(self):
            coordinates = [(2.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 0.0)]

            border = analysis.SourcePlaneBorder(coordinates, 3, centre=(1.0, 1.0))

            assert border.coordinates == [(2.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 0.0)]
            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

    class TestSetupBorderViaSourcePlaneAndMask(object):

        def test__four_coordinates_in_circle__correct_border(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
            border_mask = [True, True, True, True]

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))
            border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)

            assert border.coordinates == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

        def test__six_coordinates_two_masked__correct_border(self):
            coordinates = [(1.0, 0.0), (20., 20.), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0)]
            border_mask = [True, False, True, True, True, False]

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))
            border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)

            assert border.coordinates == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

        def test__test_other_thetas_radii(self):
            coordinates = [(2.0, 0.0), (20., 20.), (2.0, 2.0), (-1.0, -1.0), (0.0, -3.0), (1.0, 1.0)]
            border_mask = [True, False, True, True, True, False]

            source_plane = analysis.SourcePlane(coordinates, centre=(0.0, 0.0))
            border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)

            assert border.coordinates == [(2.0, 0.0), (2.0, 2.0), (-1.0, -1.0), (0.0, -3.0)]
            assert border.radii == [2.0, 2.0 * math.sqrt(2), math.sqrt(2.0), 3.0]
            assert border.thetas == [0.0, 45.0, 225.0, 270.0]

        def test__source_plane_centre_offset__coordinates_same_r_and_theta_shifted(self):
            coordinates = [(2.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 0.0)]
            border_mask = [True, True, True, True]

            source_plane = analysis.SourcePlane(coordinates, centre=(1.0, 1.0))
            border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)

            assert border.coordinates == [(2.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 0.0)]
            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

    class TestBorderPolynomial(object):
        def test__four_coordinates_in_circle__thetas_at_radius_are_each_coordinates_radius(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 0.0)]
            border_mask = [True, True, True, True, False]

            source_plane = analysis.SourcePlane(coordinates)
            border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)

            assert border.border_radius_at_theta(theta=0.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=90.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=180.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=270.0) == pytest.approx(1.0, 1e-3)

        def test__eight_coordinates_in_circle__thetas_at_each_coordinates_are_the_radius(self):
            coordinates = [(1.0, 0.0), (0.5 * math.sqrt(2), 0.5 * math.sqrt(2)), (0.0, 1.0),
                           (-0.5 * math.sqrt(2), 0.5 * math.sqrt(2)),
                           (-1.0, 0.0), (-0.5 * math.sqrt(2), -0.5 * math.sqrt(2)), (0.0, -1.0),
                           (0.5 * math.sqrt(2), -0.5 * math.sqrt(2))]

            border_mask = [True, True, True, True, True, True, True, True]

            source_plane = analysis.SourcePlane(coordinates)
            border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)

            assert border.border_radius_at_theta(theta=0.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=45.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=90.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=135.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=180.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=225.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=270.0) == pytest.approx(1.0, 1e-3)
            assert border.border_radius_at_theta(theta=315.0) == pytest.approx(1.0, 1e-3)

    class TestRelocateCoordinates(object):

        def test__outside_border_simple_cases__relocates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            source_border = analysis.SourcePlaneBorder(circle, 3, centre=(0.0, 0.0))

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(2.5, 0.37))
            assert source_border.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(25.3, -9.2))
            assert source_border.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(13.5, 0.0))
            assert source_border.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(-2.5, -0.37))
            assert source_border.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

        def test__outside_border_simple_cases_2__relocates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            source_border = analysis.SourcePlaneBorder(circle, 3, centre=(0.0, 0.0))

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(2.0, 0.0))
            assert relocated_coordinate == pytest.approx((1.0, 0.0), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(1.0, 1.0))
            assert relocated_coordinate == pytest.approx((0.5 * math.sqrt(2), 0.5 * math.sqrt(2)), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(0.0, 2.0))
            assert relocated_coordinate == pytest.approx((0.0, 1.0), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(-1.0, 1.0))
            assert relocated_coordinate == pytest.approx((-0.5 * math.sqrt(2), 0.5 * math.sqrt(2)), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(-2.0, 0.0))
            assert relocated_coordinate == pytest.approx((-1.0, 0.0), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(-1.0, -1.0))
            assert relocated_coordinate == pytest.approx((-0.5 * math.sqrt(2), -0.5 * math.sqrt(2)), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(0.0, -1.0))
            assert relocated_coordinate == pytest.approx((0.0, -1.0), 1e-3)

            relocated_coordinate = source_border.relocated_coordinate(coordinate=(1.0, -1.0))
            assert relocated_coordinate == pytest.approx((0.5 * math.sqrt(2), -0.5 * math.sqrt(2)), 1e-3)

        def test__outside_border_simple_cases_setup__via_source_plane_border_mask_routine__relocates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = circle + [(2.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0),
                                    (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)]

            border_mask = [True] * 16 + [False] * 8

            source_plane = analysis.SourcePlane(coordinates)
            source_plane.relocate_coordinates_outside_border_with_mask_and_polynomial_degree(border_mask, 3)

            source_plane.coordinates = map(lambda r: pytest.approx(r, 1e-3), source_plane.coordinates)

            assert source_plane.coordinates[:][0:16] == coordinates[:][0:16]
            assert source_plane.coordinates[:][16] == (1.0, 0.0)
            assert source_plane.coordinates[:][17] == (0.5 * math.sqrt(2), 0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][18] == (0.0, 1.0)
            assert source_plane.coordinates[:][19] == (-0.5 * math.sqrt(2), 0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][20] == (-1.0, 0.0)
            assert source_plane.coordinates[:][21] == (-0.5 * math.sqrt(2), -0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][22] == (0.0, -1.0)
            assert source_plane.coordinates[:][23] == (0.5 * math.sqrt(2), -0.5 * math.sqrt(2))

        def test__outside_border_same_as_above_but_setup_via_border_mask__relocates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = circle + [(2.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0),
                                    (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)]

            border_mask = [True] * 16 + [False] * 8

            source_plane = analysis.SourcePlane(coordinates)
            source_border = source_plane.border_with_mask_and_polynomial_degree(border_mask, 3)
            source_plane.relocate_coordinates_outside_border(source_border)

            source_plane.coordinates = map(lambda r: pytest.approx(r, 1e-3), source_plane.coordinates)

            assert source_plane.coordinates[:][0:16] == coordinates[:][0:16]
            assert source_plane.coordinates[:][16] == (1.0, 0.0)
            assert source_plane.coordinates[:][17] == (0.5 * math.sqrt(2), 0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][18] == (0.0, 1.0)
            assert source_plane.coordinates[:][19] == (-0.5 * math.sqrt(2), 0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][20] == (-1.0, 0.0)
            assert source_plane.coordinates[:][21] == (-0.5 * math.sqrt(2), -0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][22] == (0.0, -1.0)
            assert source_plane.coordinates[:][23] == (0.5 * math.sqrt(2), -0.5 * math.sqrt(2))

        def test__all_inside_border_simple_cases__no_relocations(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates_original = circle + [(0.2, 0.0), (0.1, 0.1), (0.0, 0.2), (-0.1, 0.1),
                                    (-0.2, 0.0), (-0.1, -0.1), (0.0, -0.2), (0.1, -0.1)]

            border_mask = [True] * 16 + [False] * 8

            source_plane = analysis.SourcePlane(coordinates_original)
            source_plane.relocate_coordinates_outside_border_with_mask_and_polynomial_degree(border_mask, 3)

            source_plane.coordinates = map(lambda r: pytest.approx(r, 1e-3), source_plane.coordinates)

            assert source_plane.coordinates == coordinates_original

        def test__inside_border_simple_cases_setup_via_border_mask__no_relocations(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = circle + [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5),
                                    (-0.5, 0.0), (-0.5, -0.5), (0.0, -0.5), (0.5, -0.5)]

            border_mask = [True] * 16 + [False] * 8

            source_plane = analysis.SourcePlane(coordinates)
            source_plane.relocate_coordinates_outside_border_with_mask_and_polynomial_degree(border_mask, 3)

            source_plane.coordinates = map(lambda r: pytest.approx(r, 1e-3), source_plane.coordinates)

            assert source_plane.coordinates[:][0:24] == coordinates[:][0:24]

        def test__inside_and_outside_border_simple_cases__changes_where_appropriate(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = circle + [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5),
                                    (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)]

            border_mask = [True] * 16 + [False] * 8

            source_plane = analysis.SourcePlane(coordinates)

            source_plane.relocate_coordinates_outside_border_with_mask_and_polynomial_degree(border_mask, 3)

            source_plane.coordinates = map(lambda r: pytest.approx(r, 1e-3), source_plane.coordinates)

            assert source_plane.coordinates[:][0:20] == coordinates[:][0:20]
            assert source_plane.coordinates[:][20] == (-1.0, 0.0)
            assert source_plane.coordinates[:][21] == (-0.5 * math.sqrt(2), -0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][22] == (0.0, -1.0)
            assert source_plane.coordinates[:][23] == (0.5 * math.sqrt(2), -0.5 * math.sqrt(2))

        def test__change_border_mask__works_as_above(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = [(-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)] + circle + \
                          [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5)]

            border_mask = [False] * 4 + [True] * 16 + [False] * 4

            source_plane = analysis.SourcePlane(coordinates)

            source_plane.relocate_coordinates_outside_border_with_mask_and_polynomial_degree(border_mask, 3)

            source_plane.coordinates = map(lambda r: pytest.approx(r, 1e-3), source_plane.coordinates)

            assert source_plane.coordinates[:][0] == (-1.0, 0.0)
            assert source_plane.coordinates[:][1] == (-0.5 * math.sqrt(2), -0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][2] == (0.0, -1.0)
            assert source_plane.coordinates[:][3] == (0.5 * math.sqrt(2), -0.5 * math.sqrt(2))
            assert source_plane.coordinates[:][4:24] == coordinates[:][4:24]

    class TestMoveFactors(object):
        def test__inside_border__move_factor_is_1(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

            source_border = analysis.SourcePlaneBorder(coordinates, 3, centre=(0.0, 0.0))

            assert source_border.move_factor(coordinate=(0.5, 0.0)) == 1.0
            assert source_border.move_factor(coordinate=(-0.5, 0.0)) == 1.0
            assert source_border.move_factor(coordinate=(0.25, 0.25)) == 1.0
            assert source_border.move_factor(coordinate=(0.0, 0.0)) == 1.0

        def test__outside_border_double_its_radius__move_factor_is_05(self):
            coordinates = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

            source_border = analysis.SourcePlaneBorder(coordinates, 3, centre=(0.0, 0.0))

            assert source_border.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert source_border.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert source_border.move_factor(coordinate=(-2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert source_border.move_factor(coordinate=(0.0, -2.0)) == pytest.approx(0.5, 1e-3)

        def test__outside_border_double_its_radius_and_offset__move_factor_is_05(self):
            coordinates = [(2.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 0.0)]

            source_border = analysis.SourcePlaneBorder(coordinates, 3, centre=(1.0, 1.0))

            assert source_border.move_factor(coordinate=(3.0, 1.0)) == pytest.approx(0.5, 1e-3)
            assert source_border.move_factor(coordinate=(1.0, 3.0)) == pytest.approx(0.5, 1e-3)
            assert source_border.move_factor(coordinate=(1.0, 3.0)) == pytest.approx(0.5, 1e-3)
            assert source_border.move_factor(coordinate=(3.0, 1.0)) == pytest.approx(0.5, 1e-3)


class TestRegularizationMatrix(object):

    # The regularization matrix, H, is calculated by defining a set of B matrices which describe how source-plane
    # pixels map to one another. For example, if we had a 3x3 square grid:

    #______
    #|0|1|2|
    #|3|4|5|
    #|6|7|8|
    #^^^^^^^ ( I cant find a horizontal 'high' line)

    # Lets say we want to regularize this grid so that each square pixel is regularized with a pixel to its right and
    # below it.

    # So, 0 is regularized with pixels 1 and 3, pixel 1 with 2 and 4, but pixel 2 with only pixel 5, etc. So,
    #
    # We make two 9 x 9 B matrices, which describe regularization in each direction. So for regularization to the
    # right of each pixel:

    # B_x = [-1,  1,  0,  0,  0,  0,  0,  0,  0] # [0->1] This, row 0, correspodns to pixel 0 (signified by the -1). The 1's in columns 1 is saying we want to regularize pixel 0 with pixel 1.
    #       [ 0, -1,  1,  0,  0,  0,  0,  0,  0] # [1->2] Row 1 for pixel 1 (again, the -1 tells us this), regularized with pixels 2.
    #       [ 0,  0, -1,  0,  0,  0,  0,  0,  0] # [] NOTE - pixel 2 is NOT regularized with pixel 3 (check the square grid)!
    #       [ 0,  0,  0, -1,  1,  0,  0,  0,  0] # [3->4]
    #       [ 0,  0,  0,  0, -1,  1,  0,  0,  0] # [4->5]
    #       [ 0,  0,  0,  0,  0, -1,  0,  0,  0] # [] NOTE - pixel 5 not regularized with pixel 6!
    #       [ 0,  0,  0,  0,  0,  0, -1,  1,  0] # [6->7]
    #       [ 0,  0,  0,  0,  0,  0,  0, -1,  1] # [7->8]
    #       [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # [] NOTE - Not regularized with anything

    # We now make another B matrix for the regularization beneath each pixel:

    # B_y = [-1,  0,  0,  1,  0,  0,  0,  0,  0] # [0->3] This, row 0, correspodns to pixel 0 (signified by the -1). The 1's in columns 3 is saying we want to regularize pixel 0 with pixel 3.
    #       [ 0, -1,  0,  0,  1,  0,  0,  0,  0] # [1->4] Row 1 for pixel 1 (again, the -1 tells us this), regularized with pixel 4
    #       [ 0,  0, -1,  0,  0,  1,  0,  0,  0] # [2->5]
    #       [ 0,  0,  0, -1,  0,  0,  1,  0,  0] # [3->6]
    #       [ 0,  0,  0,  0, -1,  0,  0,  1,  0] # [4->7]
    #       [ 0,  0,  0,  0,  0, -1,  0,  0,  1] # [5->8]
    #       [ 0,  0,  0,  0,  0,  0, -1,  0,  0] # [] No regularized performed in these last 3 rows / pixels
    #       [ 0,  0,  0,  0,  0,  0,  0, -1,  0] # []
    #       [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # []

    # So, we basically just make B matrices representing regularization in each direction. For each, we can then compute
    # their corresponding regularization matrix, H, as, H = B * B.T (matrix multiplication)

    # So, H_x = B_x.T, * B_x H_y = B_y.T * B_y
    # And our overall regularization matrix, H = H_x + H_y

    # For an adaptive Voronoi grid, we do the exact same thing, however we make a B matrix for every shared Voronoi vertex
    # of each soure-pixel cluster. This means that:

    # The number of B matrices we compute is equation to the the number of Voronoi vertices in the source-pixel with
    # the most Voronoi verticess.

    #### NOTE ####

    # You will notice, however, that the routine make_via_pixel_pairs doesn't use these B matrices above to compute H. This is
    # because for a Voronoi grid the B matrices have a very specific form, which means you can build H directly using just the
    # pixel_pairs between each soruce pixel Voronoi vertex, which is given by Python's Voronoi routine :).

    # Basically, this form arises between in a Voronoi grid, every pixel which it regularizes with is regularized back
    # by that pixel. I.e. every Voronoi vertex's neighbour is a neighbour of itself. This means our B matrices alwways look
    # something like:

    # B_1 = [-1,  0,  1,  0,  0] # [0->2] Note the symmetry here, pixel 0 -> 2 and pixel 2 -> 0.
    #       [ 0, -1,  0,  1,  0] # [1->3] Same for 1 -> 3 and 3 -> 1.
    #       [ 1,  0, -1,  0,  0] # [2->0]
    #       [ 0,  1,  0, -1,  0] # [3->1]
    #       [ 0,  0,  0,  1, -1] # [4->3] Problem - we dont have a corresponding entry of [3->4]


    # B_2 = [ 0,  0,  0,  0,  0] # Note how we now just have zeros wherever there is no more neighbouring pixels
    #       [ 0,  0,  0,  0,  0]
    #       [ 0,  0,  0,  0,  0]
    #       [ 0,  0,  0,  -1, 1] # [3->4] - The entry ends up in our next matrix.
    #       [ 0,  0,  0,  0, 0]

    # Thus, we can bypass matrix multiplication by exploiting this symmetry (and maximize  our use of its sparseness)
    # which is what theh routine I've wirrten does. We don't even use a B matrix when making H!

    # We should get numba of this asap!

    # TODO: All test cases assume one, constant, regularization coefficient (i.e. all regularization_weights = 1.0).
    # TODO : Need to add test cases for different regularization_weights

    def test__one_B_matrix_size_3x3_makes_correct_regularization_matrix(self):

        # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

        # This means our B matrix is:

        # [-1, 1, 0]
        # [0, -1, 1]
        # [0, 0, -1]

        # Regularization Matrix, H = B * B.T.

        test_b_matrix = np.matrix( ((-1, 1, 0),
                                    (1, -1, 0),
                                    (0, 0, 0)) )

        test_regularization_matrix = test_b_matrix.T * test_b_matrix

        no_verticies = np.array([1, 1, 0])
        pixel_pairs = np.array([[0,1]])
        regularization_weights = np.ones((3))

        regularization_matrix = analysis.RegularizationMatrix(3, regularization_weights, no_verticies, pixel_pairs)

        assert (regularization_matrix == test_regularization_matrix).all()

    def test__one_B_matrix_size_4x4__makes_correct_regularization_matrix(self):

        test_b_matrix = np.matrix( ((-1, 0, 1, 0),
                                    (0, -1, 0, 1),
                                    (1, 0, -1, 0),
                                    (0, 1, 0, -1)) )

        test_regularization_matrix = test_b_matrix.T * test_b_matrix

        no_verticies = np.array([1, 1, 1, 1])
        pixel_pairs = np.array([[0,2],[1,3]])
        regularization_weights = np.ones((4))

        regularization_matrix = analysis.RegularizationMatrix(4, regularization_weights, no_verticies, pixel_pairs)

        assert (regularization_matrix == test_regularization_matrix).all()

    def test__two_B_matrices_size_4x4__makes_correct_regularization_matrix(self):


        test_b_matrix_1 = np.matrix(((-1, 1, 0, 0),
                                     (0, -1, 1, 0),
                                     (0, 0, -1, 1),
                                     (1, 0, 0, -1)))

        test_regularization_matrix_1 = test_b_matrix_1.T * test_b_matrix_1

        test_b_matrix_2 = np.matrix(((-1, 0, 0, 1),
                                     (1, -1, 0, 0),
                                     (0, 1, -1, 0),
                                     (0, 0, 1, -1)))

        test_regularization_matrix_2 = test_b_matrix_2.T * test_b_matrix_2

        test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2

        no_verticies = np.array([2, 2, 2, 2])
        pixel_pairs = np.array([[0, 1], [1, 2], [2,3], [3, 0]])
        regularization_weights = np.ones((4))

        regularization_matrix = analysis.RegularizationMatrix(4, regularization_weights, no_verticies, pixel_pairs)

        assert (regularization_matrix == test_regularization_matrix).all()

    def test__two_B_matrices_size_4x4__makes_correct_regularization_matrix2(self):

        test_b_matrix_1 = np.matrix(((-1, 0, 1, 0),
                                     (0, -1, 1, 0),
                                     (1, 0, -1, 0),
                                     (1, 0, 0, -1)) )

        test_regularization_matrix_1 = test_b_matrix_1.T * test_b_matrix_1

        test_b_matrix_2 = np.matrix(((-1, 0, 0, 1),
                                     (0, 0, 0, 0),
                                     (0, 1, -1, 0),
                                     (0, 0, 0, 0)) )

        test_regularization_matrix_2 = test_b_matrix_2.T * test_b_matrix_2

        test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2

        no_verticies = np.array([2, 1, 2, 1])
        pixel_pairs = np.array([[0,2], [1,2], [0,3]])
        regularization_weights = np.ones((4))

        regularization_matrix = analysis.RegularizationMatrix(4, regularization_weights, no_verticies, pixel_pairs)

        assert (regularization_matrix == test_regularization_matrix).all()

    def test__two_pairs_two_B_matrices_size_3x3__makes_correct_regularization_matrix(self):

        # Here, we define the pixel_pairs first here and make the B matrices based on them.

        # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we don't
        # have to have as many B matrices as we do the source pixel with the most Voronoi vertices. We can combine the
        # rows of each B matrix wherever we like ;0.

        pixel_pairs = np.array([[0,1], [0,2]])
        no_verticies = np.array([2, 1, 1])

        test_b_matrix_1 = np.matrix(((-1, 1, 0), # Pair 1
                                     (-1, 0, 1), # Pair 2
                                     (1, -1, 0 )) ) # Pair 1 flip

        test_regularization_matrix_1 = test_b_matrix_1.T * test_b_matrix_1

        test_b_matrix_2 = np.matrix(((1, 0, -1), # Pair 2 flip
                                     (0, 0, 0),
                                     (0, 0, 0 )) )

        test_regularization_matrix_2 = test_b_matrix_2.T * test_b_matrix_2

        test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2

        regularization_weights = np.ones((3))

        regularization_matrix = analysis.RegularizationMatrix(3, regularization_weights, no_verticies, pixel_pairs)

        assert (regularization_matrix == test_regularization_matrix).all()

    def test__eight_pairs_four_B_matrices_size_6x6__makes_correct_regularization_matrix(self):

        # Again, lets exploit the freedom we have when setting up our B matrices to make matching it to pairs a lot less
        # Stressful.

        pixel_pairs = np.array([[0,2], [1,2], [0,3], [4,5], [1,5], [0,4], [2,3], [2,5]])

        no_verticies = np.array([3, 2, 4, 2, 2, 3])

        test_b_matrix_1 = np.matrix(((-1, 0, 1, 0, 0, 0), # Pair 1
                                     (0, -1, 1, 0, 0, 0), # Pair 2
                                     (-1, 0, 0, 1, 0, 0), # Pair 3
                                     (0, 0, 0, 0, -1, 1), # Pair 4
                                     (0, -1, 0, 0, 0, 1), # Pair 5
                                     (-1, 0, 0, 0, 1, 0)) ) # Pair 6

        test_regularization_matrix_1 = test_b_matrix_1.T * test_b_matrix_1

        test_b_matrix_2 = np.matrix(((0, 0, -1, 1, 0, 0), # Pair 7
                                     (0, 0, -1, 0, 0, 1), # Pair 8
                                     (0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0)) )

        test_regularization_matrix_2 = test_b_matrix_2.T * test_b_matrix_2

        test_b_matrix_3 = np.matrix(((1, 0, -1, 0, 0, 0), # Pair 1 flip
                                     (0, 1, -1, 0, 0, 0), # Pair 2 flip
                                     (1, 0, 0, -1, 0, 0), # Pair 3 flip
                                     (0, 0, 0, 0, 1, -1), # Pair 4 flip
                                     (0, 1, 0, 0, 0, -1), # Pair 5 flip
                                     (1, 0, 0, 0, -1, 0)) ) # Pair 6 flip

        test_regularization_matrix_3 = test_b_matrix_3.T * test_b_matrix_3

        test_b_matrix_4 = np.matrix(((0, 0, 1, -1, 0, 0), # Pair 7 flip
                                     (0, 0, 1, 0, 0, -1), # Pair 8 flip
                                     (0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0)) )

        test_regularization_matrix_4 = test_b_matrix_4.T * test_b_matrix_4

        test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2 + \
        test_regularization_matrix_3 + + test_regularization_matrix_4

        regularization_weights = np.ones((6))

        regularization_matrix = analysis.RegularizationMatrix(6, regularization_weights, no_verticies, pixel_pairs)

        assert (regularization_matrix == test_regularization_matrix).all()


class TestKMeans:

    def test__simple_points__sets_up_two_clusters(self):

        sparse_coordinates = np.array([[0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                       [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

        kmeans = analysis.KMeans(sparse_coordinates, n_clusters=2)

        assert (kmeans.cluster_centers_ == np.array([[2.0, 2.0]])).any()
        assert (kmeans.cluster_centers_ == np.array([[1.0, 1.0]])).any()

        assert list(kmeans.labels_).count(0) == 3
        assert list(kmeans.labels_).count(1) == 3

    def test__simple_points__sets_up_three_clusters(self):

        sparse_coordinates = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],
                                       [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                       [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

        kmeans = analysis.KMeans(sparse_coordinates, n_clusters=3)

        assert (kmeans.cluster_centers_ == np.array([[2.0, 2.0]])).any()
        assert (kmeans.cluster_centers_ == np.array([[1.0, 1.0]])).any()
        assert (kmeans.cluster_centers_ == np.array([[-1.0, -1.0]])).any()

        assert list(kmeans.labels_).count(0) == 3
        assert list(kmeans.labels_).count(1) == 3
        assert list(kmeans.labels_).count(2) == 3

    def test__simple_points__sets_up_three_clusters_more_points_in_third_cluster(self):

        sparse_coordinates = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],

                                       [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                       [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],

                                       [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                       [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                       [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                       [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

        kmeans = analysis.KMeans(sparse_coordinates, n_clusters=3)

        assert (kmeans.cluster_centers_ == np.array([[2.0, 2.0]])).any()
        assert (kmeans.cluster_centers_ == np.array([[1.0, 1.0]])).any()
    #    assert (kmeans.cluster_centers_ == np.array([[-1.0, -1.0]])).any()

        assert list(kmeans.labels_).count(0) == 3 or 6 or 12
        assert list(kmeans.labels_).count(1) == 3 or 6 or 12
        assert list(kmeans.labels_).count(2) == 3 or 6 or 12


class TestVoronoi:

    def test__simple_points__sets_up_voronoi_vertices(self):

        points = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])

        voronoi = analysis.Voronoi(points)

        assert (voronoi.vertices[0] == np.array([0., 1.])).all()
        assert (voronoi.vertices[1] == np.array([-1., 0.])).all()
        assert (voronoi.vertices[2] == np.array([1., 0.])).all()
        assert (voronoi.vertices[3] == np.array([0., -1.])).all()

    # def test__simple_points__neighbouring_points_index_reteived_correctly(self):
    #
    #     points = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    #
    #     voronoi = analysis.Voronoi(points)
    #
    #     print(voronoi.ridge_points)
    #
    #     assert (voronoi.indexes_of_neighbouring_points(point_index=0) == np.array([1, 2, 3, 4])).any()
    #     assert (voronoi.indexes_of_neighbouring_points(point_index=1) == np.array([0, 2, 4])).any()
 #       assert (voronoi.indexes_of_neighbouring_points(point_index=2) == np.array([1, 3])).any()
 #       assert (voronoi.indexes_of_neighbouring_points(point_index=3) == np.array([2, 4])).any()
 #       assert (voronoi.indexes_of_neighbouring_points(point_index=4) == np.array([4])).any()

class TestMatchCoordinatesFromClusters:

    def test__match_coordinates_to_clusters_via_nearest_neighbour__case1__correct_pairs(self):

        clusters = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        coordinates = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

        coordinates_to_cluster_index = analysis.match_coordintes_to_clusters_via_nearest_neighbour(coordinates, clusters)

        assert coordinates_to_cluster_index[0] == 0
        assert coordinates_to_cluster_index[1] == 1
        assert coordinates_to_cluster_index[2] == 2
        assert coordinates_to_cluster_index[3] == 3

    def test__match_coordinates_to_clusters_via_nearest_neighbour___case2__correct_pairs(self):

        clusters = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])

        coordinates = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1],
                                [0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])

        coordinates_to_cluster_index = analysis.match_coordintes_to_clusters_via_nearest_neighbour(coordinates, clusters)

        assert coordinates_to_cluster_index[0] == 0
        assert coordinates_to_cluster_index[1] == 1
        assert coordinates_to_cluster_index[2] == 2
        assert coordinates_to_cluster_index[3] == 3
        assert coordinates_to_cluster_index[4] == 3
        assert coordinates_to_cluster_index[5] == 2
        assert coordinates_to_cluster_index[6] == 1
        assert coordinates_to_cluster_index[7] == 0

    def test__match_coordinates_to_clusters_via_nearest_neighbour___case3__correct_pairs(self):

        clusters = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])

        coordinates = np.array([[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49], [0.51, 0.51], [1.01, 1.01], [1.51, 1.51]])

        coordinates_to_cluster_index = analysis.match_coordintes_to_clusters_via_nearest_neighbour(coordinates, clusters)

        assert coordinates_to_cluster_index[0] == 4
        assert coordinates_to_cluster_index[1] == 4
        assert coordinates_to_cluster_index[2] == 4
        assert coordinates_to_cluster_index[3] == 0
        assert coordinates_to_cluster_index[4] == 0
        assert coordinates_to_cluster_index[5] == 5

    # def test__match_coordinates_to_clusters_via_sparse_pairs__case1__correct_pairs(self):
    #
    #     clusters = np.array([[1.0, 0.0], [-1.0, 0.0]])
    #     coordinates = np.array([[1.1, 0.0], [0.9, 0.0], [-0.9, 0.0], [-1.1, 0.0]])
    #
    #     coordinates_to_cluster_index_nearest_neighbour = analysis.match_coordintes_to_clusters_via_nearest_neighbour(coordinates, clusters)
    #
    #     sparse_coordinates = np.array([[1.0, 0.0], [-1.0, 0.0]])
    #     sparse_coordinates_to_cluster_index = np.array([[0], [1]])
    #
    #     coordinates_to_cluster_index_sparse_pairs = analysis.match_coordintes_to_clusters_via_sparse_pairs(
    #                                         coordinates, clusters, sparse_coordinates, sparse_coordinates_to_cluster_index)
    #
    #     assert ( coordinates_to_cluster_index_nearest_neighbour == coordinates_to_cluster_index_sparse_pairs ).all()