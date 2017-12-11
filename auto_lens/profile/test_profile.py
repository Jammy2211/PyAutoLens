from __future__ import division, print_function

import numpy as np
import pytest
import profile


class TestEllipticalProfile(object):
    class TestAnglesFromXAxis(object):
        def test__phi_is_zero__angles_one_and_zero(self):
            elliptical_profile = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=0.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

        def test__phi_is_forty_five__angles_follow_trig(self):
            elliptical_profile = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == pytest.approx(0.707, 1e-3)
            assert sin_phi == pytest.approx(0.707, 1e-3)

        def test__phi_is_sixty__angles_follow_trig(self):
            elliptical_profile = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=60.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == pytest.approx(0.5, 1e-3)
            assert sin_phi == pytest.approx(0.866, 1e-3)

        def test__phi_is_225__angles_follow_trig_continues_round_from_x(self):
            elliptical_profile = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=225.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == pytest.approx(-0.707, 1e-3)
            assert sin_phi == pytest.approx(-0.707, 1e-3)

    class TestCoordinatesToCentre(object):
        def test__profile_centre_zeros__no_shift(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_shift__x_shifts(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.5, 0.0), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_y_shift__y_shifts(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.0, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_x_and_y_shift__x_and_y_both_shift(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.5, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_and_coordinates__correct_shifts(self):
            elliptical_profile = profile.EllipticalProfile(centre=(1.0, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.2, 0.4))

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_mass_profile__r_is_zero(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(0, 0)) == 0.0

        def test__x_coordinates_is_one__r_is_one(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(1.0, 0.0)) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self):
            elliptical_profile = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(1.0, 1.0)) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinates_first__r_includes_shift(self):
            elliptical_profile = profile.EllipticalProfile(centre=(-1.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(1.0, 0.0)) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self):
            elliptical_profile = profile.EllipticalProfile(centre=(2.0, 2.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(3.0, 3.0)) == pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAngleFromX(object):
        def test__angle_is_zero__angles_follow_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.7320))

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(-1.0, 1.0))

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_flips_back_to_45(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(-1.0, -1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -135

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, -1.0))

            assert theta_from_x == -45.0

        def test__include_shift_from_lens_centre__angles_follow_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(1.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            assert theta_from_x == 45.0

    class TestCoordinatesAngleToProfile(object):
        def test__same_angle__no_rotation(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == 1.0
            assert sin_theta == 0.0

        def test__both_45___no_rotation(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(1.0, 1e-3)
            assert sin_theta == pytest.approx(0.0, 1e-3)

        def test__45_offset_same_angle__rotation_follows_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

        def test__negative_60_offset_same_angle__rotation_follows_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=60.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.5, 1e-3)
            assert sin_theta == pytest.approx(-0.866, 1e-3)

        def test__include_lens_offset__rotation_follows_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(1.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

    class TestCoordinatesBackToCartesian(object):
        def test___phi_zero__no_rotation(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_elliptical = profile.TransformedCoordinates((1.0, 1.0))

            x, y = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert x == 1.0
            assert y == 1.0

        def test___phi_ninety__correct_calc(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            coordinates_elliptical = profile.TransformedCoordinates((1.0, 1.0))

            x, y = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert x == pytest.approx(-1.0, 1e-3)
            assert y == 1.0

        def test___phi_forty_five__correct_calc(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            coordinates_elliptical = profile.TransformedCoordinates((1.0, 1.0))

            x, y = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert x == pytest.approx(0.0, 1e-3)
            assert y == pytest.approx(2 ** 0.5, 1e-3)

    class TestRotateToElliptical(object):
        def test__phi_is_zero__returns_same_coordinates(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates = (1.0, 1.0)

            x, y = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert x == pytest.approx(1.0, 1e-3)
            assert y == pytest.approx(1.0, 1e-3)

        def test__phi_is_ninety__correct_rotation(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

        def test__phi_is_one_eighty__correct_rotation(self):
            # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=180.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert coordinates[0] == pytest.approx(-1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

        def test__phi_is_two_seventy__correct_rotation(self):
            # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=270.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert coordinates[0] == pytest.approx(-1.0, 1e-3)
            assert coordinates[1] == pytest.approx(1.0, 1e-3)

        def test__phi_is_three_sixty__correct_rotation(self):
            # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=360.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(1.0, 1e-3)

        def test__phi_is_three_one_five__correct_rotation(self):
            # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert coordinates[0] == pytest.approx(0.0, 1e-3)
            assert coordinates[1] == pytest.approx(2 ** 0.5, 1e-3)

        def test__shift_x_and_y_first__correct_rotation(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=90.0, centre=(1.0, 1.0))

            # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            coordinates = (2.0, 2.0)

            coordinates = elliptical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

    class TestCoordinateMovements(object):
        def test__moving_lens_and_coordinates__same_answer(self):
            elliptical_profile1 = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = elliptical_profile1.coordinates_rotate_to_elliptical(coordinates1)

            elliptical_profile2 = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = elliptical_profile2.coordinates_rotate_to_elliptical(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__moving_lens_and_coordinates_with_phi__same_answer(self):
            elliptical_profile1 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = elliptical_profile1.coordinates_rotate_to_elliptical(coordinates1)

            elliptical_profile2 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = elliptical_profile2.coordinates_rotate_to_elliptical(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__coordinates_both_on_centre___same_answer(self):
            elliptical_profile1 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(1, 1))
            coordinates1 = (1.0, 1.0)
            coordinates1 = elliptical_profile1.coordinates_rotate_to_elliptical(coordinates1)

            elliptical_profile2 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            coordinates2 = (-1.0, -1.0)
            coordinates2 = elliptical_profile2.coordinates_rotate_to_elliptical(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

    class TestRotateCoordinatesThenBackToCartesian(object):
        def test_are_consistent(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            coordinates_original = (5.2221, 2.6565)

            coordinates_elliptical = elliptical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

        def test_2__are_consistent(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=160.232)

            coordinates_original = (3.2, -76.6)

            coordinates_elliptical = elliptical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test_3__are_consistent(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=174.342)

            coordinates_original = (-42.2, -93.6)

            coordinates_elliptical = elliptical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test__include_offset_profile_centre__original_coordinates_shift_but_returned_coordinates_do_not_shift_back(
                self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=174.342, centre=(1.0, 1.0))

            coordinates_original = (-42.2, -93.6)

            coordinates_elliptical = elliptical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = elliptical_profile.coordinates_back_to_cartesian(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0] - 1.0, 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1] - 1.0, 1e-2)


class TestSphericalProfile(object):
    class TestAnglesFromXAxis(object):
        def test__phi_is_zero__angles_one_and_zero(self):
            spherical_profile = profile.SphericalProfile(centre=(1, 1))

            cos_phi, sin_phi = spherical_profile.angles_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

    class TestCoordinatesToCentre(object):
        def test__profile_centre_zeros__no_shift(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_shift__x_shifts(self):
            spherical_profile = profile.SphericalProfile(centre=(0.5, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_y_shift__y_shifts(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_x_and_y_shift__x_and_y_both_shift(self):
            spherical_profile = profile.SphericalProfile(centre=(0.5, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_and_coordinates__correct_shifts(self):
            spherical_profile = profile.SphericalProfile(centre=(1.0, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.2, 0.4))

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_mass_profile__r_is_zero(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0, 0))

            assert spherical_profile.coordinates_to_radius(coordinates_shift) == 0.0

        def test__x_coordinates_is_one__r_is_one(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 0))

            assert spherical_profile.coordinates_to_radius(coordinates_shift) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 1.0))

            assert spherical_profile.coordinates_to_radius(coordinates_shift) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinates_first__r_includes_shift(self):
            spherical_profile = profile.SphericalProfile(centre=(-1.0, 0.0))

            assert spherical_profile.coordinates_to_radius(coordinates=(1.0, 0.0)) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self):
            spherical_profile = profile.SphericalProfile(centre=(2.0, 2.0))

            assert spherical_profile.coordinates_to_radius(coordinates=(3.0, 3.0)) == pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAnglesFromX(object):
        def test__angle_is_zero__angles_follow_trig(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 0.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 1.7320))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(-1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_flips_back_to_45(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(-1.0, -1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -135

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, -1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -45.0

        def test__include_shift_from_lens_centre__angles_follow_trig(self):
            spherical_profile = profile.SphericalProfile(centre=(1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            assert theta_from_x == 45.0

    class TestAngleToProfile(object):
        def test__same_angle__no_rotation(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            cos_theta, sin_theta = spherical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == 1.0
            assert sin_theta == 0.0

        def test_coordinate_at_45__angle_follows_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0.0, 0.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

        def test__include_lens_offset__rotation_follows_trig(self):
            elliptical_profile = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(1.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

    class TestCoordinatesMovement(object):
        def test_phi_zero__no_rotation(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_spherical = profile.TransformedCoordinates((1.0, 1.0))

            x, y = spherical_profile.coordinates_back_to_cartesian(coordinates_spherical)

            assert x == 1.0
            assert y == 1.0

        def test__moving_lens_and_coordinates__same_answer(self):
            spherical_profile1 = profile.SphericalProfile(centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = spherical_profile1.coordinates_rotate_to_elliptical(coordinates1)

            spherical_profile2 = profile.SphericalProfile(centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = spherical_profile2.coordinates_rotate_to_elliptical(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__moving_lens_and_coordinates_with_phi__same_answer(self):
            spherical_profile1 = profile.SphericalProfile(centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = spherical_profile1.coordinates_rotate_to_elliptical(coordinates1)

            spherical_profile2 = profile.SphericalProfile(centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = spherical_profile2.coordinates_rotate_to_elliptical(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__coordinates_both_on_centre___same_answer(self):
            spherical_profile1 = profile.SphericalProfile(centre=(1, 1))
            coordinates1 = (1.0, 1.0)
            coordinates1 = spherical_profile1.coordinates_rotate_to_elliptical(coordinates1)

            spherical_profile2 = profile.SphericalProfile(centre=(-1, -1))
            coordinates2 = (-1.0, -1.0)
            coordinates2 = spherical_profile2.coordinates_rotate_to_elliptical(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

    class TestRotateCoordinatesAndThenBackToCartesian(object):
        def test__phi_is_zero__returns_same_coordinates(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates = (1.0, 1.0)

            x, y = spherical_profile.coordinates_rotate_to_elliptical(coordinates)

            assert x == pytest.approx(1.0, 1e-3)
            assert y == pytest.approx(1.0, 1e-3)

        def test__are_consistent(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = (5.2221, 2.6565)

            coordinates_spherical = spherical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = spherical_profile.coordinates_back_to_cartesian(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

        def test__2__are_consistent(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = (3.2, -76.6)

            coordinates_spherical = spherical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = spherical_profile.coordinates_back_to_cartesian(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test__3__are_consistent(self):
            spherical_profile = profile.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = (-42.2, -93.6)

            coordinates_spherical = spherical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = spherical_profile.coordinates_back_to_cartesian(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test_include_offset_profile_centre__original_coordinates_shift_but_returned_coordinates_do_not_shift_back(
                self):
            spherical_profile = profile.SphericalProfile(centre=(1.0, 1.0))

            coordinates_original = (-42.2, -93.6)

            coordinates_spherical = spherical_profile.coordinates_rotate_to_elliptical(coordinates_original)

            coordinates = spherical_profile.coordinates_back_to_cartesian(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0] - 1.0, 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1] - 1.0, 1e-2)
