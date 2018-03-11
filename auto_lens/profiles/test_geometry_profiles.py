from __future__ import division, print_function

import numpy as np
import pytest
from auto_lens.profiles import geometry_profiles


class TestEllipticalProfile(object):
    class TestAnglesFromXAxis(object):
        def test__phi_is_zero__angles_one_and_zero(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=0.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

        def test__phi_is_forty_five__angles_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == pytest.approx(0.707, 1e-3)
            assert sin_phi == pytest.approx(0.707, 1e-3)

        def test__phi_is_sixty__angles_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=60.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == pytest.approx(0.5, 1e-3)
            assert sin_phi == pytest.approx(0.866, 1e-3)

        def test__phi_is_225__angles_follow_trig_continues_round_from_x(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=225.0)

            cos_phi, sin_phi = elliptical_profile.angles_from_x_axis()

            assert cos_phi == pytest.approx(-0.707, 1e-3)
            assert sin_phi == pytest.approx(-0.707, 1e-3)

    class TestCoordinatesToCentre(object):
        def test__profile_centre_zeros__no_shift(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_shift__x_shifts(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.5, 0.0), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_y_shift__y_shifts(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_x_and_y_shift__x_and_y_both_shift(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.5, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_and_coordinates__correct_shifts(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1.0, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(0.2, 0.4))

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_mass_profile__r_is_zero(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(0, 0)) == 0.0

        def test__x_coordinates_is_one__r_is_one(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(1.0, 0.0)) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(1.0, 1.0)) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinates_first__r_includes_shift(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(-1.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(1.0, 0.0)) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(2.0, 2.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=(3.0, 3.0)) == pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAngleFromX(object):
        def test__angle_is_zero__angles_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.7320))

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(-1.0, 1.0))

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_flips_back_to_45(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=(-1.0, -1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -135

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, -1.0))

            assert theta_from_x == -45.0

        def test__include_shift_from_lens_centre__angles_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(1.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            assert theta_from_x == 45.0

    class TestCoordinatesAngleToProfile(object):
        def test__same_angle__no_rotation(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == 1.0
            assert sin_theta == 0.0

        def test__both_45___no_rotation(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(1.0, 1e-3)
            assert sin_theta == pytest.approx(0.0, 1e-3)

        def test__45_offset_same_angle__rotation_follows_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

        def test__negative_60_offset_same_angle__rotation_follows_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=60.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.5, 1e-3)
            assert sin_theta == pytest.approx(-0.866, 1e-3)

        def test__include_lens_offset__rotation_follows_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(1.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

    class TestCoordinatesBackToCartesian(object):
        def test___phi_zero__no_rotation(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            x, y = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert x == 1.0
            assert y == 1.0

        def test___phi_ninety__correct_calc(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            x, y = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert x == pytest.approx(-1.0, 1e-3)
            assert y == 1.0

        def test___phi_forty_five__correct_calc(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            x, y = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert x == pytest.approx(0.0, 1e-3)
            assert y == pytest.approx(2 ** 0.5, 1e-3)

    class TestRotateToElliptical(object):
        def test__phi_is_zero__returns_same_coordinates(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates = (1.0, 1.0)

            x, y = elliptical_profile.transform_to_reference_frame(coordinates)

            assert x == pytest.approx(1.0, 1e-3)
            assert y == pytest.approx(1.0, 1e-3)

        def test__phi_is_ninety__correct_rotation(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            # NOTE - whilst the profiles and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

        def test__phi_is_one_eighty__correct_rotation(self):
            # NOTE - whilst the profiles and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=180.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(-1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

        def test__phi_is_two_seventy__correct_rotation(self):
            # NOTE - whilst the profiles and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=270.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(-1.0, 1e-3)
            assert coordinates[1] == pytest.approx(1.0, 1e-3)

        def test__phi_is_three_sixty__correct_rotation(self):
            # NOTE - whilst the profiles and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=360.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(1.0, 1e-3)

        def test__phi_is_three_one_five__correct_rotation(self):
            # NOTE - whilst the profiles and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            coordinates = (1.0, 1.0)

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(0.0, 1e-3)
            assert coordinates[1] == pytest.approx(2 ** 0.5, 1e-3)

        def test__shift_x_and_y_first__correct_rotation(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=90.0, centre=(1.0, 1.0))

            # NOTE - whilst the profiles and coordinates are defined counter-clockwise from x, the rotation is performed
            # clockwise

            coordinates = (2.0, 2.0)

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

    class TestCoordinateMovements(object):
        def test__moving_lens_and_coordinates__same_answer(self):
            elliptical_profile1 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = elliptical_profile1.transform_to_reference_frame(coordinates1)

            elliptical_profile2 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = elliptical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__moving_lens_and_coordinates_with_phi__same_answer(self):
            elliptical_profile1 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = elliptical_profile1.transform_to_reference_frame(coordinates1)

            elliptical_profile2 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = elliptical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__coordinates_both_on_centre___same_answer(self):
            elliptical_profile1 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(1, 1))
            coordinates1 = (1.0, 1.0)
            coordinates1 = elliptical_profile1.transform_to_reference_frame(coordinates1)

            elliptical_profile2 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            coordinates2 = (-1.0, -1.0)
            coordinates2 = elliptical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

    class TestRotateCoordinatesThenBackToCartesian(object):
        def test_are_consistent(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            coordinates_original = (5.2221, 2.6565)

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

        def test_2__are_consistent(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=160.232)

            coordinates_original = (3.2, -76.6)

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test_3__are_consistent(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=174.342)

            coordinates_original = (-42.2, -93.6)

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test__include_offset_profile_centre__original_coordinates_shift_but_returned_coordinates_do_not_shift_back(
                self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=174.342, centre=(1.0, 1.0))

            coordinates_original = (-42.2, -93.6)

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

    class TestTransformedCoordinatesToEccentricRadius(object):
        def test__no_ellipticity__correct_radius(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            elliptical_radius = elliptical_profile.coordinates_to_eccentric_radius(coordinates_elliptical)

            assert elliptical_radius == pytest.approx(2.0 ** 0.5, 1e-3)

        def test__include_axis_ratio__correct_radius(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=0.5, phi=0.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            elliptical_radius = elliptical_profile.coordinates_to_eccentric_radius(coordinates_elliptical)

            # sqrt(0.5) * sqrt(1.0**2 + (1.0 / 0.5)**2 ) = sqrt(0.5) * sqrt( 5 )

            assert elliptical_radius == pytest.approx(1.58113, 1e-3)


class TestSphericalProfile(object):
    class TestAnglesFromXAxis(object):
        def test__phi_is_zero__angles_one_and_zero(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1, 1))

            cos_phi, sin_phi = spherical_profile.angles_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

    class TestCoordinatesToCentre(object):
        def test__profile_centre_zeros__no_shift(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_shift__x_shifts(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.5, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_y_shift__y_shifts(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_x_and_y_shift__x_and_y_both_shift(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.5, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.0, 0.0))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_and_coordinates__correct_shifts(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1.0, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0.2, 0.4))

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_mass_profile__r_is_zero(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(0, 0))

            assert spherical_profile.coordinates_to_radius(coordinates_shift) == 0.0

        def test__x_coordinates_is_one__r_is_one(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 0))

            assert spherical_profile.coordinates_to_radius(coordinates_shift) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 1.0))

            assert spherical_profile.coordinates_to_radius(coordinates_shift) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinates_first__r_includes_shift(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(-1.0, 0.0))

            assert spherical_profile.coordinates_to_radius(coordinates=(1.0, 0.0)) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(2.0, 2.0))

            assert spherical_profile.coordinates_to_radius(coordinates=(3.0, 3.0)) == pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAnglesFromX(object):
        def test__angle_is_zero__angles_follow_trig(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 0.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, 1.7320))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(-1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_flips_back_to_45(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(-1.0, -1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -135

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=(1.0, -1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -45.0

        def test__include_shift_from_lens_centre__angles_follow_trig(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            assert theta_from_x == 45.0

    class TestAngleToProfile(object):
        def test__same_angle__no_rotation(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=(1.0, 0.0))

            cos_theta, sin_theta = spherical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == 1.0
            assert sin_theta == 0.0

        def test_coordinate_at_45__angle_follows_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0.0, 0.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(1.0, 1.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

        def test__include_lens_offset__rotation_follows_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(1.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=(2.0, 2.0))

            cos_theta, sin_theta = elliptical_profile.coordinates_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

    class TestCoordinatesMovement(object):
        def test_phi_zero__no_rotation(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_spherical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            x, y = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert x == 1.0
            assert y == 1.0

        def test__moving_lens_and_coordinates__same_answer(self):
            spherical_profile1 = geometry_profiles.SphericalProfile(centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = spherical_profile1.transform_to_reference_frame(coordinates1)

            spherical_profile2 = geometry_profiles.SphericalProfile(centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = spherical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__moving_lens_and_coordinates_with_phi__same_answer(self):
            spherical_profile1 = geometry_profiles.SphericalProfile(centre=(0, 0))
            coordinates1 = (1.0, 1.0)
            coordinates1 = spherical_profile1.transform_to_reference_frame(coordinates1)

            spherical_profile2 = geometry_profiles.SphericalProfile(centre=(-1, -1))
            coordinates2 = (0.0, 0.0)
            coordinates2 = spherical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__coordinates_both_on_centre___same_answer(self):
            spherical_profile1 = geometry_profiles.SphericalProfile(centre=(1, 1))
            coordinates1 = (1.0, 1.0)
            coordinates1 = spherical_profile1.transform_to_reference_frame(coordinates1)

            spherical_profile2 = geometry_profiles.SphericalProfile(centre=(-1, -1))
            coordinates2 = (-1.0, -1.0)
            coordinates2 = spherical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

    class TestRotateCoordinatesAndThenBackToCartesian(object):
        def test__phi_is_zero__returns_same_coordinates(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates = (1.0, 1.0)

            x, y = spherical_profile.transform_to_reference_frame(coordinates)

            assert x == pytest.approx(1.0, 1e-3)
            assert y == pytest.approx(1.0, 1e-3)

        def test__are_consistent(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = (5.2221, 2.6565)

            coordinates_spherical = spherical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

        def test__2__are_consistent(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = (3.2, -76.6)

            coordinates_spherical = spherical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test__3__are_consistent(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = (-42.2, -93.6)

            coordinates_spherical = spherical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test_include_offset_profile_centre__original_coordinates_shift_but_returned_coordinates_do_not_shift_back(
                self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1.0, 1.0))

            coordinates_original = (-42.2, -93.6)

            coordinates_spherical = spherical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)


class TestDecorators(object):
    def test_subgrid_2x2(self):
        @geometry_profiles.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((0.5, 0.5), pixel_scale=1.0, grid_size=2)
        assert coordinates == [(1. / 3., 1. / 3.), (1. / 3., 2. / 3.), (2. / 3., 1. / 3.), (2. / 3., 2. / 3.)]

    def test_subgrid_3x3(self):
        @geometry_profiles.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((0.5, 0.5), pixel_scale=1.0, grid_size=3)
        assert coordinates == [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75),
                               (0.50, 0.25), (0.50, 0.5), (0.50, 0.75),
                               (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)]

    def test_subgrid_3x3_triple_pixel_scale_and_coordinate(self):
        @geometry_profiles.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((1.5, 1.5), pixel_scale=3.0, grid_size=3)

        assert coordinates == [(0.75, 0.75), (0.75, 1.5), (0.75, 2.25),
                               (1.50, 0.75), (1.50, 1.5), (1.50, 2.25),
                               (2.25, 0.75), (2.25, 1.5), (2.25, 2.25)]

    def test_subgrid_4x4_new_coordinates(self):
        @geometry_profiles.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((-2.0, 3.0), pixel_scale=0.1, grid_size=4)

        coordinates = list(map(lambda coords: (pytest.approx(coords[0], 1e-2), pytest.approx(coords[1], 1e-2)), coordinates))

        assert coordinates == [(-2.03, 2.97), (-2.03, 2.99), (-2.03, 3.01), (-2.03, 3.03),
                               (-2.01, 2.97), (-2.01, 2.99), (-2.01, 3.01), (-2.01, 3.03),
                               (-1.99, 2.97), (-1.99, 2.99), (-1.99, 3.01), (-1.99, 3.03),
                               (-1.97, 2.97), (-1.97, 2.99), (-1.97, 3.01), (-1.97, 3.03)]

    def test_average(self):
        @geometry_profiles.avg
        def return_input(input_list):
            return input_list

        assert return_input([1, 2, 3]) == 2
        assert return_input([(1, 10), (2, 20), (3, 30)]) == (2, 20)

    def test_iterative_subgrid(self):
        # noinspection PyUnusedLocal
        @geometry_profiles.iterative_subgrid
        def one_over_grid(coordinates, pixel_scale, grid_size):
            return 1.0 / grid_size

        assert one_over_grid(None, None, 0.51) == pytest.approx(0.5)
        assert one_over_grid(None, None, 0.21) == pytest.approx(0.2)

    def test_mask(self):
        mask_array = np.ones((10, 10))
        mask_array[0][5] = 0
        mask_array[5][5] = 0
        mask_array[6][5] = 0
        array = geometry_profiles.array_function(lambda coordinates: 1)(-5, -5, 5, 5, 1, mask=np.ma.make_mask(mask_array))

        assert array[5][5] is None
        assert array[5][6] is not None
        assert array[6][5] is None
        assert array[0][0] is not None
        assert array[0][5] is None


class TestAuxiliary(object):
    def test__side_length(self):
        assert geometry_profiles.side_length(-5, 5, 0.1) == 100

    def test__pixel_to_coordinate(self):
        assert geometry_profiles.pixel_to_coordinate(-5, 0.1, 0) == -5
        assert geometry_profiles.pixel_to_coordinate(-5, 0.1, 100) == 5
        assert geometry_profiles.pixel_to_coordinate(-5, 0.1, 50) == 0


class MockProfile(object):
    @geometry_profiles.transform_coordinates
    def is_transformed(self, coordinates):
        return isinstance(coordinates, geometry_profiles.TransformedCoordinates)

    # noinspection PyMethodMayBeStatic
    def transform_to_reference_frame(self, coordinates):
        return geometry_profiles.TransformedCoordinates((coordinates[0] + 1, coordinates[1] + 1))

    # noinspection PyMethodMayBeStatic
    def transform_from_reference_frame(self, coordinates):
        return coordinates[0], coordinates[1]

    @geometry_profiles.transform_coordinates
    def return_coordinates(self, coordinates):
        return coordinates


class TestTransform(object):
    def test_transform(self):
        mock_profile = MockProfile()
        assert mock_profile.is_transformed((0, 0))
        assert mock_profile.return_coordinates((0, 0)) == (1, 1)
        assert mock_profile.return_coordinates(
            geometry_profiles.TransformedCoordinates((0, 0))) == geometry_profiles.TransformedCoordinates((0, 0))

    def test_symmetry(self):
        p = geometry_profiles.EllipticalProfile(2, 2, (3, 5))
        assert p.transform_from_reference_frame(p.transform_to_reference_frame((5, 7))) == (5, 7)


class TestFromProfile(object):
    def test__profile_from_profile(self):
        p = geometry_profiles.Profile(centre=(1, 1))
        assert geometry_profiles.Profile.from_profile(p).centre == (1, 1)

    def test__elliptical_profile_from_profile(self):
        p = geometry_profiles.Profile(centre=(1, 1))
        elliptical_profile = geometry_profiles.EllipticalProfile.from_profile(p, axis_ratio=1, phi=2)
        assert elliptical_profile.__class__ == geometry_profiles.EllipticalProfile
        assert elliptical_profile.centre == (1, 1)
        assert elliptical_profile.axis_ratio == 1
        assert elliptical_profile.phi == 2

    def test__profile_from_elliptical_profile(self):
        elliptical_profile = geometry_profiles.EllipticalProfile(1, 2, centre=(1, 1))
        p = geometry_profiles.Profile.from_profile(elliptical_profile)
        assert p.__class__ == geometry_profiles.Profile
        assert p.centre == (1, 1)

    def test__optional_override(self):
        elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1, phi=2)
        new_geometry_profile = geometry_profiles.EllipticalProfile.from_profile(elliptical_profile, axis_ratio=3)
        assert new_geometry_profile.phi == 2
        assert new_geometry_profile.axis_ratio == 3
