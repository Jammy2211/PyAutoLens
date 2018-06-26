from __future__ import division, print_function

import numpy as np
import pytest
from src.profiles import geometry_profiles


class TestEllipticalProfile(object):
    class TestAnglesFromXAxis(object):

        def test__profile_angle_phi_is_0__cosine_and_sin_of_phi_is_1_and_0(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=0.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

        def test__profile_angle_phi_is_45__cosine_and_sin_of_phi_follow_trig__therefore_half_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == pytest.approx(0.707, 1e-3)
            assert sin_phi == pytest.approx(0.707, 1e-3)

        def test__profile_angle_phi_is_60__cosine_and_sin_of_phi_follow_trig(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=60.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == pytest.approx(0.5, 1e-3)
            assert sin_phi == pytest.approx(0.866, 1e-3)

        def test__profile_angle_phi_is_225__cosine_and_sin_of_phi_follow_trig__therefore_negative_half_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=225.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == pytest.approx(-0.707, 1e-3)
            assert sin_phi == pytest.approx(-0.707, 1e-3)

    class TestCoordinatesToCentre(object):

        def test__profile_centre_x_0_y_0__coordinates_are_x_0_y_0__no_shift_so_shifted_coordinates_are_x_0_y_0(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_05_y_0__coordinates_are_x_0_y_0__so_x_shifts_to_negative_05(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.5, 0.0), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_0_y_05__coordinates_are_x_0_y_0__so_y_shift_to_negative_05(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_x_05_y_05__coordinates_are_x_0_y_0__x_and_y_both_shift_to_negative_05(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.5, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__use_different_profile_centre_and_coordinates(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1.0, 0.5), axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=np.array([0.2, 0.4]))

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):

        def test__profile_centre_x_0_y_0__coordinates_x_0_y_0___overlap_profile_so_radial_distance_r_is_0(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=np.array([0.0, 0.0])) == 0.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_0__therefore_radial_distance_r_is_1(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=np.array([1.0, 0.0])) == 1.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1__therefore_radial_distance_r_is_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=np.array([1.0, 1.0])) == \
                   pytest.approx(np.sqrt(2), 1e-5)

        def test__profile_centre_x_negative_1_y_0__coordinates_x_1_y_0__shifts_x_to_2_so_r_is_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(-1.0, 0.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=np.array([1.0, 0.0])) == \
                   pytest.approx(2.0, 1e-5)

        def test__profile_centre_x_2_y_2__coordinates_x_3_y_3__shifts_x_and_y_to_1_so_r_is_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(centre=(2.0, 2.0), axis_ratio=1.0, phi=0.0)

            assert elliptical_profile.coordinates_to_radius(coordinates=np.array([3.0, 3.0])) == \
                   pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAngleFromX(object):

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_0__therefore_angle_from_positive_x_is_0(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 0.0]))

            assert theta_from_x == 0.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1__therefore_angle_from_positive_x_is_45(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.0]))

            assert theta_from_x == 45.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1dot732__angle_from_positive_x_is_60(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.7320]))

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__profile_centre_x_0_y_0__coordinates_in_top_left_quandrant__angle_goes_above_90(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([-1.0, 1.0]))

            assert theta_from_x == 135.0

        def test__profile_centre_x_0_y_0__coordinates_in_bottom_left_quandrant__angle_flips_to_negative_135(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_shift = elliptical_profile.coordinates_to_centre(coordinates=np.array([-1.0, -1.0]))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -135

        def test__profile_centre_x_0_y_0__coordinates_in_bottom_right_quandrant__angle_flips_to_negative_45(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, -1.0]))

            assert theta_from_x == -45.0

        def test__profile_centre_x_2_y_1__coordinates_x_4_y_3__shift_to_x_2_y_2__angle_therefore_45(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(2.0, 1.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([4.0, 3.0]))

            assert theta_from_x == 45.0

    class TestCoordinatesAngleToProfile(object):

        def test__profile_angle_phi_and_coordinates_are_both_0__cos_0_and_sin_0_give_1_and_0(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 0.0]))

            cos_theta, sin_theta = elliptical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == 1.0
            assert sin_theta == 0.0

        def test__profile_angle_phi_and_coordinates_are_both_45___cos_0_and_sin_0_give_1_and_0(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.0]))

            cos_theta, sin_theta = elliptical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(1.0, 1e-3)
            assert sin_theta == pytest.approx(0.0, 1e-3)

        def test__profile_angle_phi_0__coordinates_angle_45__sin_45_and_cos_45_give_half_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.0]))

            cos_theta, sin_theta = elliptical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

        def test__profile_angle_phi_60__coordinates_angle_0__so_sin_and_cos_negative_60(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=60.0)

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 0.0]))

            cos_theta, sin_theta = elliptical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == np.cos(np.radians(-60)) == pytest.approx(0.5, 1e-3)
            assert sin_theta == np.sin(np.radians(-60)) == pytest.approx(-0.866, 1e-3)

        def test__include_profile_centre_offset__is_used_to_compute_coordinate_angle_from_x(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(3.0, 2.0))

            theta_from_x = elliptical_profile.coordinates_angle_from_x(coordinates=np.array([4.0, 3.0]))

            cos_theta, sin_theta = elliptical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

    class TestTransformToEllipticalReferenceFrame(object):

        def test__profile_angle_phi_is_0__coordinates_x_1_y_1__returns_same_coordinates_so_x_1_y_1(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates = np.array([1.0, 1.0])

            x, y = elliptical_profile.transform_to_reference_frame(coordinates)

            assert x == pytest.approx(1.0, 1e-3)
            assert y == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_90__coordinates_x_1_y_1__rotated_coordinates_clockwise_so_x_1_y_negative_1(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            coordinates = np.array([1.0, 1.0])

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

        def test__profile_angle_phi_180__coordinates_x_1_y_1__rotated_coordinates_clockwise_so_x_and_y_negative_1(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=180.0)

            coordinates = np.array([1.0, 1.0])

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(-1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

        def test__profile_angle_phi_270__coordinates_x_1_y_1__rotated_coordinates_clockwise_so_x_negative_1_y_1(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=270.0)

            coordinates = np.array([1.0, 1.0])

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(-1.0, 1e-3)
            assert coordinates[1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_360__rotated_coordinates_are_original_coordinates_x_1_y_(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=360.0)

            coordinates = np.array([1.0, 1.0])

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_315__coordinates_x_1_y_1__rotated_coordinates_clockwise_so_x_0_y_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            coordinates = np.array([1.0, 1.0])

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(0.0, 1e-3)
            assert coordinates[1] == pytest.approx(2 ** 0.5, 1e-3)

        def test__include_profile_centre_offset__is_used_before_rotation_is_performed(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=90.0, centre=(2.0, 3.0))

            coordinates = np.array([3.0, 4.0])

            coordinates = elliptical_profile.transform_to_reference_frame(coordinates)

            assert coordinates[0] == pytest.approx(1.0, 1e-3)
            assert coordinates[1] == pytest.approx(-1.0, 1e-3)

    class TestTransformCoordinatesBackToCartesian(object):

        def test___profile_angle_phi_0__therefore_no_rotation_and_coordinates_are_unchanged(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates(np.array([1.0, 1.0]))

            x, y = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert x == 1.0
            assert y == 1.0

        def test__profile_angle_phi_90__coordinates_x_1_y_1__rotated_counter_clockwise_to_x_negative_1_y_1(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates(np.array([1.0, 1.0]))

            x, y = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert x == pytest.approx(-1.0, 1e-3)
            assert y == 1.0

        def test___profile_angle_phi_45__coordinates_x_1_y_1__rotated_counter_clockwise_to_x_0_y_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates(np.array([1.0, 1.0]))

            x, y = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert x == pytest.approx(0.0, 1e-3)
            assert y == pytest.approx(2 ** 0.5, 1e-3)

    class TestCoordinateMovements(object):

        def test__coordinates_and_centres_of_two_lenses_are_equivalent__coordinates_are_equivalent(self):
            elliptical_profile1 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0, 0))
            coordinates1 = np.array([1.0, 1.0])
            coordinates1 = elliptical_profile1.transform_to_reference_frame(coordinates1)

            elliptical_profile2 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(-1, -1))
            coordinates2 = np.array([0.0, 0.0])
            coordinates2 = elliptical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__same_as_above_but_include_angle_phi_as_55__coordinates_are_equivalent(self):
            elliptical_profile1 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(0, 0))
            coordinates1 = np.array([1.0, 1.0])
            coordinates1 = elliptical_profile1.transform_to_reference_frame(coordinates1)

            elliptical_profile2 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            coordinates2 = np.array([0.0, 0.0])
            coordinates2 = elliptical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__coordinates_are_again_the_same_after_centre_shift__coordinates_equivalent(self):
            elliptical_profile1 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(1, 1))
            coordinates1 = np.array([1.0, 1.0])
            coordinates1 = elliptical_profile1.transform_to_reference_frame(coordinates1)

            elliptical_profile2 = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            coordinates2 = np.array([-1.0, -1.0])
            coordinates2 = elliptical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

    class TestRotateCoordinatesThenBackToCartesian(object):

        def test__coordinates_are_transformed_to_and_from_reference_frame__equal_to_original_values(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            coordinates_original = np.array([5.2221, 2.6565])

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

        def test__same_as_above_different_values(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=160.232)

            coordinates_original = np.array([3.2, -76.6])

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

        def test__same_again_another_set_of_values(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=174.342)

            coordinates_original = np.array([-42.2, -93.6])

            coordinates_elliptical = elliptical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = elliptical_profile.transform_from_reference_frame(coordinates_elliptical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

    class TestTransformedCoordinatesToEccentricRadius(object):

        def test__profile_axis_ratio_1__r_is_root_2__therefore_ecccentric_radius_is_elliptical_radius_is_root_2(self):
            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates(np.array([1.0, 1.0]))

            eccentric_radius = elliptical_profile.coordinates_to_eccentric_radius(coordinates_elliptical)

            assert eccentric_radius == pytest.approx(2.0 ** 0.5, 1e-3)

        def test__same_as_above_but_axis_ratio_is_05__r_follows_elliptical_coordinate_equation(self):
            # eccentric_radius = sqrt(axis_ratio) * sqrt(  x**2 +   (y**2 / axis_ratio**2))
            # eccentric_radius =        sqrt(0.5) * sqrt(1.0**2 + (1.0**2 /        0.5**2))
            # eccentric radius =        sqrt(0.5) * sqrt( 5 ) = 1.58113

            elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=0.5, phi=0.0)

            coordinates_elliptical = geometry_profiles.TransformedCoordinates(np.array([1.0, 1.0]))

            eccentric_radius = elliptical_profile.coordinates_to_eccentric_radius(coordinates_elliptical)

            assert eccentric_radius == pytest.approx(1.58113, 1e-3)


class TestSphericalProfile(object):
    class TestAnglesFromXAxis(object):

        def test__profile_angle_phi_is_0__cosine_and_sin_of_phi_is_1_and_0(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1, 1))

            cos_phi, sin_phi = spherical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

    class TestCoordinatesToCentre(object):

        def test__profile_centre_x_0_y_0__coordinates_are_x_0_y_0__no_shift_so_shifted_coordinates_are_x_0_y_0(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_05_y_0__coordinates_are_x_0_y_0__so_x_shifts_to_negative_05(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.5, 0.0))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__profile_centre_x_0_y_05__coordinates_are_x_0_y_0__so_y_shift_to_negative_05(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__profile_centre_x_05_y_05__coordinates_are_x_0_y_0__x_and_y_both_shift_to_negative_05(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.5, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=np.array([0.0, 0.0]))

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__use_different_profile_centre_and_coordinates(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1.0, 0.5))

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=np.array([0.2, 0.4]))

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):

        def test__profile_centre_x_0_y_0__coordinates_x_0_y_0___overlap_profile_so_radial_distance_r_is_0(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            assert spherical_profile.coordinates_to_radius(coordinates=np.array([0.0, 0.0])) == 0.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_0__therefore_radial_distance_r_is_1(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            assert spherical_profile.coordinates_to_radius(coordinates=np.array([1.0, 0.0])) == 1.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1__therefore_radial_distance_r_is_root_2(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            assert spherical_profile.coordinates_to_radius(coordinates=np.array([1.0, 1.0])) == \
                   pytest.approx(np.sqrt(2), 1e-5)

        def test__profile_centre_x_negative_1_y_0__coordinates_x_1_y_0__shifts_x_to_2_so_r_is_2(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(-1.0, 0.0))

            assert spherical_profile.coordinates_to_radius(coordinates=np.array([1.0, 0.0])) == \
                   pytest.approx(2.0, 1e-5)

        def test__profile_centre_x_2_y_2__coordinates_x_3_y_3__shifts_x_and_y_to_1_so_r_is_root_2(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(2.0, 2.0))

            assert spherical_profile.coordinates_to_radius(coordinates=np.array([3.0, 3.0])) == \
                   pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAngleFromX(object):

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_0__therefore_angle_from_positive_x_is_0(self):
            spherical_profile = geometry_profiles.SphericalProfile()

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 0.0]))

            assert theta_from_x == 0.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1__therefore_angle_from_positive_x_is_45(self):
            spherical_profile = geometry_profiles.SphericalProfile()

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.0]))

            assert theta_from_x == 45.0

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1dot732__angle_from_positive_x_is_60(self):
            spherical_profile = geometry_profiles.SphericalProfile()

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.7320]))

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__profile_centre_x_0_y_0__coordinates_in_top_left_quandrant__angle_goes_above_90(self):
            spherical_profile = geometry_profiles.SphericalProfile()

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([-1.0, 1.0]))

            assert theta_from_x == 135.0

        def test__profile_centre_x_0_y_0__coordinates_in_bottom_left_quandrant__angle_flips_to_negative_135(self):
            spherical_profile = geometry_profiles.SphericalProfile()

            coordinates_shift = spherical_profile.coordinates_to_centre(coordinates=np.array([-1.0, -1.0]))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates_shift)

            assert theta_from_x == -135

        def test__profile_centre_x_0_y_0__coordinates_in_bottom_right_quandrant__angle_flips_to_negative_45(self):
            spherical_profile = geometry_profiles.SphericalProfile()

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, -1.0]))

            assert theta_from_x == -45.0

        def test__profile_centre_x_2_y_1__coordinates_x_4_y_3__shift_to_x_2_y_2__angle_therefore_45(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(2.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([4.0, 3.0]))

            assert theta_from_x == 45.0

    class TesCoordinatestAngleToProfile(object):

        def test__profile_angle_phi_and_coordinates_are_both_0__cos_0_and_sin_0_give_1_and_0(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 0.0]))

            cos_theta, sin_theta = spherical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == 1.0
            assert sin_theta == 0.0

        def test__profile_angle_phi_and_coordinates_are_both_45___cos_0_and_sin_0_give_1_and_0(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([1.0, 1.0]))

            cos_theta, sin_theta = spherical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

        def test__include_profile_centre_offset__is_used_to_compute_coordinate_angle_from_x(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(1.0, 1.0))

            theta_from_x = spherical_profile.coordinates_angle_from_x(coordinates=np.array([2.0, 2.0]))

            cos_theta, sin_theta = spherical_profile.cos_and_sin_of_angle_to_profile(theta_from_x)

            assert cos_theta == pytest.approx(0.707, 1e-3)
            assert sin_theta == pytest.approx(0.707, 1e-3)

    class TestCoordinatesMovement(object):

        def test__profile_cenre_x_0_y_0__coordinates_x_1_y_1__no_coordinate_movement_so_x_1_y_1(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_spherical = geometry_profiles.TransformedCoordinates((1.0, 1.0))

            x, y = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert x == 1.0
            assert y == 1.0

        def test__coordinates_and_centres_of_two_lenses_are_equivalent__coordinates_are_equivalent(self):
            spherical_profile1 = geometry_profiles.SphericalProfile(centre=(0, 0))
            coordinates1 = np.array([1.0, 1.0])
            coordinates1 = spherical_profile1.transform_to_reference_frame(coordinates1)

            spherical_profile2 = geometry_profiles.SphericalProfile(centre=(-1, -1))
            coordinates2 = np.array([0.0, 0.0])
            coordinates2 = spherical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

        def test__coordinates_are_again_the_same_after_centre_shift__coordinates_equivalent(self):
            spherical_profile1 = geometry_profiles.SphericalProfile(centre=(1, 1))
            coordinates1 = np.array([1.0, 1.0])
            coordinates1 = spherical_profile1.transform_to_reference_frame(coordinates1)

            spherical_profile2 = geometry_profiles.SphericalProfile(centre=(-1, -1))
            coordinates2 = np.array([-1.0, -1.0])
            coordinates2 = spherical_profile2.transform_to_reference_frame(coordinates2)

            assert coordinates1[0] == coordinates2[0]
            assert coordinates1[1] == coordinates2[1]

    class TestTransformCoordinatesAndThenBackToCartesian(object):

        def test__profile_centre_x_0_y_0__coordinates_x_1_y_1__returns_x_1_y_1_so_same_coordinates(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates = np.array([1.0, 1.0])

            x, y = spherical_profile.transform_to_reference_frame(coordinates)

            assert x == pytest.approx(1.0, 1e-3)
            assert y == pytest.approx(1.0, 1e-3)

        def test__coordinates_are_transformed_to_and_from_reference_frame__equal_to_original_values(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = np.array([5.2221, 2.6565])

            coordinates_spherical = spherical_profile.transform_to_reference_frame(coordinates_original)

            coordinates = spherical_profile.transform_from_reference_frame(coordinates_spherical)

            assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
            assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

        def test__same_as_above_different_coordinates__equal_to_original_value(self):
            spherical_profile = geometry_profiles.SphericalProfile(centre=(0.0, 0.0))

            coordinates_original = np.array([3.2, -76.6])

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

        coordinates = list(
            map(lambda coords: (pytest.approx(coords[0], 1e-2), pytest.approx(coords[1], 1e-2)), coordinates))

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
        assert type(mock_profile.return_coordinates(np.array([0, 0]))) == np.ndarray
        assert mock_profile.is_transformed(np.array([0, 0]))
        assert (mock_profile.return_coordinates(np.array([0, 0])) == np.array([1, 1])).all()
        assert mock_profile.return_coordinates(geometry_profiles.TransformedCoordinates(np.array([0, 0]))) == \
               geometry_profiles.TransformedCoordinates((np.array([0, 0])))

    def test_symmetry(self):
        p = geometry_profiles.EllipticalProfile((3, 5), 2, 2)
        assert (p.transform_from_reference_frame(p.transform_to_reference_frame((5, 7))) == (5, 7)).all()


class TestFromProfile(object):

    def test__profile_from_profile__centre_x_1_y_1_is_passed(self):
        p = geometry_profiles.Profile(centre=(1, 1))
        assert geometry_profiles.Profile.from_profile(p).centre == (1, 1)

    def test__elliptical_profile_from_profile__centre_x_1_y_1__axis_ratio_1__phi_2__all_are_passed(self):
        p = geometry_profiles.Profile(centre=(1, 1))
        elliptical_profile = geometry_profiles.EllipticalProfile.from_profile(p, axis_ratio=1, phi=2)
        assert elliptical_profile.__class__ == geometry_profiles.EllipticalProfile
        assert elliptical_profile.centre == (1, 1)
        assert elliptical_profile.axis_ratio == 1
        assert elliptical_profile.phi == 2

    def test__profile_from_elliptical_profile__centre_x_1_y_1_is_passed(self):
        elliptical_profile = geometry_profiles.EllipticalProfile(centre=(1, 1), axis_ratio=1, phi=2)
        p = geometry_profiles.Profile.from_profile(elliptical_profile)
        assert p.__class__ == geometry_profiles.Profile
        assert p.centre == (1, 1)

    def test__elliptcal_profile_from_elliptical_profile__optionally_override_axis_ratio_with_3(self):
        elliptical_profile = geometry_profiles.EllipticalProfile(axis_ratio=1, phi=2)
        new_geometry_profile = geometry_profiles.EllipticalProfile.from_profile(elliptical_profile, axis_ratio=3)
        assert new_geometry_profile.phi == 2
        assert new_geometry_profile.axis_ratio == 3
