from __future__ import division, print_function

import numpy as np
import pytest
import profile


@pytest.fixture(name='circular')
def circular_sersic():
    return profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                      effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='elliptical')
def elliptical_sersic():
    return profile.SersicLightProfile(axis_ratio=0.5, phi=0.0, flux=1.0,
                                      effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='vertical')
def vertical_sersic():
    return profile.SersicLightProfile(axis_ratio=0.5, phi=90.0, flux=1.0,
                                      effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='dev_vaucouleurs')
def dev_vaucouleurs_profile():
    return profile.DevVaucouleursLightProfile(centre=(0.0, 0.1), axis_ratio=0.6, phi=15.0, flux=2.0,
                                              effective_radius=0.9)


@pytest.fixture(name="exponential")
def exponential_profile():
    return profile.ExponentialLightProfile(centre=(1, -1), axis_ratio=0.5, phi=45.0, flux=3.0,
                                           effective_radius=0.2)


@pytest.fixture(name="core")
def core_profile():
    return profile.CoreSersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                          effective_radius=5, sersic_index=4.0, radius_break=0.01,
                                          flux_break=0.1, gamma=1, alpha=1)


class TestEllipticalProfile(object):
    def test__coordinates_to_centre__mass_centre_zeros__no_shift(self):
        power_law = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == 0.0
        assert coordinates_shift[1] == 0.0

    def test__coordinates_to_centre__mass_centre_x_shift__x_shifts(self):
        power_law = profile.EllipticalProfile(centre=(0.5, 0.0), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == -0.5
        assert coordinates_shift[1] == 0.0

    def test__coordinates_to_centre__mass_centre_y_shift__y_shifts(self):
        power_law = profile.EllipticalProfile(centre=(0.0, 0.5), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == 0.0
        assert coordinates_shift[1] == -0.5

    def test__coordinates_to_centre__mass_centre_x_and_y_shift__x_and_y_both_shift(self):
        power_law = profile.EllipticalProfile(centre=(0.5, 0.5), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == -0.5
        assert coordinates_shift[1] == -0.5

    def test__coordinates_to_centre__mass_centre_and_coordinates__correct_shifts(self):
        power_law = profile.EllipticalProfile(centre=(1.0, 0.5), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.2, 0.4))

        assert coordinates_shift[0] == -0.8
        assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    def test__coordinates_to_radius__coordinates_overlap_mass_profile__r_is_zero(self):
        power_law = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0, 0))

        assert power_law.coordinates_to_radius(coordinates_shift) == 0.0

    def test__coordinates_to_radius__x_coordinates_is_one__r_is_one(self):
        power_law = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0))

        assert power_law.coordinates_to_radius(coordinates_shift) == 1.0

    def test__coordinates_to_radius__x_and_y_coordinates_are_one__r_is_root_two(self):
        power_law = profile.EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        assert power_law.coordinates_to_radius(coordinates_shift) == pytest.approx(np.sqrt(2), 1e-5)

    def test__angles_from_x_axis__phi_is_zero__angles_one_and_zero(self):
        power_law = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=0.0)

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == 1.0
        assert sin_phi == 0.0

    def test__angles_from_x_axis__phi_is_forty_five__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0)

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == pytest.approx(0.707, 1e-3)
        assert sin_phi == pytest.approx(0.707, 1e-3)

    def test__angles_from_x_axis__phi_is_sixty__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=60.0)

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == pytest.approx(0.5, 1e-3)
        assert sin_phi == pytest.approx(0.866, 1e-3)

    def test__coordinates_angle_from_x__angle_is_zero__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == 0.0

    def test__coordinates_angle_from_x__angle_is_forty_five__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == 45.0

    def test__coordinates_angle_from_x__angle_is_sixty__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.7320))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == pytest.approx(60.0, 1e-3)

    def test__coordinates_angle_from_x__top_left_quandrant__angle_goes_above_90(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(-1.0, 1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == 135.0

    def test__coordinates_angle_from_x__bottom_left_quandrant__angle_flips_back_to_45(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(-1.0, -1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == -135

    def test__coordinates_angle_from_x__bottom_right_quandrant__angle_flips_back_to_above_90(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, -1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == -45.0

    def test__coordinates_angle_to_mass_profile__same_angle__no_rotation(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        cos_theta, sin_theta = power_law.coordinates_angle_to_profile(theta_from_x)

        assert cos_theta == 1.0
        assert sin_theta == 0.0

    def test__coordinates_angle_to_mass_profile_both_45___no_rotation(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=45.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        cos_theta, sin_theta = power_law.coordinates_angle_to_profile(theta_from_x)

        assert cos_theta == pytest.approx(1.0, 1e-3)
        assert sin_theta == pytest.approx(0.0, 1e-3)

    def test__coordinates_angle_to_mass_profile_45_offset_same_angle__rotation_follows_trig(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        cos_theta, sin_theta = power_law.coordinates_angle_to_profile(theta_from_x)

        assert cos_theta == pytest.approx(0.707, 1e-3)
        assert sin_theta == pytest.approx(0.707, 1e-3)

    def test__coordinates_angle_to_mass_profile_negative_60_offset_same_angle__rotation_follows_trig(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=60.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        cos_theta, sin_theta = power_law.coordinates_angle_to_profile(theta_from_x)

        assert cos_theta == pytest.approx(0.5, 1e-3)
        assert sin_theta == pytest.approx(-0.866, 1e-3)

    def test__coordinates_back_to_cartesian__phi_zero__no_rotation(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates_elliptical = profile.TransformedCoordinates((1.0, 1.0))

        x, y = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert x == 1.0
        assert y == 1.0

    def test__coordinates_back_to_cartesian__phi_ninety__correct_calc(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=90.0)

        coordinates_elliptical = profile.TransformedCoordinates((1.0, 1.0))

        x, y = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert x == pytest.approx(-1.0, 1e-3)
        assert y == 1.0

    def test__coordinates_back_to_cartesian__phi_forty_five__correct_calc(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=45.0)

        coordinates_elliptical = profile.TransformedCoordinates((1.0, 1.0))

        x, y = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert x == pytest.approx(0.0, 1e-3)
        assert y == pytest.approx(2 ** 0.5, 1e-3)

    def test__rotate_to_elliptical__phi_is_zero__returns_same_coordinates(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0)

        coordinates = (1.0, 1.0)

        x, y = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert x == pytest.approx(1.0, 1e-3)
        assert y == pytest.approx(1.0, 1e-3)

    def test__rotate_to_elliptical__phi_is_ninety__correct_rotation(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=90.0)

        # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
        # clockwise

        coordinates = (1.0, 1.0)

        coordinates = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert coordinates[0] == pytest.approx(1.0, 1e-3)
        assert coordinates[1] == pytest.approx(-1.0, 1e-3)

    def test__rotate_to_elliptical__phi_is_one_eighty__correct_rotation(self):
        # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
        # clockwise

        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=180.0)

        coordinates = (1.0, 1.0)

        coordinates = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert coordinates[0] == pytest.approx(-1.0, 1e-3)
        assert coordinates[1] == pytest.approx(-1.0, 1e-3)

    def test__rotate_to_elliptical__phi_is_two_seventy__correct_rotation(self):
        # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
        # clockwise

        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=270.0)

        coordinates = (1.0, 1.0)

        coordinates = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert coordinates[0] == pytest.approx(-1.0, 1e-3)
        assert coordinates[1] == pytest.approx(1.0, 1e-3)

    def test__rotate_to_elliptical__phi_is_three_sixty__correct_rotation(self):
        # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
        # clockwise

        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=360.0)

        coordinates = (1.0, 1.0)

        coordinates = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert coordinates[0] == pytest.approx(1.0, 1e-3)
        assert coordinates[1] == pytest.approx(1.0, 1e-3)

    def test__rotate_to_elliptical__phi_is_three_one_five__correct_rotation(self):
        # NOTE - whilst the profile and coordinates are defined counter-clockwise from x, the rotation is performed
        # clockwise

        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=315.0)

        coordinates = (1.0, 1.0)

        coordinates = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert coordinates[0] == pytest.approx(0.0, 1e-3)
        assert coordinates[1] == pytest.approx(2 ** 0.5, 1e-3)

    def test__rotate_to_elliptical__moving_lens_and_coordinates__same_answer(self):
        power_law1 = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0, 0))
        coordinates1 = (1.0, 1.0)
        coordinates1 = power_law1.coordinates_rotate_to_elliptical(coordinates1)

        power_law2 = profile.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(-1, -1))
        coordinates2 = (0.0, 0.0)
        coordinates2 = power_law2.coordinates_rotate_to_elliptical(coordinates2)

        assert coordinates1[0] == coordinates2[0]
        assert coordinates1[1] == coordinates2[1]

    def test__rotate_to_elliptical__moving_lens_and_coordinates_with_phi__same_answer(self):
        power_law1 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(0, 0))
        coordinates1 = (1.0, 1.0)
        coordinates1 = power_law1.coordinates_rotate_to_elliptical(coordinates1)

        power_law2 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
        coordinates2 = (0.0, 0.0)
        coordinates2 = power_law2.coordinates_rotate_to_elliptical(coordinates2)

        assert coordinates1[0] == coordinates2[0]
        assert coordinates1[1] == coordinates2[1]

    def test__rotate_to_elliptical__coordinates_both_on_centre___same_answer(self):
        power_law1 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(1, 1))
        coordinates1 = (1.0, 1.0)
        coordinates1 = power_law1.coordinates_rotate_to_elliptical(coordinates1)

        power_law2 = profile.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
        coordinates2 = (-1.0, -1.0)
        coordinates2 = power_law2.coordinates_rotate_to_elliptical(coordinates2)

        assert coordinates1[0] == coordinates2[0]
        assert coordinates1[1] == coordinates2[1]

    def test_rotate_to_elliptical_coordinates_back_to_cartesian__are_consistent(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=315.0)

        coordinates_original = (5.2221, 2.6565)

        coordinates_elliptical = power_law.coordinates_rotate_to_elliptical(coordinates_original)

        coordinates = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
        assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

    def test_rotate_to_elliptical_coordinates_back_to_cartesian_2__are_consistent(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=160.232)

        coordinates_original = (3.2, -76.6)

        coordinates_elliptical = power_law.coordinates_rotate_to_elliptical(coordinates_original)

        coordinates = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
        assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

    def test_rotate_to_elliptical_coordinates_back_to_cartesian_3__are_consistent(self):
        power_law = profile.EllipticalProfile(axis_ratio=1.0, phi=174.342)

        coordinates_original = (-42.2, -93.6)

        coordinates_elliptical = power_law.coordinates_rotate_to_elliptical(coordinates_original)

        coordinates = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
        assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)


class TestSphericalProfile(object):
    def test__coordinates_to_centre__mass_centre_zeros__no_shift(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == 0.0
        assert coordinates_shift[1] == 0.0

    def test__coordinates_to_centre__mass_centre_x_shift__x_shifts(self):
        power_law = profile.SphericalProfile(centre=(0.5, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == -0.5
        assert coordinates_shift[1] == 0.0

    def test__coordinates_to_centre__mass_centre_y_shift__y_shifts(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.5))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == 0.0
        assert coordinates_shift[1] == -0.5

    def test__coordinates_to_centre__mass_centre_x_and_y_shift__x_and_y_both_shift(self):
        power_law = profile.SphericalProfile(centre=(0.5, 0.5))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == -0.5
        assert coordinates_shift[1] == -0.5

    def test__coordinates_to_centre__mass_centre_and_coordinates__correct_shifts(self):
        power_law = profile.SphericalProfile(centre=(1.0, 0.5))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.2, 0.4))

        assert coordinates_shift[0] == -0.8
        assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    def test__coordinates_to_radius__coordinates_overlap_mass_profile__r_is_zero(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0, 0))

        assert power_law.coordinates_to_radius(coordinates_shift) == 0.0

    def test__coordinates_to_radius__x_coordinates_is_one__r_is_one(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0))

        assert power_law.coordinates_to_radius(coordinates_shift) == 1.0

    def test__coordinates_to_radius__x_and_y_coordinates_are_one__r_is_root_two(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        assert power_law.coordinates_to_radius(coordinates_shift) == pytest.approx(np.sqrt(2), 1e-5)

    def test__angles_from_x_axis__phi_is_zero__angles_one_and_zero(self):
        power_law = profile.SphericalProfile(centre=(1, 1))

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == 1.0
        assert sin_phi == 0.0

    def test__coordinates_angle_from_x__angle_is_zero__angles_follow_trig(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == 0.0

    def test__coordinates_angle_from_x__angle_is_forty_five__angles_follow_trig(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == 45.0

    def test__coordinates_angle_from_x__angle_is_sixty__angles_follow_trig(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.7320))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == pytest.approx(60.0, 1e-3)

    def test__coordinates_angle_from_x__top_left_quandrant__angle_goes_above_90(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(-1.0, 1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == 135.0

    def test__coordinates_angle_from_x__bottom_left_quandrant__angle_flips_back_to_45(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(-1.0, -1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == -135

    def test__coordinates_angle_from_x__bottom_right_quandrant__angle_flips_back_to_above_90(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, -1.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        assert theta_from_x == -45.0

    def test__coordinates_angle_to_mass_profile__same_angle__no_rotation(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        theta_from_x = power_law.coordinates_angle_from_x(coordinates_shift)

        cos_theta, sin_theta = power_law.coordinates_angle_to_profile(theta_from_x)

        assert cos_theta == 1.0
        assert sin_theta == 0.0

    def test__coordinates_back_to_cartesian__phi_zero__no_rotation(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_elliptical = (1.0, 1.0)

        x, y = power_law.coordinates_back_to_cartesian(profile.TransformedCoordinates(coordinates_elliptical))

        assert x == 1.0
        assert y == 1.0

    def test__rotate_to_elliptical__phi_is_zero__returns_same_coordinates(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates = (1.0, 1.0)

        x, y = power_law.coordinates_rotate_to_elliptical(coordinates)

        assert x == pytest.approx(1.0, 1e-3)
        assert y == pytest.approx(1.0, 1e-3)

    def test_rotate_to_elliptical_coordinates_back_to_cartesian__are_consistent(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_original = (5.2221, 2.6565)

        coordinates_elliptical = power_law.coordinates_rotate_to_elliptical(coordinates_original)

        coordinates = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-5)
        assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-5)

    def test_rotate_to_elliptical_coordinates_back_to_cartesian_2__are_consistent(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_original = (3.2, -76.6)

        coordinates_elliptical = power_law.coordinates_rotate_to_elliptical(coordinates_original)

        coordinates = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
        assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)

    def test_rotate_to_elliptical_coordinates_back_to_cartesian_3__are_consistent(self):
        power_law = profile.SphericalProfile(centre=(0.0, 0.0))

        coordinates_original = (-42.2, -93.6)

        coordinates_elliptical = power_law.coordinates_rotate_to_elliptical(coordinates_original)

        coordinates = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert coordinates[0] == pytest.approx(coordinates_original[0], 1e-2)
        assert coordinates[1] == pytest.approx(coordinates_original[1], 1e-2)


class TestLightProfile(object):
    def test__setup_sersic__correct_values(self, circular):
        assert circular.x_cen == 0.0
        assert circular.y_cen == 0.0
        assert circular.axis_ratio == 1.0
        assert circular.phi == 0.0
        assert circular.flux == 1.0
        assert circular.effective_radius == 0.6
        assert circular.sersic_index == 4.0
        assert circular.sersic_constant == pytest.approx(7.66925, 1e-3)

    def test__flux_at_radius__correct_value(self, circular):
        flux_at_radius = circular.flux_at_radius(radius=1.0)

        assert flux_at_radius == pytest.approx(0.351797, 1e-3)

    def test__flux_at_radius_2__correct_value(self):
        sersic = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=3.0,
                                            effective_radius=2.0, sersic_index=2.0)

        flux_at_radius = sersic.flux_at_radius(
            radius=1.5)  # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

        assert flux_at_radius == pytest.approx(4.90657319276, 1e-3)

    def test__setup_exponential__correct_values(self, exponential):
        assert exponential.x_cen == 1.0
        assert exponential.y_cen == -1.0
        assert exponential.axis_ratio == 0.5
        assert exponential.phi == 45.0
        assert exponential.flux == 3.0
        assert exponential.effective_radius == 0.2
        assert exponential.sersic_index == 1.0
        assert exponential.sersic_constant == pytest.approx(1.678378, 1e-3)

    def test__setup_dev_vaucouleurs__correct_values(self, dev_vaucouleurs):
        assert dev_vaucouleurs.x_cen == 0.0
        assert dev_vaucouleurs.y_cen == 0.1
        assert dev_vaucouleurs.axis_ratio == 0.6
        assert dev_vaucouleurs.phi == 15.0
        assert dev_vaucouleurs.flux == 2.0
        assert dev_vaucouleurs.effective_radius == 0.9
        assert dev_vaucouleurs.sersic_index == 4.0
        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66925, 1e-3)

    def test__core_sersic_light_profile(self, core):
        assert core.flux_at_radius(0.01) == 0.1


class TestEllipticalPowerLaw(object):
    def test__setup_elliptical_power_law__correct_values(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                          einstein_radius=1.0
                                                          , slope=2.0)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.0
        assert power_law.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5


class TestSurfaceDensity(object):
    def test__flip_coordinates_lens_center__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        surface_density_1 = power_law.compute_surface_density(coordinates=(1.0, 1.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        surface_density_2 = power_law.compute_surface_density(coordinates=(0.0, 0.0))

        assert surface_density_1 == surface_density_2

    def test__rotation_coordinates_90_circular__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        surface_density_1 = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                          einstein_radius=1.0, slope=2.0)

        surface_density_2 = power_law.compute_surface_density(coordinates=(0.0, 1.0))

        assert surface_density_1 == surface_density_2

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                          einstein_radius=1.0, slope=2.2)

        surface_density_1 = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                          einstein_radius=1.0, slope=2.2)

        surface_density_2 = power_law.compute_surface_density(coordinates=(0.0, 1.0))

        assert surface_density_1 == surface_density_2

    def test__simple_case__correct_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        surface_density = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.5, 1e-3)

    def test__double_einr__doubles_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=2.0, slope=2.0)

        surface_density = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.5 * 2.0, 1e-3)

    def test__different_axis_ratio__new_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        surface_density = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.66666, 1e-3)

    def test__slope_increase__new_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=2.3)

        surface_density = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.466666, 1e-3)

    def test__slope_decrease__new_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=2.0, slope=1.7)

        surface_density = power_law.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(1.4079, 1e-3)


class TestPotential(object):
    def test__flip_coordinates_lens_center__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        potential_1 = power_law.compute_potential(coordinates=(1.0, 1.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        potential_2 = power_law.compute_potential(coordinates=(0.0, 0.0))

        assert potential_1 == potential_2

    def test__rotation_coordinates_90_circular__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        potential_1 = power_law.compute_potential(coordinates=(1.0, 0.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                          einstein_radius=1.0, slope=2.0)

        potential_2 = power_law.compute_potential(coordinates=(0.0, 1.0))

        assert potential_1 == potential_2

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                          einstein_radius=1.0, slope=2.2)

        potential_1 = power_law.compute_potential(coordinates=(1.0, 0.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                          einstein_radius=1.0, slope=2.2)

        potential_2 = power_law.compute_potential(coordinates=(0.0, 1.0))

        assert potential_1 == potential_2

    def test__compare_to_isothermal_ratio_of_two_potentials__same_ratio(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0)

        potential_isothermal_1 = isothermal.compute_potential(coordinates=(0.1625, 0.1625))

        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.8, phi=45.0,
                                                             einstein_radius=1.6)

        potential_isothermal_2 = isothermal.compute_potential(coordinates=(0.1625, 0.1625))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                          einstein_radius=1.0, slope=2.0)

        potential_power_law_1 = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                          einstein_radius=1.6, slope=2.0)

        potential_power_law_2 = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        isothermal_ratio = potential_isothermal_1 / potential_isothermal_2
        power_law_ratio = potential_power_law_1 / potential_power_law_2

        assert isothermal_ratio == pytest.approx(power_law_ratio, 1e-3)

    def test__compare_to_fortran_ratio_of_two_power_laws__same_ratio(sefl):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=2.2)

        potential_1 = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=2.1)

        potential_2 = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        ratio = potential_1 / potential_2

        assert ratio == pytest.approx((1.53341 / 1.34381), 1e-3)

    def test__compare_to_isothermal__same_potential(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0)

        potential_isothermal = isothermal.compute_potential(coordinates=(0.1625, 0.1625))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                          einstein_radius=1.0, slope=2.0)

        potential_power_law = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential_isothermal == pytest.approx(potential_power_law, 1e-3)

    def test__compare_to_fortran_values_slope_22__same_potential(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=2.2)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(1.53341, 1e-3)

    def test__compare_to_fortran_values_slope_21__same_potential(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=2.1)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(1.34381, 1e-3)

    def test__compare_to_fortran_values_slope_20__same_potential(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=2.0)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(1.19268, 1e-3)

    def test__compare_to_fortran_values_slope_19__same_potential(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=1.9)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(1.06949, 1e-3)

    def test__compare_to_fortran_values_slope_18__same_potential(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=1.8)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(0.96723, 1e-3)


class TestDeflections(object):
    def test__flip_coordinates_lens_center__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        deflection_angle_1 = power_law.compute_deflection_angle(coordinates=(1.0, 1.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        deflection_angle_2 = power_law.compute_deflection_angle(coordinates=(0.0, 0.0))

        # Foro deflection angles, a flip of coordinates also reverses the deflection angles
        deflection_angle_2 = list(map(lambda l: -1.0 * l, deflection_angle_2))

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[0], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[1], 1e-5)

    def test__rotation_coordinates_90_circular__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        deflection_angle_1 = power_law.compute_deflection_angle(coordinates=(1.0, 0.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                          einstein_radius=1.0, slope=2.0)

        deflection_angle_2 = power_law.compute_deflection_angle(coordinates=(0.0, 1.0))

        # Foro deflection angles, a 90 degree rtation flips the x / y coordinates

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[1], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[0], 1e-5)

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                          einstein_radius=1.0, slope=2.2)

        deflection_angle_1 = power_law.compute_deflection_angle(coordinates=(1.0, 0.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                          einstein_radius=1.0, slope=2.2)

        deflection_angle_2 = power_law.compute_deflection_angle(coordinates=(0.0, 1.0))

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[1], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[0], 1e-5)

    def test__identical_as_sie_compare_ratio__same_defls(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0)

        defls_isothermal = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        defls_power_law = power_law.compute_deflection_angle(coordinates=(1.0, 1.0))

        ratio_isothermal = defls_isothermal[0] / defls_isothermal[1]
        ratio_power_law = defls_power_law[0] / defls_power_law[1]

        assert ratio_isothermal == pytest.approx(ratio_power_law, 1e-3)

    def test__identical_as_sie_compare_values__same_defls(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0)

        defls_isothermal = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        defls_power_law = power_law.compute_deflection_angle(coordinates=(1.0, 1.0))

        assert defls_isothermal[0] == pytest.approx(defls_power_law[0], 1e-3)
        assert defls_isothermal[1] == pytest.approx(defls_power_law[1], 1e-3)

    def test__compare_to_fortran_slope_isothermal__same_defls(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=2.0)

        defls = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] == pytest.approx(0.50734, 1e-3)
        assert defls[1] == pytest.approx(0.79421, 1e-3)

    def test__compare_to_fortran_slope_above_isothermal__same_defls(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=2.5)

        defls = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] == pytest.approx(0.99629, 1e-3)
        assert defls[1] == pytest.approx(1.29641, 1e-3)

    def test__compare_to_fortran_slope_below_isothermal__same_defls(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                          einstein_radius=1.0, slope=1.5)

        defls = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] == pytest.approx(0.26729, 1e-3)
        assert defls[1] == pytest.approx(0.48036, 1e-3)

    def test__compare_to_fortran_different_values__same_defls(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.3, slope=1.9)

        defls = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] / defls[1] == pytest.approx(-0.53353, 1e-3)
        assert defls[0] == pytest.approx(-0.60205, 1e-3)
        assert defls[1] == pytest.approx(1.12841, 1e-3)

    def test__compare_to_fortran_different_values_2__same_defls(self):
        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=150.0,
                                                          einstein_radius=1.3, slope=2.2)

        defls = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] / defls[1] == pytest.approx(-0.27855, 1e-3)
        assert defls[0] == pytest.approx(-0.35096, 1e-3)
        assert defls[1] == pytest.approx(1.25995, 1e-3)


class TestCoredEllipticalPowerLaw(object):
    def test__setup_cored_elliptical_power_law__correct_values(self):
        power_law = profile.CoredEllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                               einstein_radius=1.0, slope=2.2, core_radius=0.1)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.2
        assert power_law.core_radius == 0.1
        assert power_law.einstein_radius_rescaled == pytest.approx(0.40480,
                                                                   1e-3)  # (3 - slope) / (1 + axis_ratio) * (1.0+0.1**2)**1.2 = (3 - 2) / (1 + 1) * (1.1)**1.2 = 0.5


class TestSurfaceDensity2(object):
    def test__function__gives_correct_values(self):
        power_law = profile.CoredEllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                               einstein_radius=1.0, slope=2.2, core_radius=0.1)

        kappa = power_law.surface_density_func(eta=1.0)

        assert kappa == pytest.approx(0.40239, 1e-4)

    def test__function__same_as_power_law_no_core(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.)

        kappa_core = power_law_core.surface_density_func(eta=3.0)

        power_law = profile.EllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                          einstein_radius=1.0, slope=2.2)

        kappa = power_law.surface_density_func(eta=3.0)

        assert kappa == kappa_core

    def test__flip_coordinates_lens_center__same_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        surface_density_1 = power_law_core.compute_surface_density(coordinates=(1.0, 1.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        surface_density_2 = power_law_core.compute_surface_density(coordinates=(0.0, 0.0))

        assert surface_density_1 == surface_density_2

    def test__rotation_coordinates_90_circular__same_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        surface_density_1 = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        surface_density_2 = power_law_core.compute_surface_density(coordinates=(0.0, 1.0))

        assert surface_density_1 == surface_density_2

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.2)

        surface_density_1 = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.2)

        surface_density_2 = power_law_core.compute_surface_density(coordinates=(0.0, 1.0))

        assert surface_density_1 == surface_density_2

    def test__simple_case__correct_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        surface_density = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.50990, 1e-3)

    def test__double_einr__new_value_now_isnt_quite_double(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=2.0, slope=2.0, core_radius=0.2)

        surface_density = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(1.0002, 1e-3)

    def test__different_axis_ratio__new_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        surface_density = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.

        assert surface_density == pytest.approx(0.50990 * 1.33333, 1e-3)

    def test__slope_increase__new_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.3, core_radius=0.2)

        surface_density = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.4787, 1e-3)

    def test__slope_decrease__new_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                                    einstein_radius=2.0, slope=1.7, core_radius=0.2)

        surface_density = power_law_core.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(1.4079, 1e-3)


class TestPotential2(object):
    def test__compare_to_fortran__same_values2(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0)

        potential = isothermal.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(0.18525, 1e-3)

    def test__flip_coordinates_lens_center__same_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        potential_1 = power_law_core.compute_potential(coordinates=(1.0, 1.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        potential_2 = power_law_core.compute_potential(coordinates=(0.0, 0.0))

        assert potential_1 == potential_2

    def test__rotation_coordinates_90_circular__same_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        potential_1 = power_law_core.compute_potential(coordinates=(1.0, 0.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.2)

        potential_2 = power_law_core.compute_potential(coordinates=(0.0, 1.0))

        assert potential_1 == potential_2

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.2)

        potential_1 = power_law_core.compute_potential(coordinates=(1.0, 0.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.2)

        potential_2 = power_law_core.compute_potential(coordinates=(0.0, 1.0))

        assert potential_1 == potential_2

    def test__same_as_sie_for_no_core(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.)

        potential_core = power_law_core.compute_potential(coordinates=(0.1, 0.1))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                          einstein_radius=1.0, slope=2.2)

        potential = power_law.compute_potential(coordinates=(0.1, 0.1))

        assert potential_core == potential

    def test__value_via_fortran__same_value(self):
        power_law = profile.CoredEllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                               einstein_radius=1.3, slope=1.8, core_radius=0.2)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(0.72932, 1e-3)

    def test__value_via_fortran_2__same_value(self):
        power_law = profile.CoredEllipticalPowerLawMassProfile(centre=(-0.2, 0.2), axis_ratio=0.6, phi=120.0,
                                                               einstein_radius=0.5, slope=2.4, core_radius=0.5)

        potential = power_law.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(0.040915, 1e-3)


class TestDeflections2(object):
    def test__flip_coordinates_lens_center__flips_deflection_angles(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.3)

        deflection_angle_1 = power_law_core.compute_deflection_angle(coordinates=(1.0, 1.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.3)

        deflection_angle_2 = power_law_core.compute_deflection_angle(coordinates=(0.0, 0.0))

        # Foro deflection angles, a flip of coordinates also reverses the deflection angles
        deflection_angle_2 = list(map(lambda l: -1.0 * l, deflection_angle_2))

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[0], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[1], 1e-5)

    def test__rotation_coordinates_90_circular__flips_x_and_y_deflection_angles(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.3)

        deflection_angle_1 = power_law_core.compute_deflection_angle(coordinates=(1.0, 0.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                    einstein_radius=1.0, slope=2.0, core_radius=0.3)

        deflection_angle_2 = power_law_core.compute_deflection_angle(coordinates=(0.0, 1.0))

        # Foro deflection angles, a 90 degree rtation flips the x / y coordinates

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[1], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[0], 1e-5)

    def test__rotation_90_ellpitical_cordinates_on_corners__flips_x_and_y_deflection_angles(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.3)

        deflection_angle_1 = power_law_core.compute_deflection_angle(coordinates=(1.0, 0.0))

        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                    einstein_radius=1.0, slope=2.2, core_radius=0.3)

        deflection_angle_2 = power_law_core.compute_deflection_angle(coordinates=(0.0, 1.0))

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[1], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[0], 1e-5)

    def test__compute_deflection__same_as_power_law_for_core_0(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.3, -0.1), axis_ratio=0.7, phi=60.0,
                                                                    einstein_radius=1.1, slope=2.1, core_radius=0.0)

        deflections_core = power_law_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        power_law = profile.EllipticalPowerLawMassProfile(centre=(0.3, -0.1), axis_ratio=0.7, phi=60.0,
                                                          einstein_radius=1.1, slope=2.1)

        deflections_power_law = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert deflections_core[0] == pytest.approx(deflections_power_law[0], 1e-6)
        assert deflections_core[1] == pytest.approx(deflections_power_law[1], 1e-6)

    def test__compute_deflection__ratio_via_fortran__same_ratio(self):
        power_law_core = profile.CoredEllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                    einstein_radius=1.3, slope=1.8, core_radius=0.2)

        deflections = power_law_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        ratio = deflections[0] / deflections[1]

        assert ratio == pytest.approx(-0.55607, 1e-3)

    def test__compute_deflection__value_via_fortran__same_value(self):
        power_law = profile.CoredEllipticalPowerLawMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                               einstein_radius=1.3, slope=1.8, core_radius=0.2)

        deflections = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert deflections[0] == pytest.approx(-0.56229, 1e-3)
        assert deflections[1] == pytest.approx(1.0112, 1e-3)

    def test__compute_deflection__value_via_fortran_2__same_value(self):
        power_law = profile.CoredEllipticalPowerLawMassProfile(centre=(-0.2, 0.2), axis_ratio=0.6, phi=120.0,
                                                               einstein_radius=0.5, slope=2.4, core_radius=0.5)

        deflections = power_law.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert deflections[0] == pytest.approx(0.20117, 1e-3)
        assert deflections[1] == pytest.approx(0.01960, 1e-3)


class TestEllipticalIsothermal(object):
    # TODO: Add surface density / more potential tests

    def test__setup_elliptical_power_law__correct_values(self):
        power_law = profile.EllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                            einstein_radius=1.0)

        assert power_law.x_cen == 1.0
        assert power_law.y_cen == 1.0
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.0
        assert power_law.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

    def test__coordinates_to_eccentric_radius(self, elliptical):
        assert elliptical.coordinates_to_eccentric_radius((1, 1)) == pytest.approx(
            elliptical.coordinates_to_eccentric_radius(
                (-1, -1)), 1e-10)

    def test__flux_at_coordinates(self, elliptical):
        assert elliptical.flux_at_coordinates((1, 1)) == pytest.approx(
            elliptical.flux_at_coordinates((-1, -1)), 1e-10)


class TestEllipticalDeflections(object):
    def test_no_coordinate_rotation__correct_values(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        # normalization = 2.0*(1/(1+q))*einr*q / (sqrt(1-q**2))
        # normalization = (1/1.5)*1*0.5 / (sqrt(0.75) = 0.7698
        # Psi = sqrt (q ** 2 * (x**2) + y**2 = 0.25 + 1) = sqrt(1.25)

        # defl_x = normalization * atan(sqrt(1-q**2) x / Psi )
        # defl_x = 0.7698 * atan(sqrt(0.75)/sqrt(1.25) = 0.50734

        # defl_y = normalization * atanh(sqrt(1-q**2) y / (Psi) )

        assert defls[0] == pytest.approx(0.50734, 1e-3)
        assert defls[1] == pytest.approx(0.79420, 1e-3)

    def test_coordinate_rotation_90__defl_x_same_defl_y_flip(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=90.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        assert defls[0] == pytest.approx(0.79420, 1e-3)
        assert defls[1] == pytest.approx(0.50734, 1e-3)

    def test_coordinate_rotation_180__both_defl_flip(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=180.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        assert defls[0] == pytest.approx(0.50734, 1e-3)
        assert defls[1] == pytest.approx(0.79420, 1e-3)

    def test_coordinate_rotation_45__defl_y_zero_new_defl_x(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        # 45 degree aligns the mass profile with the axes, so there is no deflection acoss y.

        assert defls[0] == pytest.approx(0.5698, 1e-3)
        assert defls[1] == pytest.approx(0.5700, 1e-3)

    def test_double_einr__double_defl_angles(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=2.0)

        defls = isothermal.compute_deflection_angle(coordinates=(1.0, 1.0))

        assert defls[0] == pytest.approx(0.5698 * 2.0, 1e-3)
        assert defls[1] == pytest.approx(0.5700 * 2.0, 1e-3)

    def test_flip_coordinaates_and_centren__same_defl(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(-1, -1), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(0.0, 0.0))

        assert defls[0] == pytest.approx(0.50734, 1e-3)
        assert defls[1] == pytest.approx(0.79420, 1e-3)

    def test_another_q__new_defl_values(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.25, phi=0.0,
                                                             einstein_radius=2.0)

        defls = isothermal.compute_deflection_angle(coordinates=(-1.0, -1.0))

        assert defls[0] == pytest.approx(-0.62308, 1e-3)
        assert defls[1] == pytest.approx(-1.43135, 1e-3)

    def test_compare_to_fortran__same_values(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=0.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] == pytest.approx(0.50734, 1e-3)
        assert defls[1] == pytest.approx(0.79421, 1e-3)

    def test_compare_to_fortran__same_values2(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0)

        defls = isothermal.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert defls[0] == pytest.approx(0.57002, 1e-3)
        assert defls[1] == pytest.approx(0.57002, 1e-3)


class TestCoredEllipticalIsothermal(object):
    def test__setup_elliptical_isothermal_core__correct_values(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        assert isothermal_core.x_cen == 1.0
        assert isothermal_core.y_cen == 1.0
        assert isothermal_core.axis_ratio == 1.0
        assert isothermal_core.phi == 45.0
        assert isothermal_core.einstein_radius == 1.0
        assert isothermal_core.slope == 2.0
        assert isothermal_core.core_radius == 0.2
        assert isothermal_core.einstein_radius_rescaled == 0.52  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5


class TestSurfaceDensity3(object):
    def test__function__gives_correct_values(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                       einstein_radius=1.0, core_radius=0.1)

        kappa = isothermal_core.surface_density_func(eta=1.0)

        assert kappa == pytest.approx(0.50249, 1e-4)

    def test__function__same_as_isothermal_core_no_core(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                                       einstein_radius=1.0, core_radius=0.0)

        kappa_core = isothermal_core.surface_density_func(eta=3.0)

        isothermal = profile.EllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                             einstein_radius=1.0)

        kappa = isothermal.surface_density_func(eta=3.0)

        assert kappa == kappa_core

    def test__flip_coordinates_lens_center__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density_1 = isothermal_core.compute_surface_density(coordinates=(1.0, 1.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density_2 = isothermal_core.compute_surface_density(coordinates=(0.0, 0.0))

        assert surface_density_1 == surface_density_2

    def test__rotation_coordinates_90_circular__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density_1 = isothermal_core.compute_surface_density(coordinates=(1.0, 0.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density_2 = isothermal_core.compute_surface_density(coordinates=(0.0, 1.0))

        assert surface_density_1 == surface_density_2

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density_1 = isothermal_core.compute_surface_density(coordinates=(1.0, 0.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density_2 = isothermal_core.compute_surface_density(coordinates=(0.0, 1.0))

        assert surface_density_1 == surface_density_2

    def test__simple_case__correct_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density = isothermal_core.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(0.50990, 1e-3)

    def test__double_einr__new_value_now_isnt_quite_double(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=2.0, core_radius=0.2)

        surface_density = isothermal_core.compute_surface_density(coordinates=(1.0, 0.0))

        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0

        assert surface_density == pytest.approx(1.0002, 1e-3)

    def test__different_axis_ratio__new_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        surface_density = isothermal_core.compute_surface_density(coordinates=(1.0, 0.0))

        # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.

        assert surface_density == pytest.approx(0.50990 * 1.33333, 1e-3)


class TestCoredPotential(object):
    def test__flip_coordinates_lens_center__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        potential_1 = isothermal_core.compute_potential(coordinates=(1.0, 1.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        potential_2 = isothermal_core.compute_potential(coordinates=(0.0, 0.0))

        assert potential_1 == potential_2

    def test__rotation_coordinates_90_circular__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        potential_1 = isothermal_core.compute_potential(coordinates=(1.0, 0.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        potential_2 = isothermal_core.compute_potential(coordinates=(0.0, 1.0))

        assert potential_1 == potential_2

    def test__rotation_90_ellpitical_cordinates_on_corners__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.2)

        potential_1 = isothermal_core.compute_potential(coordinates=(1.0, 0.0))

        isothermal = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                  einstein_radius=1.0, core_radius=0.2)

        potential_2 = isothermal.compute_potential(coordinates=(0.0, 1.0))

        assert potential_1 == potential_2

    # def test__same_as_sie_for_no_core(self):
    #
    #     isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=0.9, phi=45.0,
    #                                                       einstein_radius=1.0, core_radius=0.)
    #
    #     potential_core = isothermal_core.compute_potential(coordinates=(0.1, 0.1))
    #
    #     isothermal = profile.EllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=0.9, phi=45.0,
    #                                                       einstein_radius=1.0)
    #
    #     potential = isothermal.compute_potential(coordinates=(0.1, 0.1))
    #
    #     assert potential_core == potential

    def test__compute_potential__ratio_via_fortran__same_ratio(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                       einstein_radius=1.3, core_radius=0.2)

        potential_1 = isothermal_core.compute_potential(coordinates=(0.1625, 0.1625))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(-0.2, 0.2), axis_ratio=0.6, phi=120.0,
                                                                       einstein_radius=0.5, core_radius=0.5)

        potential_2 = isothermal_core.compute_potential(coordinates=(0.1625, 0.1625))

        ratio = potential_1 / potential_2

        assert ratio == pytest.approx(0.76642 / 0.06036, 1e-3)

    def test__compute_potential__value_via_fortran__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                       einstein_radius=1.3, core_radius=0.2)

        potential = isothermal_core.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(0.76642, 1e-3)

    def test__compute_potential__value_via_fortran_2__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(-0.2, 0.2), axis_ratio=0.6, phi=120.0,
                                                                       einstein_radius=0.5, core_radius=0.5)

        potential = isothermal_core.compute_potential(coordinates=(0.1625, 0.1625))

        assert potential == pytest.approx(0.06036, 1e-3)


class TestDeflections3(object):
    def test__flip_coordinates_lens_center__flips_deflection_angles(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.3)

        deflection_angle_1 = isothermal_core.compute_deflection_angle(coordinates=(1.0, 1.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.3)

        deflection_angle_2 = isothermal_core.compute_deflection_angle(coordinates=(0.0, 0.0))

        # Foro deflection angles, a flip of coordinates also reverses the deflection angles
        deflection_angle_2 = list(map(lambda l: -1.0 * l, deflection_angle_2))

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[0], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[1], 1e-5)

    def test__rotation_coordinates_90_circular__flips_x_and_y_deflection_angles(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.3)

        deflection_angle_1 = isothermal_core.compute_deflection_angle(coordinates=(1.0, 0.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=90.0,
                                                                       einstein_radius=1.0, core_radius=0.3)

        deflection_angle_2 = isothermal_core.compute_deflection_angle(coordinates=(0.0, 1.0))

        # Foro deflection angles, a 90 degree rtation flips the x / y coordinates

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[1], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[0], 1e-5)

    def test__rotation_90_ellpitical_cordinates_on_corners__flips_x_and_y_deflection_angles(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0,
                                                                       einstein_radius=1.0, core_radius=0.3)

        deflection_angle_1 = isothermal_core.compute_deflection_angle(coordinates=(1.0, 0.0))

        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                                       einstein_radius=1.0, core_radius=0.3)

        deflection_angle_2 = isothermal_core.compute_deflection_angle(coordinates=(0.0, 1.0))

        assert deflection_angle_1[0] == pytest.approx(deflection_angle_2[1], 1e-5)
        assert deflection_angle_1[1] == pytest.approx(deflection_angle_2[0], 1e-5)

    def test__same_as_isothermal_core_for_core_0(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.3, -0.1), axis_ratio=0.7, phi=60.0,
                                                                       einstein_radius=1.1, core_radius=0.0)

        deflections_core = isothermal_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        isothermal_core = profile.EllipticalIsothermalMassProfile(centre=(0.3, -0.1), axis_ratio=0.7, phi=60.0,
                                                                  einstein_radius=1.1)

        deflections_isothermal_core = isothermal_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert deflections_core[0] == pytest.approx(deflections_isothermal_core[0], 1e-6)
        assert deflections_core[1] == pytest.approx(deflections_isothermal_core[1], 1e-6)

    def test__ratio_via_fortran__same_ratio(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                       einstein_radius=1.3, core_radius=0.2)

        deflections = isothermal_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        ratio = deflections[0] / deflections[1]

        assert ratio == pytest.approx(-0.53649 / 0.98365, 1e-3)

    def test__value_via_fortran__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(0.5, -0.7), axis_ratio=0.7, phi=60.0,
                                                                       einstein_radius=1.3, core_radius=0.2)

        deflections = isothermal_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert deflections[0] == pytest.approx(-0.53649, 1e-3)
        assert deflections[1] == pytest.approx(0.98365, 1e-3)

    def test__value_via_fortran_2__same_value(self):
        isothermal_core = profile.CoredEllipticalIsothermalMassProfile(centre=(-0.2, 0.2), axis_ratio=0.6, phi=120.0,
                                                                       einstein_radius=0.5, core_radius=0.5)

        deflections = isothermal_core.compute_deflection_angle(coordinates=(0.1625, 0.1625))

        assert deflections[0] == pytest.approx(0.30750, 1e-3)
        assert deflections[1] == pytest.approx(0.03144, 1e-3)


class TestCombinedProfiles(object):
    def test__summation(self, circular):
        combined = profile.CombinedLightProfile(circular, circular)
        assert combined.flux_at_coordinates((0, 0)) == 2 * circular.flux_at_coordinates((0, 0))

    def test_1d_symmetry(self):
        sersic1 = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                             effective_radius=0.6, sersic_index=4.0)

        sersic2 = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                             effective_radius=0.6, sersic_index=4.0, centre=(100, 0))

        combined = profile.CombinedLightProfile(sersic1, sersic2)
        assert combined.flux_at_coordinates((0, 0)) == combined.flux_at_coordinates((100, 0))
        assert combined.flux_at_coordinates((49, 0)) == combined.flux_at_coordinates((51, 0))

    def test_2d_symmetry(self):
        sersic1 = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                             effective_radius=0.6, sersic_index=4.0)

        sersic2 = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                             effective_radius=0.6, sersic_index=4.0, centre=(100, 0))
        sersic3 = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                             effective_radius=0.6, sersic_index=4.0, centre=(0, 100))

        sersic4 = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                             effective_radius=0.6, sersic_index=4.0, centre=(100, 100))

        combined = profile.CombinedLightProfile(sersic1, sersic2, sersic3, sersic4)

        assert combined.flux_at_coordinates((49, 0)) == pytest.approx(combined.flux_at_coordinates((51, 0)), 1e-5)
        assert combined.flux_at_coordinates((0, 49)) == pytest.approx(combined.flux_at_coordinates((0, 51)), 1e-5)
        assert combined.flux_at_coordinates((100, 49)) == pytest.approx(combined.flux_at_coordinates((100, 51)), 1e-5)
        assert combined.flux_at_coordinates((49, 49)) == pytest.approx(combined.flux_at_coordinates((51, 51)), 1e-5)

    def test_combined_mass_profile(self):
        isothermal = profile.EllipticalIsothermalMassProfile(centre=(1, 1), axis_ratio=0.5, phi=45.0,
                                                             einstein_radius=1.0)

        combined = profile.CombinedMassProfile(isothermal, isothermal)

        combined_deflection_angle = combined.compute_deflection_angle((0.1, 0.1))
        isothermal_deflection_angle = isothermal.compute_deflection_angle((0.1, 0.1))

        assert combined_deflection_angle[0] == 2 * isothermal_deflection_angle[0]
        assert combined_deflection_angle[1] == 2 * isothermal_deflection_angle[1]


class TestEquivalentProfile(object):
    def test_as_sersic_profile(self, circular):
        copy = circular.as_sersic_profile()

        assert copy.centre == circular.centre
        assert copy.axis_ratio == circular.axis_ratio
        assert copy.phi == circular.phi
        assert copy.flux == circular.flux
        assert copy.sersic_index == circular.sersic_index

    def test_x_as_y(self, circular, exponential, dev_vaucouleurs, core):
        def assert_shared_base(x, y):
            assert x.centre == y.centre
            assert x.axis_ratio == y.axis_ratio
            assert x.phi == y.phi
            assert x.flux == y.flux

        assert_shared_base(circular, circular.as_exponential_profile())
        assert_shared_base(exponential, exponential.as_dev_vaucouleurs_profile())
        assert_shared_base(dev_vaucouleurs, dev_vaucouleurs.as_core_sersic_profile(1, 1, 1, 1))
        assert_shared_base(core, core.as_sersic_profile())
