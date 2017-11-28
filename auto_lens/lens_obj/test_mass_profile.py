from __future__ import division, print_function

import numpy as np
import pytest
import profile


# TODO: OK, so we hid the coordinate shift internally and made the EllipticalProfile Abstract Class. Note the effect
# TODO: here. The coordinate shift function doesn't have to be called a bunch of times and all of the tests correspond
# TODO: to the Abstract class rather than the concrete child. The tests look neater and that's probably a good sign.


# noinspection PyClassHasNoInit
class TestEllipticalPowerLaw:
    def test__coordinates_to_centre__mass_centre_zeros__no_shift(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == 0.0
        assert coordinates_shift[1] == 0.0

    def test__coordinates_to_centre__mass_centre_x_shift__x_shifts(self):
        power_law = profile.EllipticalProfile(x_cen=0.5, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == -0.5
        assert coordinates_shift[1] == 0.0

    def test__coordinates_to_centre__mass_centre_y_shift__y_shifts(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.5, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == 0.0
        assert coordinates_shift[1] == -0.5

    def test__coordinates_to_centre__mass_centre_x_and_y_shift__x_and_y_both_shift(self):
        power_law = profile.EllipticalProfile(x_cen=0.5, y_cen=0.5, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.0, 0.0))

        assert coordinates_shift[0] == -0.5
        assert coordinates_shift[1] == -0.5

    def test__coordinates_to_centre__mass_centre_and_coordinates__correct_shifts(self):
        power_law = profile.EllipticalProfile(x_cen=1.0, y_cen=0.5, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0.2, 0.4))

        assert coordinates_shift[0] == -0.8
        assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    def test__coordinates_to_radius__coordinates_overlap_mass_profile__r_is_zero(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0., axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(0, 0))

        assert power_law.coordinates_to_radius(coordinates_shift) == 0.0

    def test__coordinates_to_radius__x_coordinates_is_one__r_is_one(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0., axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0))

        assert power_law.coordinates_to_radius(coordinates_shift) == 1.0

    def test__coordinates_to_radius__x_and_y_coordinates_are_one__r_is_root_two(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0., axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        assert power_law.coordinates_to_radius(coordinates_shift) == pytest.approx(np.sqrt(2), 1e-5)

    def test__coordinates_to_radius__mass_profile_moves_instead__r_is_root_two(self):
        power_law = profile.EllipticalProfile(x_cen=1.0, y_cen=1.0, axis_ratio=1.0, phi=0.0)

        assert power_law.coordinates_to_radius((0.0, 0.0)) == pytest.approx(np.sqrt(2), 1e-5)

    def test__angles_from_x_axis__phi_is_zero__angles_one_and_zero(self):
        power_law = profile.EllipticalProfile(x_cen=1.0, y_cen=1.0, axis_ratio=1.0, phi=0.0)

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == 1.0
        assert sin_phi == 0.0

    def test__angles_from_x_axis__phi_is_forty_five__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(x_cen=1.0, y_cen=1.0, axis_ratio=1.0, phi=45.0)

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == pytest.approx(0.707, 1e-3)
        assert sin_phi == pytest.approx(0.707, 1e-3)

    def test__angles_from_x_axis__phi_is_sixty__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(x_cen=1.0, y_cen=1.0, axis_ratio=1.0, phi=60.0)

        cos_phi, sin_phi = power_law.angles_from_x_axis()

        assert cos_phi == pytest.approx(0.5, 1e-3)
        assert sin_phi == pytest.approx(0.866, 1e-3)

    def test__coordinates_angle_from_x__angle_is_zero__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)

        assert cos_theta == 1.0
        assert sin_theta == 0.0

    def test__coordinates_angle_from_x__angle_is_forty_five__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)

        assert cos_theta == pytest.approx(0.707, 1e-3)
        assert sin_theta == pytest.approx(0.707, 1e-3)

    def test__coordinates_angle_from_x__angle_is_sixty__angles_follow_trig(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.7320))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)

        assert cos_theta == pytest.approx(0.5, 1e-3)
        assert sin_theta == pytest.approx(0.866, 1e-3)

    def test__coordinates_angle_to_mass_profile__same_angle__no_rotation(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)
        cos_theta, sin_theta = power_law.coordinates_angle_to_mass_profile(cos_theta, sin_theta)

        assert cos_theta == 1.0
        assert sin_theta == 0.0

    def test__coordinates_angle_to_mass_profile_both_45___no_rotation(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=45.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)
        cos_theta, sin_theta = power_law.coordinates_angle_to_mass_profile(cos_theta, sin_theta)

        assert cos_theta == pytest.approx(1.0, 1e-3)
        assert sin_theta == pytest.approx(0.0, 1e-3)

    def test__coordinates_angle_to_mass_profile_45_offset_same_angle__rotation_follows_trig(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 1.0))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)
        cos_theta, sin_theta = power_law.coordinates_angle_to_mass_profile(cos_theta, sin_theta)

        assert cos_theta == pytest.approx(0.707, 1e-3)
        assert sin_theta == pytest.approx(0.707, 1e-3)

    def test__coordinates_angle_to_mass_profile_negative_60_offset_same_angle__rotation_follows_trig(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=60.0)

        coordinates_shift = power_law.coordinates_to_centre(coordinates=(1.0, 0.0))

        cos_theta, sin_theta = power_law.coordinates_angle_from_x(coordinates_shift)
        cos_theta, sin_theta = power_law.coordinates_angle_to_mass_profile(cos_theta, sin_theta)

        assert cos_theta == pytest.approx(0.5, 1e-3)
        assert sin_theta == pytest.approx(-0.866, 1e-3)

    def test__coordinates_back_to_cartesian__phi_zero__no_rotation(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=0.0)

        coordinates_elliptical = (1.0, 1.0)

        x, y = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert x == 1.0
        assert y == 1.0

    def test__coordinates_back_to_cartesian__phi_ninety__correct_calc(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=90.0)

        coordinates_elliptical = (1.0, 1.0)

        x, y = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert x == pytest.approx(-1.0, 1e-3)
        assert y == 1.0

    def test__coordinates_back_to_cartesian__phi_forty_five__correct_calc(self):
        power_law = profile.EllipticalProfile(x_cen=0.0, y_cen=0.0, axis_ratio=1.0, phi=45.0)

        coordinates_elliptical = (1.0, 1.0)

        x, y = power_law.coordinates_back_to_cartesian(coordinates_elliptical)

        assert x == pytest.approx(0.0, 1e-3)
        assert y == pytest.approx(2 ** 0.5, 1e-3)
