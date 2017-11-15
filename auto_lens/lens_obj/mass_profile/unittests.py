from __future__ import division, print_function

import numpy as np
import pytest
import mass_profile

class TestGeom:

    def test__translate_coordinates__no_shift_in__no_shifts_out(self):

        x_new, y_new = translate_coordinates(x=0.0, y=0.0, x_cen=0.0, y_cen=0.0)

        assert x_new == 0.0
        assert y_new == 0.0

    def test__translate_coordinates__x_shift_in__x_is_shifted(self):

        x_new, y_new = translate_coordinates(x=0.0, y=0.0, x_cen=0.5, y_cen=0.0)

        assert x_new == -0.5
        assert y_new == 0.0

    def test__translate_coordinates__y_shift_in__y_is_shifted(self):

        x_new, y_new = translate_coordinates(x=0.0, y=0.0, x_cen=0.0, y_cen=0.5)

        assert x_new == 0.0
        assert y_new == -0.5

    def test__translate_coordinates__x_and_y_shift_in__x_and_y_shifted(self):

        x_new, y_new = translate_coordinates(x=0.0, y=0.0, x_cen=0.5, y_cen=0.5)

        assert x_new == -0.5
        assert y_new == -0.5

    def test__translate_coordinates__x_and_y_shift_2_in__x_and_y_shifted(self):

        x_new, y_new = translate_coordinates(x=0.2, y=-0.4, x_cen=1.0, y_cen=-0.5)

        assert x_new == -0.8
        assert y_new == pytest.approx(0.1, 1e-5)

    def test__calc_radial_distance__x_and_y_zero_in__r_out_zero(self):

        assert calc_radial_distance(x=0.0, y=0.0) == 0.0

    def test__calc_radial_distance__x_one_y_zero_in__r_out_1(self):

        assert calc_radial_distance(x=1.0, y=0.0) == 1.0

    def test__calc_radial_distance__x_one_y_one__r_out_root_2(self):

        assert calc_radial_distance(x=1.0, y=1.0) == pytest.approx(np.sqrt(2), 1e-5)

    def test__calc_radial_distance__other_x_y_values__correct_value(self):

        assert calc_radial_distance(x=-1.0, y=1.0) == pytest.approx(np.sqrt(2), 1e-5)
        assert calc_radial_distance(x=1.0, y=-1.0) == pytest.approx(np.sqrt(2), 1e-5)
        assert calc_radial_distance(x=-1.0, y=-1.0) == pytest.approx(np.sqrt(2), 1e-5)

    def test__rotate_coordinates__no_rotation_in__no_rotation_out(self):

        x_new, y_new = rotate_coordinates(x=1.0, y=0.0, phi_degrees=0.)

        assert x_new == pytest.approx(1.0, 1e-5)
        assert y_new == pytest.approx(0.0, 1e-5)

    def test__rotate_coordinates__45_deg_rotation_in__rotation_out(self):

        x_new, y_new = rotate_coordinates(x=1.0, y=0.0, phi_degrees=45.)

        assert x_new == pytest.approx(0.5, 1e-5)
        assert y_new == pytest.approx(-0.5, 1e-5)

    def test__rotate_coordinates__90_deg_rotation_in__rotation_out(self):

        x_new, y_new = rotate_coordinates(x=1.0, y=0.0, phi_degrees=90.)

        assert x_new == pytest.approx(0.0, 1e-5)
        assert y_new == pytest.approx(-1.0, 1e-5)

    def test__rotate_coordinates__180_deg_rotation_in__rotation_out(self):

        x_new, y_new = rotate_coordinates(x=1.0, y=0.0, phi_degrees=180.)

        assert x_new == pytest.approx(-1.0, 1e-5)
        assert y_new == pytest.approx(0.0, 1e-5)

#    def test__sie_defl_angle__mass_model_in__correct_defl_angles(self):
