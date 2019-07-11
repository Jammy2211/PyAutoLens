from __future__ import division, print_function

import numpy as np
import pytest

from os import path
import autofit as af
from autolens import dimensions as dim
from autolens.model.profiles import geometry_profiles as gp

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(config_path='{}/../../test_files/config/radial_min'.format(directory))


class TestMemoize(object):

    def test_add_to_cache(self):
        class MyProfile(object):
            # noinspection PyMethodMayBeStatic
            @gp.cache
            def my_method(self, grid, grid_radial_minimum=None):
                return grid

        profile = MyProfile()
        other_profile = MyProfile()
        assert not hasattr(profile, "cache")

        profile.my_method(np.array([0]))
        assert hasattr(profile, "cache")
        assert not hasattr(other_profile, "cache")
        assert len(profile.cache) == 1

        profile.my_method(np.array([0]))
        assert len(profile.cache) == 1

        profile.my_method(np.array([1]))
        assert len(profile.cache) == 2

    def test_get_from_cache(self):
        class CountingProfile(object):
            def __init__(self):
                self.count = 0

            @gp.cache
            def my_method(self, grid, grid_radial_minimum=None):
                self.count += 1
                return self.count

        profile = CountingProfile()

        assert profile.my_method(grid=np.array([0]), grid_radial_minimum=None) == 1
        assert profile.my_method(grid=np.array([1]), grid_radial_minimum=None) == 2
        assert profile.my_method(grid=np.array([2]), grid_radial_minimum=None) == 3
        assert profile.my_method(grid=np.array([0]), grid_radial_minimum=None) == 1
        assert profile.my_method(grid=np.array([1]), grid_radial_minimum=None) == 2

    def test_multiple_cached_methods(self):
        class MultiMethodProfile(object):
            @gp.cache
            def method_one(self, grid, grid_radial_minimum=None):
                return grid

            @gp.cache
            def method_two(self, grid, grid_radial_minimum=None):
                return grid

        profile = MultiMethodProfile()

        array = np.array([0])
        profile.method_one(array)
        assert profile.method_one(array) is array
        assert profile.method_two(np.array([0])) is not array


class TestGeometryProfile(object):

    def test__constructor_and_units(self):
        
        profile = gp.GeometryProfile(centre=(1.0, 2.0))

        assert profile.centre == (1.0, 2.0)
        assert isinstance(profile.centre[0], dim.Length)
        assert isinstance(profile.centre[1], dim.Length)
        assert profile.centre[0].unit == 'arcsec'
        assert profile.centre[1].unit == 'arcsec'


class TestEllipticalProfile(object):

    class TestConstuctorUnits(object):

        def test__constructor_and_units(self):

            profile = gp.EllipticalProfile(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0)

            assert profile.centre == (1.0, 2.0)
            assert isinstance(profile.centre[0], dim.Length)
            assert isinstance(profile.centre[1], dim.Length)
            assert profile.centre[0].unit == 'arcsec'
            assert profile.centre[1].unit == 'arcsec'

            assert profile.axis_ratio == 0.5
            assert isinstance(profile.axis_ratio, float)

            assert profile.phi == 45.0
            assert isinstance(profile.phi, float)

    class TestAnglesFromXAxis(object):

        def test__profile_angle_phi_is_0__cosine_and_sin_of_phi_is_1_and_0(self):
            elliptical_profile = gp.EllipticalProfile(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == 1.0
            assert sin_phi == 0.0

        def test__profile_angle_phi_is_45__cosine_and_sin_of_phi_follow_trig__therefore_half_root_2(self):
            elliptical_profile = gp.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=45.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == pytest.approx(0.707, 1e-3)
            assert sin_phi == pytest.approx(0.707, 1e-3)

        def test__profile_angle_phi_is_60__cosine_and_sin_of_phi_follow_trig(self):
            elliptical_profile = gp.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=60.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == pytest.approx(0.5, 1e-3)
            assert sin_phi == pytest.approx(0.866, 1e-3)

        def test__profile_angle_phi_is_225__cosine_and_sin_of_phi_follow_trig__therefore_negative_half_root_2(self):
            elliptical_profile = gp.EllipticalProfile(centre=(1, 1), axis_ratio=1.0, phi=225.0)

            cos_phi, sin_phi = elliptical_profile.cos_and_sin_from_x_axis()

            assert cos_phi == pytest.approx(-0.707, 1e-3)
            assert sin_phi == pytest.approx(-0.707, 1e-3)

    class TestTransformGrid(object):

        def test__profile_angle_phi_is_0__grid_x_1_y_1__returns_same_grid_so_x_1_y_1(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(1.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test___profile_angle_phi_45__grid_y_1_x_1__rotated_counter_clockwise_to_y_0_x_root_2(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(2 ** 0.5, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(0.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_90__grid_y_1_x_1__rotated_grid_clockwise_so_y_1_x_negative_1(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(-1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(1.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_90__grid_y_0_x_1__rotated_grid_clockwise_so_y_1_x_0(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=90.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[0.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(-1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(0.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(0.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_180__grid_y_1_x_1__rotated_grid_clockwise_so_y_and_x_negative_1(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=180.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(-1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(-1.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_270__grid_y_1_x_1__rotated_grid_clockwise_so_y_negative_1_x_1(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=270.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(-1.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_360__rotated_grid_are_original_grid_x_1_y_(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=360.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(1.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__profile_angle_phi_315__grid_y_1_x_1__rotated_grid_clockwise_so_y_0_x_root_2(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=315.0)

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == pytest.approx(2 ** 0.5, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(0.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__include_profile_centre_offset__is_used_before_rotation_is_performed(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=90.0, centre=(2.0, 3.0))

            transformed_grid = elliptical_profile.transform_grid_to_reference_frame(grid=np.array([[3.0, 4.0]]))

            assert transformed_grid[0, 0] == pytest.approx(-1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(1.0, 1e-3)

            transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(transformed_grid)

            assert transformed_back_grid[0, 0] == pytest.approx(3.0, 1e-3)
            assert transformed_back_grid[0, 1] == pytest.approx(4.0, 1e-3)

        def test__random_values__grid_are_transformed_to_and_from_reference_frame__equal_to_original_values(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=45.0)

            grid_original = np.array([[5.2221, 2.6565]])

            grid_elliptical = elliptical_profile.transform_grid_to_reference_frame(grid_original)

            transformed_grid = elliptical_profile.transform_grid_from_reference_frame(grid_elliptical)

            assert transformed_grid[0, 0] == pytest.approx(grid_original[0, 0], 1e-5)
            assert transformed_grid[0, 1] == pytest.approx(grid_original[0, 1], 1e-5)

    class TestCoordinateMovements(object):

        def test__grid_and_centres_of_two_lenses_are_equivalent__grid_are_equivalent(self):
            elliptical_profile1 = gp.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(0, 0))
            grid1 = elliptical_profile1.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            elliptical_profile2 = gp.EllipticalProfile(axis_ratio=1.0, phi=0.0, centre=(-1, -1))
            grid2 = elliptical_profile2.transform_grid_to_reference_frame(grid=np.array([[0.0, 0.0]]))

            assert grid1[0, 0] == grid2[0, 0]
            assert grid1[0, 1] == grid2[0, 1]

        def test__same_as_above_but_include_angle_phi_as_55__grid_are_equivalent(self):
            elliptical_profile1 = gp.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(0, 0))
            grid1 = elliptical_profile1.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            elliptical_profile2 = gp.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            grid2 = elliptical_profile2.transform_grid_to_reference_frame(grid=np.array([[0.0, 0.0]]))

            assert grid1[0, 0] == grid2[0, 0]
            assert grid1[0, 1] == grid2[0, 1]

        def test__grid_are_again_the_same_after_centre_shift__grid_equivalent(self):
            elliptical_profile1 = gp.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(1, 1))
            grid1 = elliptical_profile1.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            elliptical_profile2 = gp.EllipticalProfile(axis_ratio=1.0, phi=55.0, centre=(-1, -1))
            grid2 = elliptical_profile2.transform_grid_to_reference_frame(grid=np.array([[-1.0, -1.0]]))

            assert grid1[0, 0] == grid2[0, 0]
            assert grid1[0, 1] == grid2[0, 1]

    class TestTransformedGridToEccentricRadius(object):

        def test__profile_axis_ratio_1__r_is_root_2__therefore_ecccentric_radius_is_elliptical_radius_is_root_2(self):
            elliptical_profile = gp.EllipticalProfile(axis_ratio=1.0, phi=0.0)

            eccentric_radius = elliptical_profile.grid_to_eccentric_radii(np.array([[1.0, 1.0]]))

            assert eccentric_radius == pytest.approx(2.0 ** 0.5, 1e-3)

        def test__same_as_above_but_axis_ratio_is_05__r_follows_elliptical_coordinate_equation(self):
            # eccentric_radius = sqrt(axis_ratio) * sqrt(  x**2 +   (y**2 / axis_ratio**2))
            # eccentric_radius =        sqrt(0.5) * sqrt(1.0**2 + (1.0**2 /        0.5**2))
            # eccentric radius =        sqrt(0.5) * sqrt( 5 ) = 1.58113

            elliptical_profile = gp.EllipticalProfile(axis_ratio=0.5, phi=0.0)

            eccentric_radius = elliptical_profile.grid_to_eccentric_radii(np.array([[1.0, 1.0]]))

            assert eccentric_radius == pytest.approx(1.58113, 1e-3)


class TestSphericalProfile(object):

    class TestConstuctorUnits(object):

        def test__constructor_and_unit_conversions(self):
            profile = gp.SphericalProfile(centre=(1.0, 2.0))

            assert profile.centre == (1.0, 2.0)
            assert isinstance(profile.centre[0], dim.Length)
            assert isinstance(profile.centre[1], dim.Length)
            assert profile.centre[0].unit == 'arcsec'
            assert profile.centre[1].unit == 'arcsec'

    class TestCoordinatesMovement(object):

        def test__profile_cenre_y_0_x_0__grid_y_1_x_1__no_coordinate_movement_so_y_1_x_1(self):
            spherical_profile = gp.SphericalProfile(centre=(0.0, 0.0))

            transformed_grid = spherical_profile.transform_grid_from_reference_frame(np.array([[1.0, 1.0]]))

            assert transformed_grid[0, 0] == 1.0
            assert transformed_grid[0, 1] == 1.0

        def test__grid_and_centres_of_two_lenses_are_equivalent__grid_are_equivalent(self):
            spherical_profile1 = gp.SphericalProfile(centre=(0, 0))

            grid1 = spherical_profile1.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            spherical_profile2 = gp.SphericalProfile(centre=(-1, -1))
            grid2 = spherical_profile2.transform_grid_to_reference_frame(grid=np.array([[0.0, 0.0]]))

            assert grid1[0, 0] == grid2[0, 0]
            assert grid1[0, 1] == grid2[0, 1]

        def test__grid_are_again_the_same_after_centre_shift__grid_equivalent(self):
            spherical_profile1 = gp.SphericalProfile(centre=(1, 1))
            grid1 = spherical_profile1.transform_grid_to_reference_frame(grid=np.array([[1.0, 1.0]]))

            spherical_profile2 = gp.SphericalProfile(centre=(-1, -1))
            grid2 = spherical_profile2.transform_grid_to_reference_frame(grid=np.array([[-1.0, -1.0]]))

            assert grid1[0, 0] == grid2[0, 0]
            assert grid1[0, 1] == grid2[0, 1]

    class TestTransformCoordinates(object):

        def test__profile_centre_y_0_x_0__grid_y_1_x_1__returns_y_1_x_1_so_same_grid(self):
            spherical_profile = gp.SphericalProfile(centre=(0.0, 0.0))

            grid = np.array([[1.0, 1.0]])

            transformed_grid = spherical_profile.transform_grid_to_reference_frame(grid)

            assert transformed_grid[0, 0] == pytest.approx(1.0, 1e-3)
            assert transformed_grid[0, 1] == pytest.approx(1.0, 1e-3)

        def test__grid_are_transformed_to_and_from_reference_frame__equal_to_original_values(self):
            spherical_profile = gp.SphericalProfile(centre=(0.0, 0.0))

            grid_original = np.array([[5.2221, 2.6565]])

            grid_spherical = spherical_profile.transform_grid_to_reference_frame(grid_original)

            transformed_grid = spherical_profile.transform_grid_from_reference_frame(grid_spherical)

            assert transformed_grid[0, 0] == pytest.approx(grid_original[0, 0], 1e-5)
            assert transformed_grid[0, 1] == pytest.approx(grid_original[0, 1], 1e-5)


class MockGridRadialMinimum(object):

    def __init__(self):
        pass

    def grid_to_grid_radii(self, grid):
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    @gp.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        return grid


# class TestGridRadialMinimum(object):
#
#     def test__mock_profile__grid_radial_minimum_is_0_or_below_radial_coordinates__no_changes(self):
#         grid = np.array([[2.5, 0.0], [4.0, 0.0], [6.0, 0.0]])
#         mock_profile = MockGridRadialMinimum()
#
#         deflections = mock_profile.deflections_from_grid(grid=grid)
#         assert (deflections == grid).all()
#
#     def test__mock_profile__grid_radial_minimum_is_above_some_radial_coordinates__moves_them_grid_radial_minimum(self):
#         grid = np.array([[2.0, 0.0], [1.0, 0.0], [6.0, 0.0]])
#         mock_profile = MockGridRadialMinimum()
#
#         deflections = mock_profile.deflections_from_grid(grid=grid)
#
#         assert (deflections == np.array([[2.5, 0.0], [2.5, 0.0], [6.0, 0.0]])).all()
#
#     def test__mock_profile__same_as_above_but_diagonal_coordinates(self):
#         grid = np.array([[np.sqrt(2.0), np.sqrt(2.0)], [1.0, np.sqrt(8.0)], [np.sqrt(8.0), np.sqrt(8.0)]])
#
#         mock_profile = MockGridRadialMinimum()
#
#         deflections = mock_profile.deflections_from_grid(grid=grid)
#
#         assert deflections == pytest.approx(np.array([[1.7677, 1.7677], [1.0, np.sqrt(8.0)],
#                                                       [np.sqrt(8), np.sqrt(8.0)]]), 1.0e-4)
