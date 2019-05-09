import math

import numpy as np
import pytest
from astropy import cosmology as cosmo

from autofit import conf
from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.model.profiles import mass_profiles as mp

from test.unit.mock.mock_cosmology import MockCosmology

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    conf.instance = conf.default


class TestPointMass(object):

    def test__constructor_and_units(self):

        point_mass = mp.PointMass(centre=(1.0, 2.0), einstein_radius=2.0)

        assert point_mass.centre == (1.0, 2.0)
        assert isinstance(point_mass.centre[0], dim.Length)
        assert isinstance(point_mass.centre[1], dim.Length)
        assert point_mass.centre[0].unit == 'arcsec'
        assert point_mass.centre[1].unit == 'arcsec'

        assert point_mass.einstein_radius == 2.0
        assert isinstance(point_mass.einstein_radius, dim.Length)
        assert point_mass.einstein_radius.unit_length == 'arcsec'

    def test__deflections__correct_values(self):
        # The radial coordinate at (1.0, 1.0) is sqrt(2)
        # This is decomposed into (y,x) angles of sin(45) = cos(45) = sqrt(2) / 2.0
        # Thus, for an EinR of 1.0, the deflection angle is (1.0 / sqrt(2)) * (sqrt(2) / 2.0)

        point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.5, 1e-3)

        point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=2.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[2.0, 2.0]]))
        assert deflections[0, 0] == pytest.approx(0.25, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.25, 1e-3)

        point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[2.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(0.4, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.2, 1e-3)

        point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=2.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[4.0, 9.0]]))
        assert deflections[0, 0] == pytest.approx(8.0 / 97.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(18.0 / 97.0, 1e-3)

        point_mass = mp.PointMass(centre=(1.0, 2.0), einstein_radius=1.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.5, 1e-3)

    def test__deflections__change_geometry(self):
        point_mass_0 = mp.PointMass(centre=(0.0, 0.0))
        point_mass_1 = mp.PointMass(centre=(1.0, 1.0))
        defls_0 = point_mass_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        defls_1 = point_mass_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert defls_0[0, 0] == pytest.approx(-defls_1[0, 0], 1e-5)
        assert defls_0[0, 1] == pytest.approx(-defls_1[0, 1], 1e-5)

        point_mass_0 = mp.PointMass(centre=(0.0, 0.0))
        point_mass_1 = mp.PointMass(centre=(0.0, 0.0))
        defls_0 = point_mass_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = point_mass_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

    def test__multiple_coordinates_in__multiple_coordinates_out(self):
        point_mass = mp.PointMass(centre=(1.0, 2.0), einstein_radius=1.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.5, 1e-3)
        assert deflections[1, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[1, 1] == pytest.approx(0.5, 1e-3)
        assert deflections[2, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[2, 1] == pytest.approx(0.5, 1e-3)

        point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections = point_mass.deflections_from_grid(grid=np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]]))
        assert deflections[0, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.5, 1e-3)

        assert deflections[1, 0] == pytest.approx(0.25, 1e-3)
        assert deflections[1, 1] == pytest.approx(0.25, 1e-3)

        assert deflections[2, 0] == pytest.approx(0.5, 1e-3)
        assert deflections[2, 1] == pytest.approx(0.5, 1e-3)

        assert deflections[3, 0] == pytest.approx(0.25, 1e-3)
        assert deflections[3, 1] == pytest.approx(0.25, 1e-3)

    def test__deflections_of_profile__dont_use_interpolate_and_cache_decorators(self):
        point_mass = mp.PointMass(centre=(-0.3, 0.2), einstein_radius=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = point_mass.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = point_mass.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()
    # def test__mass(self):
    #
    #     point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.91716)
    #
    #     print(point_mass.mass)
    #
    #     assert point_mass.mass == pytest.approx(1.3332e11, 1e9)


class TestCoredPowerLaw(object):

    def test__constructor_and_units(self):
        
        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0,
                                                     einstein_radius=1.0, slope=2.2, core_radius=0.1)

        assert cored_power_law.centre == (1.0, 2.0)
        assert isinstance(cored_power_law.centre[0], dim.Length)
        assert isinstance(cored_power_law.centre[1], dim.Length)
        assert cored_power_law.centre[0].unit == 'arcsec'
        assert cored_power_law.centre[1].unit == 'arcsec'

        assert cored_power_law.axis_ratio == 0.5
        assert isinstance(cored_power_law.axis_ratio, float)

        assert cored_power_law.phi == 45.0
        assert isinstance(cored_power_law.phi, float)

        assert cored_power_law.einstein_radius == 1.0
        assert isinstance(cored_power_law.einstein_radius, dim.Length)
        assert cored_power_law.einstein_radius.unit_length == 'arcsec'

        assert cored_power_law.slope == 2.2
        assert isinstance(cored_power_law.slope, float)

        assert cored_power_law.core_radius == 0.1
        assert isinstance(cored_power_law.core_radius, dim.Length)
        assert cored_power_law.core_radius.unit_length == 'arcsec'

        assert cored_power_law.einstein_radius_rescaled == pytest.approx(0.53333333, 1.0e-4)

        cored_power_law = mp.SphericalCoredPowerLaw(centre=(1.0, 2.0), einstein_radius=1.0, slope=2.2, core_radius=0.1)

        assert cored_power_law.centre == (1.0, 2.0)
        assert isinstance(cored_power_law.centre[0], dim.Length)
        assert isinstance(cored_power_law.centre[1], dim.Length)
        assert cored_power_law.centre[0].unit == 'arcsec'
        assert cored_power_law.centre[1].unit == 'arcsec'

        assert cored_power_law.axis_ratio == 1.0
        assert isinstance(cored_power_law.axis_ratio, float)

        assert cored_power_law.phi == 0.0
        assert isinstance(cored_power_law.phi, float)

        assert cored_power_law.einstein_radius == 1.0
        assert isinstance(cored_power_law.einstein_radius, dim.Length)
        assert cored_power_law.einstein_radius.unit_length == 'arcsec'

        assert cored_power_law.slope == 2.2
        assert isinstance(cored_power_law.slope, float)

        assert cored_power_law.core_radius == 0.1
        assert isinstance(cored_power_law.core_radius, dim.Length)
        assert cored_power_law.core_radius.unit_length == 'arcsec'

        assert cored_power_law.einstein_radius_rescaled == pytest.approx(0.4, 1.0e-4)

    def test__convergence_correct_values(self):
        cored_power_law = mp.SphericalCoredPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.2, core_radius=0.1)
        assert cored_power_law.convergence_func(radius=1.0) == pytest.approx(0.39762, 1e-4)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0,
                                                     slope=2.3, core_radius=0.2)
        assert cored_power_law.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.45492, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=2.0,
                                                     slope=1.7, core_radius=0.2)
        assert cored_power_law.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(1.3887, 1e-3)

    def test__potential_correct_values(self):
        power_law = mp.SphericalCoredPowerLaw(centre=(-0.7, 0.5), einstein_radius=1.0, slope=1.8, core_radius=0.2)
        assert power_law.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(0.54913, 1e-3)

        power_law = mp.SphericalCoredPowerLaw(centre=(0.2, -0.2), einstein_radius=0.5, slope=2.4, core_radius=0.5)
        assert power_law.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(0.01820, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.2, -0.2), axis_ratio=0.6, phi=120.0,
                                                     einstein_radius=0.5, slope=2.4, core_radius=0.5)
        assert cored_power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.02319, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                     einstein_radius=1.3, slope=1.8, core_radius=0.2)
        assert cored_power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.71185, 1e-3)

    def test__deflections__correct_values(self):
        power_law = mp.SphericalCoredPowerLaw(centre=(-0.7, 0.5), einstein_radius=1.0,
                                              slope=1.8, core_radius=0.2)
        deflections = power_law.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.80677, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.30680, 1e-3)

        power_law = mp.SphericalCoredPowerLaw(centre=(0.2, -0.2), einstein_radius=0.5,
                                              slope=2.4, core_radius=0.5)
        deflections = power_law.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(-0.00321, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.09316, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                     einstein_radius=1.3, slope=1.8, core_radius=0.2)
        defls = cored_power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.9869, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.54882, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.2, -0.2), axis_ratio=0.6, phi=120.0,
                                                     einstein_radius=0.5, slope=2.4, core_radius=0.5)

        defls = cored_power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.01111, 1e-3)
        assert defls[0, 1] == pytest.approx(0.11403, 1e-3)

    def test__surfce_density__change_geometry(self):
        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(1.0, 1.0), )
        assert cored_power_law_0.convergence_from_grid(
            grid=np.array([[1.0, 1.0]])) == cored_power_law_1.convergence_from_grid(grid=np.array([[0.0, 0.0]]))

        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        assert cored_power_law_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])) == cored_power_law_1.convergence_from_grid(grid=np.array([[0.0, 1.0]]))

        cored_power_law_0 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        cored_power_law_1 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert cored_power_law_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])) == cored_power_law_1.convergence_from_grid(grid=np.array([[0.0, 1.0]]))

    def test__potential__change_geometry(self):
        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(1.0, 1.0))
        assert cored_power_law_0.potential_from_grid(
            grid=np.array([[1.0, 1.0]])) == cored_power_law_1.potential_from_grid(grid=np.array([[0.0, 0.0]]))

        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        assert cored_power_law_0.potential_from_grid(
            grid=np.array([[1.0, 0.0]])) == cored_power_law_1.potential_from_grid(grid=np.array([[0.0, 1.0]]))

        cored_power_law_0 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        cored_power_law_1 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert cored_power_law_0.potential_from_grid(
            grid=np.array([[1.0, 0.0]])) == cored_power_law_1.potential_from_grid(grid=np.array([[0.0, 1.0]]))

    def test__deflections__change_geometry(self):
        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(1.0, 1.0))
        defls_0 = cored_power_law_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        defls_1 = cored_power_law_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert defls_0[0, 0] == pytest.approx(-defls_1[0, 0], 1e-5)
        assert defls_0[0, 1] == pytest.approx(-defls_1[0, 1], 1e-5)

        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        defls_0 = cored_power_law_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = cored_power_law_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

        cored_power_law_0 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        cored_power_law_1 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        defls_0 = cored_power_law_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = cored_power_law_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

    def test__multiple_coordinates_in__multiple_quantities_out(self):
        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0,
                                                     slope=2.3, core_radius=0.2)
        assert cored_power_law.convergence_from_grid(grid=np.array([[0.0, 1.0], [0.0, 1.0]]))[0] == pytest.approx(
            0.45492, 1e-3)
        assert cored_power_law.convergence_from_grid(grid=np.array([[0.0, 1.0], [0.0, 1.0]]))[1] == pytest.approx(
            0.45492, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.2, -0.2), axis_ratio=0.6, phi=120.0,
                                                     einstein_radius=0.5,
                                                     slope=2.4, core_radius=0.5)
        assert cored_power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))[
                   0] == pytest.approx(0.02319, 1e-3)
        assert cored_power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))[
                   1] == pytest.approx(0.02319, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                     einstein_radius=1.3, slope=1.8, core_radius=0.2)
        defls = cored_power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.9869, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.54882, 1e-3)
        assert defls[1, 0] == pytest.approx(0.9869, 1e-3)
        assert defls[1, 1] == pytest.approx(-0.54882, 1e-3)

    def test__spherical_and_elliptical_match(self):
        elliptical = mp.EllipticalCoredPowerLaw(centre=(1.1, 1.1), axis_ratio=1.0, phi=0.0, einstein_radius=3.0,
                                                slope=2.2, core_radius=0.1)
        spherical = mp.SphericalCoredPowerLaw(centre=(1.1, 1.1), einstein_radius=3.0, slope=2.2, core_radius=0.1)

        assert elliptical.convergence_from_grid(grid) == pytest.approx(spherical.convergence_from_grid(grid),
                                                                       1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                     einstein_radius=1.3, slope=1.8, core_radius=0.2)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)
        true_deflections = cored_power_law.deflections_from_grid(grid=regular)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = cored_power_law.deflections_from_grid(grid=regular_with_interp)
        assert np.max(true_deflections[:, 0] - interp_deflections[:, 0]) < 0.1
        assert np.max(true_deflections[:, 1] - interp_deflections[:, 1]) < 0.1

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = cored_power_law.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(self):
        cored_power_law = mp.SphericalCoredPowerLaw(centre=(-0.7, 0.5), einstein_radius=1.3, slope=1.8, core_radius=0.2)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)
        true_deflections = cored_power_law.deflections_from_grid(grid=regular)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = cored_power_law.deflections_from_grid(grid=regular_with_interp)
        assert np.max(true_deflections[:, 0] - interp_deflections[:, 0]) < 0.1
        assert np.max(true_deflections[:, 1] - interp_deflections[:, 1]) < 0.1

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = cored_power_law.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()

    def test__summarize_in_units(self):

        cored_power_law = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.0, slope=2.0)

        summary_text = cored_power_law.summarize_in_units(radii=[dim.Length(10.0), dim.Length(500.0)], prefix='pl_',
                                                          unit_length='arcsec', unit_mass='angular', whitespace=50)

        index = 0

        assert summary_text[index] == 'Mass Profile = SphericalCoredPowerLaw' ; index += 1
        assert summary_text[index] ==  '' ; index += 1
        assert summary_text[index] == 'pl_einstein_radius                                1.00 arcsec' ; index += 1
        assert summary_text[index] == 'pl_einstein_mass                                  3.1416e+00 angular' ; index += 1
        assert summary_text[index] == 'pl_mass_within_10.00_arcsec                       3.1416e+01 angular' ; index += 1
        assert summary_text[index] == 'pl_mass_within_500.00_arcsec                      1.5708e+03 angular' ; index += 1


class TestPowerLaw(object):

    def test__constructor_and_units(self):

        power_law = mp.EllipticalPowerLaw(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, einstein_radius=1.0,
                                          slope=2.0)

        assert power_law.centre == (1.0, 2.0)
        assert isinstance(power_law.centre[0], dim.Length)
        assert isinstance(power_law.centre[1], dim.Length)
        assert power_law.centre[0].unit == 'arcsec'
        assert power_law.centre[1].unit == 'arcsec'

        assert power_law.axis_ratio == 0.5
        assert isinstance(power_law.axis_ratio, float)

        assert power_law.phi == 45.0
        assert isinstance(power_law.phi, float)

        assert power_law.einstein_radius == 1.0
        assert isinstance(power_law.einstein_radius, dim.Length)
        assert power_law.einstein_radius.unit_length == 'arcsec'

        assert power_law.slope == 2.0
        assert isinstance(power_law.slope, float)

        assert power_law.core_radius == 0.0
        assert isinstance(power_law.core_radius, dim.Length)
        assert power_law.core_radius.unit_length == 'arcsec'

        assert power_law.einstein_radius_rescaled == pytest.approx(0.6666666666, 1.0e-4)

        power_law = mp.SphericalPowerLaw(centre=(1.0, 2.0), einstein_radius=1.0, slope=2.0)

        assert power_law.centre == (1.0, 2.0)
        assert isinstance(power_law.centre[0], dim.Length)
        assert isinstance(power_law.centre[1], dim.Length)
        assert power_law.centre[0].unit == 'arcsec'
        assert power_law.centre[1].unit == 'arcsec'

        assert power_law.axis_ratio == 1.0
        assert isinstance(power_law.axis_ratio, float)

        assert power_law.phi == 0.0
        assert isinstance(power_law.phi, float)

        assert power_law.einstein_radius == 1.0
        assert isinstance(power_law.einstein_radius, dim.Length)
        assert power_law.einstein_radius.unit_length == 'arcsec'

        assert power_law.slope == 2.0
        assert isinstance(power_law.slope, float)

        assert power_law.core_radius == 0.0
        assert isinstance(power_law.core_radius, dim.Length)
        assert power_law.core_radius.unit_length == 'arcsec'

        assert power_law.einstein_radius_rescaled == pytest.approx(0.5, 1.0e-4)

    def test__convergence_correct_values(self):
        isothermal = mp.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0)
        assert isothermal.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(0.5, 1e-3)

        isothermal = mp.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.2)
        assert isothermal.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.4, 1e-3)

        power_law = mp.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.2)
        assert power_law.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.4, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                          einstein_radius=1.0, slope=2.3)
        assert power_law.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.466666, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                          einstein_radius=2.0, slope=1.7)
        assert power_law.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(1.4079, 1e-3)

    def test__potential_correct_values(self):
        power_law = mp.SphericalPowerLaw(centre=(-0.7, 0.5), einstein_radius=1.3, slope=2.3)
        assert power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(1.90421, 1e-3)

        power_law = mp.SphericalPowerLaw(centre=(-0.7, 0.5), einstein_radius=1.3, slope=1.8)
        assert power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.93758, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0, einstein_radius=1.3,
                                          slope=2.2)
        assert power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(1.53341, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0, einstein_radius=1.3,
                                          slope=1.8)
        assert power_law.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.96723, 1e-3)

    def test__deflections__correct_values(self):
        power_law = mp.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0, slope=2.0)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-0.31622, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.94868, 1e-3)

        power_law = mp.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0, slope=2.5)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-1.59054, 1e-3)
        assert defls[0, 1] == pytest.approx(-4.77162, 1e-3)

        power_law = mp.SphericalPowerLaw(centre=(0.2, 0.2), einstein_radius=1.0, slope=1.5)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-0.06287, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.18861, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0, slope=2.5)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(1.29641, 1e-3)
        assert defls[0, 1] == pytest.approx(0.99629, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0, slope=1.5)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.48036, 1e-3)
        assert defls[0, 1] == pytest.approx(0.26729, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0, einstein_radius=1.3,
                                          slope=1.9)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(1.12841, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.60205, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=150.0, einstein_radius=1.3,
                                          slope=2.2)
        defls = power_law.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(1.25995, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.35096, 1e-3)

    def test__compare_to_cored_power_law(self):
        power_law = mp.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0, einstein_radius=1.0, slope=2.3)
        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                     einstein_radius=1.0, slope=2.3, core_radius=0.0)

        assert power_law.potential_from_grid(grid) == pytest.approx(cored_power_law.potential_from_grid(grid), 1e-3)
        assert power_law.potential_from_grid(grid) == pytest.approx(cored_power_law.potential_from_grid(grid), 1e-3)
        assert power_law.deflections_from_grid(grid) == pytest.approx(cored_power_law.deflections_from_grid(grid), 1e-3)
        assert power_law.deflections_from_grid(grid) == pytest.approx(cored_power_law.deflections_from_grid(grid), 1e-3)

    def test__spherical_and_elliptical_match(self):
        elliptical = mp.EllipticalPowerLaw(centre=(1.1, 1.1), axis_ratio=0.9999, phi=0.0, einstein_radius=3.0,
                                           slope=2.4)
        spherical = mp.SphericalPowerLaw(centre=(1.1, 1.1), einstein_radius=3.0, slope=2.4)

        assert elliptical.convergence_from_grid(grid) == pytest.approx(spherical.convergence_from_grid(grid),
                                                                       1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        power_law = mp.EllipticalPowerLaw(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0, einstein_radius=1.3, slope=1.8)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = power_law.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = power_law.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(self):
        power_law = mp.SphericalPowerLaw(centre=(-0.7, 0.5), einstein_radius=1.3, slope=1.8)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = power_law.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = power_law.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()


class TestCoredIsothermal(object):

    def test__constructor_and_units(self):

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0,
                                                        einstein_radius=1.0, core_radius=0.1)

        assert cored_isothermal.centre == (1.0, 2.0)
        assert isinstance(cored_isothermal.centre[0], dim.Length)
        assert isinstance(cored_isothermal.centre[1], dim.Length)
        assert cored_isothermal.centre[0].unit == 'arcsec'
        assert cored_isothermal.centre[1].unit == 'arcsec'

        assert cored_isothermal.axis_ratio == 0.5
        assert isinstance(cored_isothermal.axis_ratio, float)

        assert cored_isothermal.phi == 45.0
        assert isinstance(cored_isothermal.phi, float)

        assert cored_isothermal.einstein_radius == 1.0
        assert isinstance(cored_isothermal.einstein_radius, dim.Length)
        assert cored_isothermal.einstein_radius.unit_length == 'arcsec'

        assert cored_isothermal.slope == 2.0
        assert isinstance(cored_isothermal.slope, float)

        assert cored_isothermal.core_radius == 0.1
        assert isinstance(cored_isothermal.core_radius, dim.Length)
        assert cored_isothermal.core_radius.unit_length == 'arcsec'

        assert cored_isothermal.einstein_radius_rescaled == pytest.approx(0.6666666666, 1.0e-4)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(1.0, 2.0), einstein_radius=1.0, core_radius=0.1)

        assert cored_isothermal.centre == (1.0, 2.0)
        assert isinstance(cored_isothermal.centre[0], dim.Length)
        assert isinstance(cored_isothermal.centre[1], dim.Length)
        assert cored_isothermal.centre[0].unit == 'arcsec'
        assert cored_isothermal.centre[1].unit == 'arcsec'

        assert cored_isothermal.axis_ratio == 1.0
        assert isinstance(cored_isothermal.axis_ratio, float)

        assert cored_isothermal.phi == 0.0
        assert isinstance(cored_isothermal.phi, float)

        assert cored_isothermal.einstein_radius == 1.0
        assert isinstance(cored_isothermal.einstein_radius, dim.Length)
        assert cored_isothermal.einstein_radius.unit_length == 'arcsec'

        assert cored_isothermal.slope == 2.0
        assert isinstance(cored_isothermal.slope, float)

        assert cored_isothermal.core_radius == 0.1
        assert isinstance(cored_isothermal.core_radius, dim.Length)
        assert cored_isothermal.core_radius.unit_length == 'arcsec'

        assert cored_isothermal.einstein_radius_rescaled == pytest.approx(0.5, 1.0e-4)

    def test__convergence_correct_values(self):
        cored_isothermal = mp.SphericalCoredIsothermal(centre=(1, 1), einstein_radius=1., core_radius=0.1)
        assert cored_isothermal.convergence_func(radius=1.0) == pytest.approx(0.49752, 1e-4)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(1, 1), einstein_radius=1.0, core_radius=0.1)
        assert cored_isothermal.convergence_func(radius=1.0) == pytest.approx(0.49752, 1e-4)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.2)
        assert cored_isothermal.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(0.49029, 1e-3)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(0.0, 0.0), einstein_radius=2.0, core_radius=0.2)
        assert cored_isothermal.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(2.0 * 0.49029,
                                                                                                    1e-3)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.2)
        assert cored_isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.49029, 1e-3)

        # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.
        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                        einstein_radius=1.0, core_radius=0.2)
        assert cored_isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
            0.49029 * 1.33333, 1e-3)

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                        einstein_radius=2.0, core_radius=0.2)
        assert cored_isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 0.49029,
                                                                                                    1e-3)

        # for axis_ratio = 1.0, the factor is 1/2
        # for axis_ratio = 0.5, the factor is 1/(1.5)
        # So the change in the value is 0.5 / (1/1.5) = 1.0 / 0.75
        # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.
        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0,
                                                        core_radius=0.2)
        assert cored_isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
            (1.0 / 0.75) * 0.49029, 1e-3)

    def test__potential__correct_values(self):
        isothermal_core = mp.SphericalCoredIsothermal(centre=(-0.7, 0.5), einstein_radius=1.3, core_radius=0.2)
        assert isothermal_core.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(0.72231, 1e-3)

        isothermal_core = mp.SphericalCoredIsothermal(centre=(0.2, -0.2), einstein_radius=0.5, core_radius=0.5)
        assert isothermal_core.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(0.03103, 1e-3)

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                        einstein_radius=1.3, core_radius=0.2)
        assert cored_isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.74354, 1e-3)

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.2, -0.2), axis_ratio=0.6, phi=120.0,
                                                        einstein_radius=0.5, core_radius=0.5)
        assert cored_isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.04024, 1e-3)

    def test__deflections__correct_values(self):
        isothermal_core = mp.SphericalCoredIsothermal(centre=(-0.7, 0.5), einstein_radius=1.3, core_radius=0.2)
        deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.98582, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.37489, 1e-3)

        isothermal_core = mp.SphericalCoredIsothermal(centre=(0.2, -0.2), einstein_radius=0.5, core_radius=0.5)
        deflections = isothermal_core.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(-0.00559, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.16216, 1e-3)

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                        einstein_radius=1.3, core_radius=0.2)
        defls = cored_isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.95429, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.52047, 1e-3)

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.2, - 0.2), axis_ratio=0.6, phi=120.0,
                                                        einstein_radius=0.5, core_radius=0.5)
        defls = cored_isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.02097, 1e-3)
        assert defls[0, 1] == pytest.approx(0.20500, 1e-3)

    def test__compare_to_cored_power_law(self):
        power_law = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0, einstein_radius=1.0,
                                                 core_radius=0.1)
        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                     einstein_radius=1.0, slope=2.0, core_radius=0.1)

        assert power_law.potential_from_grid(grid) == pytest.approx(cored_power_law.potential_from_grid(grid), 1e-3)
        assert power_law.potential_from_grid(grid) == pytest.approx(cored_power_law.potential_from_grid(grid), 1e-3)
        assert power_law.deflections_from_grid(grid) == pytest.approx(cored_power_law.deflections_from_grid(grid), 1e-3)
        assert power_law.deflections_from_grid(grid) == pytest.approx(cored_power_law.deflections_from_grid(grid), 1e-3)

    def test__spherical_and_elliptical_match(self):
        elliptical = mp.EllipticalCoredIsothermal(centre=(1.1, 1.1), axis_ratio=0.9999, phi=0.0, einstein_radius=3.0,
                                                  core_radius=1.0)
        spherical = mp.SphericalCoredIsothermal(centre=(1.1, 1.1), einstein_radius=3.0, core_radius=1.0)

        assert elliptical.convergence_from_grid(grid) == pytest.approx(spherical.convergence_from_grid(grid),
                                                                       1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0,
                                                        einstein_radius=1.3, core_radius=0.2)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = cored_isothermal.deflections_from_grid(grid=regular_with_interp)
        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = cored_isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(self):
        cored_isothermal = mp.SphericalCoredIsothermal(centre=(-0.7, 0.5), einstein_radius=1.3, core_radius=0.2)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = cored_isothermal.deflections_from_grid(grid=regular_with_interp)
        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = cored_isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()


class TestIsothermal(object):

    def test__constructor_and_units(self):

        isothermal = mp.EllipticalIsothermal(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, einstein_radius=1.0)

        assert isothermal.centre == (1.0, 2.0)
        assert isinstance(isothermal.centre[0], dim.Length)
        assert isinstance(isothermal.centre[1], dim.Length)
        assert isothermal.centre[0].unit == 'arcsec'
        assert isothermal.centre[1].unit == 'arcsec'

        assert isothermal.axis_ratio == 0.5
        assert isinstance(isothermal.axis_ratio, float)

        assert isothermal.phi == 45.0
        assert isinstance(isothermal.phi, float)

        assert isothermal.einstein_radius == 1.0
        assert isinstance(isothermal.einstein_radius, dim.Length)
        assert isothermal.einstein_radius.unit_length == 'arcsec'

        assert isothermal.slope == 2.0
        assert isinstance(isothermal.slope, float)

        assert isothermal.core_radius == 0.0
        assert isinstance(isothermal.core_radius, dim.Length)
        assert isothermal.core_radius.unit_length == 'arcsec'

        assert isothermal.einstein_radius_rescaled == pytest.approx(0.6666666666, 1.0e-4)

        isothermal = mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0)

        assert isothermal.centre == (1.0, 2.0)
        assert isinstance(isothermal.centre[0], dim.Length)
        assert isinstance(isothermal.centre[1], dim.Length)
        assert isothermal.centre[0].unit == 'arcsec'
        assert isothermal.centre[1].unit == 'arcsec'

        assert isothermal.axis_ratio == 1.0
        assert isinstance(isothermal.axis_ratio, float)

        assert isothermal.phi == 0.0
        assert isinstance(isothermal.phi, float)

        assert isothermal.einstein_radius == 1.0
        assert isinstance(isothermal.einstein_radius, dim.Length)
        assert isothermal.einstein_radius.unit_length == 'arcsec'

        assert isothermal.slope == 2.0
        assert isinstance(isothermal.slope, float)

        assert isothermal.core_radius == 0.0
        assert isinstance(isothermal.core_radius, dim.Length)
        assert isothermal.core_radius.unit_length == 'arcsec'

        assert isothermal.einstein_radius_rescaled == pytest.approx(0.5, 1.0e-4)

    def test__convergence__correct_values(self):
        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0
        isothermal = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)
        assert isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.5 * 2.0, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0)
        assert isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.5, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=2.0)
        assert isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.5 * 2.0, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0)
        assert isothermal.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.66666, 1e-3)

    def test__potential__correct_values(self):
        isothermal = mp.SphericalIsothermal(centre=(-0.7, 0.5), einstein_radius=1.3)
        assert isothermal.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(1.23435, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0, einstein_radius=1.3)
        assert isothermal.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(1.19268, 1e-3)

    def test__deflections__correct_values(self):
        isothermal = mp.SphericalIsothermal(centre=(-0.7, 0.5), einstein_radius=1.3)
        deflections = isothermal.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(1.21510, 1e-4)
        assert deflections[0, 1] == pytest.approx(-0.46208, 1e-4)

        isothermal = mp.SphericalIsothermal(centre=(-0.1, 0.1), einstein_radius=5.0)
        deflections = isothermal.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(4.88588, 1e-4)
        assert deflections[0, 1] == pytest.approx(1.06214, 1e-4)

        isothermal = mp.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0)
        defls = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.79421, 1e-3)
        assert defls[0, 1] == pytest.approx(0.50734, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0, 0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0)
        defls = isothermal.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.79421, 1e-3)
        assert defls[0, 1] == pytest.approx(0.50734, 1e-3)

    def test__compare_to_cored_power_law(self):
        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                             einstein_radius=1.0)
        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=45.0,
                                                     einstein_radius=1.0, core_radius=0.0)

        assert isothermal.potential_from_grid(grid) == pytest.approx(cored_power_law.potential_from_grid(grid), 1e-3)
        assert isothermal.potential_from_grid(grid) == pytest.approx(cored_power_law.potential_from_grid(grid), 1e-3)
        assert isothermal.deflections_from_grid(grid) == pytest.approx(cored_power_law.deflections_from_grid(grid),
                                                                       1e-3)
        assert isothermal.deflections_from_grid(grid) == pytest.approx(cored_power_law.deflections_from_grid(grid),
                                                                       1e-3)

    def test__spherical_and_elliptical_match(self):
        elliptical = mp.EllipticalIsothermal(centre=(1.1, 1.1), axis_ratio=0.9999, phi=0.0, einstein_radius=3.0)
        spherical = mp.SphericalIsothermal(centre=(1.1, 1.1), einstein_radius=3.0)

        assert elliptical.convergence_from_grid(grid) == pytest.approx(spherical.convergence_from_grid(grid),
                                                                       1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)

    def test__radius_of_critical_curve(self):

        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)
        assert sis.average_convergence_of_1_radius_in_units(unit_length='arcsec') == pytest.approx(2.0, 1e-4)

        sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.8, phi=0.0)
        assert sie.average_convergence_of_1_radius_in_units(unit_length='arcsec') == pytest.approx(1.0, 1e-4)

        sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=3.0, axis_ratio=0.5, phi=0.0)
        assert sie.average_convergence_of_1_radius_in_units(unit_length='arcsec') == pytest.approx(3.0, 1e-4)

        sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=8.0, axis_ratio=0.2, phi=0.0)
        assert sie.average_convergence_of_1_radius_in_units(unit_length='arcsec') == pytest.approx(8.0, 1e-4)

    def test__deflections_of_elliptical_profile__dont_use_interpolate_and_cache_decorators(self):
        isothermal = mp.EllipticalIsothermal(centre=(-0.7, 0.5), axis_ratio=0.7, phi=60.0, einstein_radius=1.3)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = isothermal.deflections_from_grid(grid=regular_with_interp)
        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(self):
        isothermal = mp.SphericalIsothermal(centre=(-0.7, 0.5), einstein_radius=1.3)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = isothermal.deflections_from_grid(grid=regular_with_interp)
        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()


class TestAbstractNFW(object):

    def test__coord_function_f__correct_values(self):
        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        # r > 1

        assert truncated_nfw.coord_func_f(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([0.604599, 0.435209]), 1.0e-4)

        # r < 1
        assert truncated_nfw.coord_func_f(grid_radius=np.array([0.5, 1.0 / 3.0])) == \
               pytest.approx(1.52069, 1.86967, 1.0e-4)
        #
        # r == 1
        assert (truncated_nfw.coord_func_f(grid_radius=np.array([1.0, 1.0])) == np.array([1.0, 1.0])).all()

    def test__coord_function_g__correct_values(self):
        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        # r > 1

        assert truncated_nfw.coord_func_g(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([0.13180, 0.070598]), 1.0e-4)

        # r < 1

        assert truncated_nfw.coord_func_g(grid_radius=np.array([0.5, 1.0 / 3.0])) == \
               pytest.approx(np.array([0.69425, 0.97838]), 1.0e-4)

        # r == 1
        assert (truncated_nfw.coord_func_g(grid_radius=np.array([1.0, 1.0])) == np.array([1.0 / 3.0, 1.0 / 3.0])).all()

    def test__coord_function_h__correct_values(self):
        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        assert truncated_nfw.coord_func_h(grid_radius=np.array([0.5, 3.0])) == \
               pytest.approx(np.array([0.134395, 0.840674]), 1.0e-4)

    def test__coord_function_k__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.coord_func_k(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([-0.09983408, -0.06661738]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=4.0)

        assert truncated_nfw.coord_func_k(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([-0.19869011, -0.1329414]), 1.0e-4)

    def test__coord_function_l__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.coord_func_l(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([0.00080191, 0.00080191]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        assert truncated_nfw.coord_func_l(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([.00178711, 0.00178711]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)
        assert truncated_nfw.coord_func_l(grid_radius=np.array([3.0, 3.0])) == \
               pytest.approx(np.array([0.00044044, 0.00044044]), 1.0e-4)

    def test__coord_function_m__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.coord_func_m(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([0.0398826, 0.0398826]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        assert truncated_nfw.coord_func_m(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([0.06726646, 0.06726646]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        assert truncated_nfw.coord_func_m(grid_radius=np.array([3.0, 3.0])) == \
               pytest.approx(np.array([0.06946888, 0.06946888]), 1.0e-4)

    def test__rho_at_scale_radius__numerical_values_in_default_units(self):
        cosmology = MockCosmology(kpc_per_arcsec=2.0, critical_surface_density=2.0)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        rho = nfw.rho_at_scale_radius_for_units(redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(1.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
        rho = nfw.rho_at_scale_radius_for_units(redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(3.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
        rho = nfw.rho_at_scale_radius_for_units(redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(0.25, 1e-3)

    def test__rho_at_scale_radius__unit_conversions(self):
        cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0)

        nfw_arcsec = mp.SphericalNFW(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                     kappa_s=1.0,
                                     scale_radius=dim.Length(1.0, 'arcsec'))

        rho = nfw_arcsec.rho_at_scale_radius_for_units(unit_length='arcsec', unit_mass='solMass', redshift_profile=0.5,
                                                       redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(2.0, 1e-3)

        # When converting to kpc, the critical surface density is divided by kpc_per_arcsec**2.0 = 2.0**2.0
        # The scale radius also becomes scale_radius*kpc_per_arcsec = 2.0

        rho = nfw_arcsec.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.5,
                                                       redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(0.5 / 2.0, 1e-3)

        rho = nfw_arcsec.rho_at_scale_radius_for_units(unit_length='arcsec', unit_mass='angular', redshift_profile=0.5,
                                                       redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(1.0, 1e-3)

        # This will make the critical surface density 4.0, with the same conversions as above

        cosmology = MockCosmology(arcsec_per_kpc=0.25, kpc_per_arcsec=4.0, critical_surface_density=2.0)

        rho = nfw_arcsec.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.5,
                                                       redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(0.5 / 4.0, 1e-3)

        cosmology = MockCosmology(arcsec_per_kpc=0.25, kpc_per_arcsec=4.0, critical_surface_density=4.0)

        rho = nfw_arcsec.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.5,
                                                       redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(0.25 / 4.0, 1e-3)

        nfw_kpc = mp.SphericalNFW(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                  kappa_s=1.0,
                                  scale_radius=dim.Length(1.0, 'kpc'))

        cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0)
        rho = nfw_kpc.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.5,
                                                    redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(0.5, 1e-3)

        rho = nfw_kpc.rho_at_scale_radius_for_units(unit_length='arcsec', unit_mass='solMass', redshift_profile=0.5,
                                                    redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(2.0 / 0.5, 1e-3)

        rho = nfw_kpc.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='angular', redshift_profile=0.5,
                                                    redshift_source=1.0, cosmology=cosmology)
        assert rho == pytest.approx(1.0, 1e-3)

    def test__delta_concentration_value_in_default_units(self):
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        cosmology = MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=1.0,
                                  cosmic_average_density=1.0)

        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.5, redshift_source=1.0, unit_mass='solMass',
                                                                cosmology=cosmology)
        assert delta_concentration == pytest.approx(1.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.5, redshift_source=1.0, unit_mass='solMass',
                                                                cosmology=cosmology)
        assert delta_concentration == pytest.approx(3.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.5, redshift_source=1.0, unit_mass='solMass',
                                                                cosmology=cosmology)
        assert delta_concentration == pytest.approx(0.25, 1e-3)

        # cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
        #                           cosmic_average_density=1.0)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc', unit_mass='solMass', redshift_lens=0.5,
        #                                               redshift_source=1.0, cosmology=cosmology)
        # assert delta_concentration == pytest.approx(0.5, 1e-3)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc', unit_mass='solMass', redshift_lens=0.5, redshift_source=1.0,
        #                                               cosmology=cosmology)
        # assert delta_concentration == pytest.approx(1.0, 1e-3)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=20.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc',  unit_mass='solMass', redshift_lens=0.5, redshift_source=1.0,
        #                                               cosmology=cosmology)
        # assert delta_concentration == pytest.approx(0.05, 1e-3)
        #
        # cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
        #                           cosmic_average_density=2.0)
        #
        # nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        # delta_concentration = nfw.delta_concentration(unit_length='kpc',  unit_mass='solMass', redshift_lens=0.5, redshift_source=1.0,
        #                                               cosmology=cosmology)
        # assert delta_concentration == pytest.approx(0.25, 1e-3)

    def test__solve_concentration(self):
        cosmology = MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=1.0,
                                  cosmic_average_density=1.0)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        concentration = nfw.concentration_for_units(redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

        assert concentration == pytest.approx(0.0074263, 1.0e-4)

    def test__radius_at_200__different_length_units_include_conversions(self):
        nfw_arcsec = mp.SphericalNFW(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                     kappa_s=1.0,
                                     scale_radius=dim.Length(1.0, 'arcsec'))

        cosmology = MockCosmology(arcsec_per_kpc=0.2, kpc_per_arcsec=5.0)

        concentration = nfw_arcsec.concentration_for_units(cosmology=cosmology, redshift_profile=0.5, redshift_source=1.0)

        radius_200 = nfw_arcsec.radius_at_200_for_units(unit_length='arcsec', redshift_profile=0.5, redshift_source=1.0,
                                                        cosmology=cosmology)

        assert radius_200 == concentration * 1.0

        cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        concentration = nfw_arcsec.concentration_for_units(unit_length='kpc', cosmology=cosmology, redshift_profile=0.5,
                                                           redshift_source=1.0)

        radius_200 = nfw_arcsec.radius_at_200_for_units(unit_length='kpc', redshift_profile=0.5, redshift_source=1.0,
                                                        cosmology=cosmology)

        assert radius_200 == 2.0 * concentration * 1.0

        nfw_kpc = mp.SphericalNFW(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                  kappa_s=1.0,
                                  scale_radius=dim.Length(1.0, 'kpc'))

        cosmology = MockCosmology(arcsec_per_kpc=0.2, kpc_per_arcsec=5.0)

        concentration = nfw_kpc.concentration_for_units(unit_length='kpc', redshift_profile=0.5, redshift_source=1.0,
                                                        cosmology=cosmology)

        radius_200 = nfw_kpc.radius_at_200_for_units(unit_length='kpc', redshift_profile=0.5, redshift_source=1.0,
                                                     cosmology=cosmology)

        assert radius_200 == concentration * 1.0

        cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        concentration = nfw_kpc.concentration_for_units(unit_length='arcsec', redshift_profile=0.5, redshift_source=1.0,
                                                        cosmology=cosmology)

        radius_200 = nfw_kpc.radius_at_200_for_units(unit_length='arcsec', redshift_profile=0.5, redshift_source=1.0,
                                                     cosmology=cosmology)

        assert radius_200 == concentration * 1.0 / 2.0

    def test__mass_at_200__unit_conversions_work(self):
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        cosmology = MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=1.0,
                                  cosmic_average_density=1.0)

        radius_at_200 = nfw.radius_at_200_for_units(unit_length='arcsec', redshift_profile=0.5, redshift_source=1.0,
                                                    cosmology=cosmology)

        mass_at_200 = nfw.mass_at_200_for_units(cosmology=cosmology, redshift_profile=0.5, redshift_source=1.0,
                                                unit_length='arcsec',
                                                unit_mass='solMass')

        mass_calc = 200.0 * ((4.0 / 3.0) * np.pi) * cosmology.cosmic_average_density * (radius_at_200 ** 3.0)
        assert mass_at_200 == pytest.approx(mass_calc, 1.0e-5)

        # cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
        #                           cosmic_average_density=1.0)
        #
        # radius_at_200 = nfw.radius_at_200_for_units(unit_length='arcsec', redshift_lens=0.5, redshift_source=1.0,
        #                                             cosmology=cosmology)
        #
        # mass_at_200 = nfw.mass_at_200(cosmology=cosmology, redshift_lens=0.5, redshift_source=1.0, unit_length='arcsec',
        #                               unit_mass='solMass')
        #
        # mass_calc = 200.0 * ((4.0 / 3.0) * np.pi) * cosmology.cosmic_average_density * (radius_at_200 ** 3.0)
        # assert mass_at_200 == pytest.approx(mass_calc, 1.0e-5)

    def test__values_of_quantities_for_real_cosmology__include_unit_conversions(self):

        cosmology = cosmo.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)

        nfw = mp.SphericalTruncatedNFW(kappa_s=0.5, scale_radius=5.0, truncation_radius=10.0)

        rho = nfw.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, cosmology=cosmology)

        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='kpc',
                                                                unit_mass='solMass', redshift_of_cosmic_average_density='local',
                                                                cosmology=cosmology)

        concentration = nfw.concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='kpc',
                                                    unit_mass='solMass', redshift_of_cosmic_average_density='local',
                                                    cosmology=cosmology)

        radius_at_200 = nfw.radius_at_200_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='local',
                                                    cosmology=cosmology)

        mass_at_200 = nfw.mass_at_200_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                redshift_source=2.5, redshift_of_cosmic_average_density='local',
                                                cosmology=cosmology)

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='local',
                                                    cosmology=cosmology)

        assert rho == pytest.approx(29027857.01622403, 1.0e-4)
        assert delta_concentration == pytest.approx(213451.19421263796, 1.0e-4)
        assert concentration == pytest.approx(18.6605624462417, 1.0e-4)
        assert radius_at_200 == pytest.approx(623.7751567997697, 1.0e-4)
        assert mass_at_200 == pytest.approx(27651532986258.375, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(14877085957074.299, 1.0e-4)
        

        rho = nfw.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, cosmology=cosmology)

        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='kpc',
                                                                unit_mass='solMass', redshift_of_cosmic_average_density='profile',
                                                                cosmology=cosmology)

        concentration = nfw.concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='kpc',
                                                    unit_mass='solMass', redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        radius_at_200 = nfw.radius_at_200_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        mass_at_200 = nfw.mass_at_200_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                cosmology=cosmology)

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        assert rho == pytest.approx(29027857.01622403, 1.0e-4)
        assert delta_concentration == pytest.approx(110665.28111397651, 1.0e-4)
        assert concentration == pytest.approx(14.401574489517804, 1.0e-4)
        assert radius_at_200 == pytest.approx(481.40801817963467, 1.0e-4)
        assert mass_at_200 == pytest.approx(24516707575366.09, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(13190486262169.797, 1.0e-4)

        nfw = mp.SphericalTruncatedNFW(kappa_s=0.5, scale_radius=dim.Length(3.0, 'kpc'),
                                      truncation_radius=dim.Length(7.0, 'kpc'))

        rho = nfw.rho_at_scale_radius_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, cosmology=cosmology)

        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='kpc',
                                                                unit_mass='solMass', redshift_of_cosmic_average_density='profile',
                                                                cosmology=cosmology)

        concentration = nfw.concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='kpc',
                                                    unit_mass='solMass', redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        radius_at_200 = nfw.radius_at_200_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        mass_at_200 = nfw.mass_at_200_for_units(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                cosmology=cosmology)

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(unit_length='kpc', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        assert rho == pytest.approx(323442484.90222085, 1.0e-4)
        assert delta_concentration == pytest.approx(1233086.3244882922, 1.0e-4)
        assert concentration == pytest.approx(36.61521013005619, 1.0e-4)
        assert radius_at_200 == pytest.approx(109.84563039016857, 1.0e-4)
        assert mass_at_200 == pytest.approx(291253092446.923, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(177609204745.61484, 1.0e-4)
        
        rho = nfw.rho_at_scale_radius_for_units(unit_length='arcsec', unit_mass='solMass', redshift_profile=0.6,
                                                redshift_source=2.5, cosmology=cosmology)

        delta_concentration = nfw.delta_concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='arcsec',
                                                                unit_mass='solMass', redshift_of_cosmic_average_density='profile',
                                                                cosmology=cosmology)

        concentration = nfw.concentration_for_units(redshift_profile=0.6, redshift_source=2.5, unit_length='arcsec',
                                                    unit_mass='solMass', redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        radius_at_200 = nfw.radius_at_200_for_units(unit_length='arcsec', unit_mass='solMass', redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        mass_at_200 = nfw.mass_at_200_for_units(unit_length='arcsec', unit_mass='solMass', redshift_profile=0.6,
                                                redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                cosmology=cosmology)

        mass_at_truncation_radius = nfw.mass_at_truncation_radius(unit_length='arcsec', unit_mass='solMass',
                                                                  redshift_profile=0.6,
                                                    redshift_source=2.5, redshift_of_cosmic_average_density='profile',
                                                    cosmology=cosmology)

        kpc_per_arcsec = 1.0 / cosmology.arcsec_per_kpc_proper(z=0.6).value

        assert rho == pytest.approx(323442484.90222085 * kpc_per_arcsec**3.0, 1.0e-4)
        assert delta_concentration == pytest.approx(1233086.3244882922, 1.0e-4)
        assert concentration == pytest.approx(36.61521013005619, 1.0e-4)
        assert radius_at_200 == pytest.approx(109.84563039016857 / kpc_per_arcsec, 1.0e-4)
        assert mass_at_200 == pytest.approx(291253092446.923, 1.0e-4)
        assert mass_at_truncation_radius == pytest.approx(177609204745.61484, 1.0e-4)


class TestGeneralizedNFW(object):

    def test__constructor_and_units(self):
        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=45.0,
        #                                    kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)
        #
        # assert gnfw.centre == (0.7, 1.0)
        # assert gnfw.axis_ratio == 0.7
        # assert gnfw.phi == 45.0
        # assert gnfw.kappa_s == 2.0
        # assert gnfw.inner_slope == 1.5
        # assert gnfw.scale_radius == 10.0

        gnfw = mp.SphericalGeneralizedNFW(centre=(1.0, 2.0), kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)

        assert gnfw.centre == (1.0, 2.0)
        assert isinstance(gnfw.centre[0], dim.Length)
        assert isinstance(gnfw.centre[1], dim.Length)
        assert gnfw.centre[0].unit == 'arcsec'
        assert gnfw.centre[1].unit == 'arcsec'

        assert gnfw.axis_ratio == 1.0
        assert isinstance(gnfw.axis_ratio, float)

        assert gnfw.phi == 0.0
        assert isinstance(gnfw.phi, float)

        assert gnfw.kappa_s == 2.0
        assert isinstance(gnfw.kappa_s, float)

        assert gnfw.inner_slope == 1.5
        assert isinstance(gnfw.inner_slope, float)

        assert gnfw.scale_radius == 10.0
        assert isinstance(gnfw.scale_radius, dim.Length)
        assert gnfw.scale_radius.unit_length == 'arcsec'

    # def test__coord_func_x_above_1(self):
    #     assert mp.EllipticalNFW.coord_func(2.0) == pytest.approx(0.60459, 1e-3)
    #
    #     assert mp.EllipticalNFW.coord_func(0.5) == pytest.approx(1.5206919, 1e-3)
    #
    #     assert mp.EllipticalNFW.coord_func(1.0) == 1.0

    def test__convergence_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0)
        assert gnfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.30840, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, inner_slope=1.5, scale_radius=1.0)
        assert gnfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.30840 * 2, 1e-3)

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.5,
        #                                    phi=90.0, inner_slope=1.5, scale_radius=1.0)
        # assert gnfw.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.30840, 1e-3)
        #
        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, axis_ratio=0.5,
        #                                    phi=90.0, inner_slope=1.5, scale_radius=1.0)
        # assert gnfw.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.30840 * 2, 1e-3)

    def test__potential_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0)
        assert gnfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]])) == pytest.approx(0.00920, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=8.0)
        assert gnfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]])) == pytest.approx(0.17448, 1e-3)

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=5.0, axis_ratio=0.5,
        #                                    phi=100.0, inner_slope=1.0, scale_radius=10.0)
        # assert gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]])) == pytest.approx(2.4718, 1e-4)

    def test__deflections_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0)
        defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.43501, 1e-3)
        assert defls[0, 1] == pytest.approx(0.37701, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0)
        defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-9.31254, 1e-3)
        assert defls[0, 1] == pytest.approx(-3.10418, 1e-3)

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.3,
        #                                    phi=100.0, inner_slope=0.5, scale_radius=8.0)
        # defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        # assert defls[0, 0] == pytest.approx(0.26604, 1e-3)
        # assert defls[0, 1] == pytest.approx(0.58988, 1e-3)
        #
        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.3, 0.2), kappa_s=2.5, axis_ratio=0.5,
        #                                    phi=100.0, inner_slope=1.5, scale_radius=4.0)
        # defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        # assert defls[0, 0] == pytest.approx(-5.99032, 1e-3)
        # assert defls[0, 1] == pytest.approx(-4.02541, 1e-3)

    def test__surfce_density__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0))
        assert gnfw_0.convergence_from_grid(grid=np.array([[1.0, 1.0]])) == gnfw_1.convergence_from_grid(
            grid=np.array([[0.0, 0.0]]))

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        assert gnfw_0.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]))

        # gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        # gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        # assert gnfw_0.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.convergence_from_grid(
        #     grid=np.array([[0.0, 1.0]]))

    def test__potential__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0))
        assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 1.0]])) == gnfw_1.potential_from_grid(
            grid=np.array([[0.0, 0.0]]))

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.potential_from_grid(
            grid=np.array([[0.0, 1.0]]))

        # gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        # gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        # assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.potential_from_grid(
        #     grid=np.array([[0.0, 1.0]]))

    def test__deflections__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0)
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0)
        defls_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        defls_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert defls_0[0, 0] == pytest.approx(-defls_1[0, 0], 1e-5)
        assert defls_0[0, 1] == pytest.approx(-defls_1[0, 1], 1e-5)

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0)
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0)
        defls_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

        # gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0,
        #                                      inner_slope=1.5, scale_radius=1.0)
        # gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, kappa_s=1.0,
        #                                      inner_slope=1.5, scale_radius=1.0)
        # defls_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        # defls_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        # assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        # assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(self):
        gNFW = mp.SphericalGeneralizedNFW(centre=(-0.7, 0.5), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = gNFW.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = gNFW.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    # def test__compare_to_nfw(self):
    #     nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0, scale_radius=5.0)
    #     gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0,
    #                                        inner_slope=1.0, scale_radius=5.0)
    #
    #     assert nfw.potential_from_grid(grid) == pytest.approx(gnfw.potential_from_grid(grid), 1e-3)
    #     assert nfw.potential_from_grid(grid) == pytest.approx(gnfw.potential_from_grid(grid), 1e-3)
    #     assert nfw.deflections_from_grid(grid) == pytest.approx(gnfw.deflections_from_grid(grid), 1e-3)
    #     assert nfw.deflections_from_grid(grid) == pytest.approx(gnfw.deflections_from_grid(grid), 1e-3)

    # def test__spherical_and_elliptical_match(self):
    #     elliptical = mp.EllipticalGeneralizedNFW(centre=(0.1, 0.2), axis_ratio=1.0, phi=0.0, kappa_s=2.0,
    #                                              inner_slope=1.5, scale_radius=3.0)
    #     spherical = mp.SphericalGeneralizedNFW(centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0)
    #
    #     assert elliptical.convergence_from_grid(grid) == pytest.approx(spherical.convergence_from_grid(grid),
    #                                                                        1e-4)
    #     assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
    #     assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)


class TestTruncatedNFW(object):

    def test__constructor_and_units(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(1.0, 2.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.centre == (1.0, 2.0)
        assert isinstance(truncated_nfw.centre[0], dim.Length)
        assert isinstance(truncated_nfw.centre[1], dim.Length)
        assert truncated_nfw.centre[0].unit == 'arcsec'
        assert truncated_nfw.centre[1].unit == 'arcsec'

        assert truncated_nfw.axis_ratio == 1.0
        assert isinstance(truncated_nfw.axis_ratio, float)

        assert truncated_nfw.phi == 0.0
        assert isinstance(truncated_nfw.phi, float)

        assert truncated_nfw.kappa_s == 2.0
        assert isinstance(truncated_nfw.kappa_s, float)

        assert truncated_nfw.inner_slope == 1.0
        assert isinstance(truncated_nfw.inner_slope, float)

        assert truncated_nfw.scale_radius == 10.0
        assert isinstance(truncated_nfw.scale_radius, dim.Length)
        assert truncated_nfw.scale_radius.unit_length == 'arcsec'

        assert truncated_nfw.truncation_radius == 2.0
        assert isinstance(truncated_nfw.truncation_radius, dim.Length)
        assert truncated_nfw.truncation_radius.unit_length == 'arcsec'

    def test__convergence_correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == \
               pytest.approx(2.0 * 0.046409642, 1.0e-4)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[1.0, 1.0]])) == \
               pytest.approx(2.0 * 0.10549515, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == \
               pytest.approx(6.0 * 0.046409642, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=5.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == \
               pytest.approx(1.51047026, 1.0e-4)

    def test__deflections_correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        # factor = (4.0 * kappa_s * scale_radius / (r / scale_radius))

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 1.0 * 1.0) / (2.0 / 1.0)
        assert deflections[0, 0] == pytest.approx(factor * 0.38209715, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[0.0, 2.0]]))

        assert deflections[0, 0] == pytest.approx(0.0, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(factor * 0.38209715, 1.0e-4)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

        factor = (4.0 * 1.0 * 1.0) / (np.sqrt(2) / 1.0)
        assert deflections[0, 0] == pytest.approx((1.0 / np.sqrt(2)) * factor * 0.3125838, 1.0e-4)
        assert deflections[0, 1] == pytest.approx((1.0 / np.sqrt(2)) * factor * 0.3125838, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 2.0 * 1.0) / (2.0 / 1.0)
        assert deflections[0, 0] == pytest.approx(factor * 0.38209715, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0,
                                                 truncation_radius=2.0)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(2.1702661386, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

    def test__compare_nfw_and_truncated_nfw_with_large_truncation_radius__convergence_and_deflections_identical(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0,
                                                 truncation_radius=50000.0)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)

        truncated_nfw_convergence = truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 2.0], [3.0, 1.0],
                                                                                       [-1.0, -9.0]]))
        nfw_convergence = nfw.convergence_from_grid(grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]]))

        assert truncated_nfw_convergence == pytest.approx(nfw_convergence, 1.0e-4)

        truncated_nfw_deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 2.0], [3.0, 1.0],
                                                                                       [-1.0, -9.0]]))
        nfw_deflections = nfw.deflections_from_grid(grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]]))

        assert truncated_nfw_deflections == pytest.approx(nfw_deflections, 1.0e-4)

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(-0.7, 0.5), kappa_s=1.0, scale_radius=8.0,
                                                 truncation_radius=2.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = truncated_nfw.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = truncated_nfw.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()

    def test__mass_at_truncation_radius__values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=1.0)

        cosmology = MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=1.0,
                                  cosmic_average_density=1.0)

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_profile=0.5, redshift_source=1.0,
                                                                            unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)

        assert mass_at_truncation_radius == pytest.approx(0.00009792581, 1.0e-5)

        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
        #                                          truncation_radius=1.0)
        #
        # cosmology = MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=2.0,
        #                           cosmic_average_density=3.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.00008789978, 1.0e-5)
        #
        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0,
        #                                          truncation_radius=1.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.0000418378, 1.0e-5)
        #
        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=8.0,
        #                                          truncation_radius=4.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.0000421512, 1.0e-4)
        #
        # truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=8.0,
        #                                          truncation_radius=4.0)
        #
        # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_lens=0.5, redshift_source=1.0,
        #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
        #
        # assert mass_at_truncation_radius == pytest.approx(0.00033636625, 1.0e-4)

    def test_summarize_in_units(self):

        cosmology = cosmo.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)

        nfw = mp.SphericalTruncatedNFW(kappa_s=0.5, scale_radius=5.0, truncation_radius=10.0)

        summary_text = nfw.summarize_in_units(
            radii=[dim.Length(10.0), dim.Length(500.0)], prefix='nfw_',
            unit_length='kpc', unit_mass='solMass',
            redshift_profile=0.6, redshift_source=2.5, redshift_of_cosmic_average_density='profile',
            whitespace=50, cosmology=cosmology)

        index = 0

        assert summary_text[index] == 'Mass Profile = SphericalTruncatedNFW' ; index += 1
        assert summary_text[index] == '' ; index += 1
        assert summary_text[index] == 'nfw_einstein_radius                               15.38 kpc' ; index += 1
        assert summary_text[index] == 'nfw_einstein_mass                                 1.4418e+12 solMass' ; index += 1
        assert summary_text[index] == 'nfw_mass_within_10.00_kpc                         5.2061e+12 solMass' ; index += 1
        assert summary_text[index] == 'nfw_mass_within_500.00_kpc                        7.3287e+12 solMass' ; index += 1
        assert summary_text[index] == 'nfw_rho_at_scale_radius                           29027857.02 solMass / kpc3' ; index += 1
        print(summary_text[index])
        assert summary_text[index] == 'nfw_delta_concentration                           110665.28' ; index += 1
        assert summary_text[index] == 'nfw_concentration                                 14.40' ; index += 1
        assert summary_text[index] == 'nfw_radius_at_200x_cosmic_density                 481.41 kpc' ; index += 1
        assert summary_text[index] == 'nfw_mass_at_200x_cosmic_density                   2.4517e+13 solMass' ; index += 1
        assert summary_text[index] == 'nfw_mass_at_truncation_radius                     1.3190e+13 solMass' ; index += 1


class TestTruncatedNFWChallenge(object):

    def test__constructor_and_units(self):

        truncated_nfw = mp.SphericalTruncatedNFWChallenge(centre=(1.0, 2.0), kappa_s=2.0, scale_radius=10.0)

        assert truncated_nfw.centre == (1.0, 2.0)
        assert isinstance(truncated_nfw.centre[0], dim.Length)
        assert isinstance(truncated_nfw.centre[1], dim.Length)
        assert truncated_nfw.centre[0].unit == 'arcsec'
        assert truncated_nfw.centre[1].unit == 'arcsec'

        assert truncated_nfw.axis_ratio == 1.0
        assert isinstance(truncated_nfw.axis_ratio, float)

        assert truncated_nfw.phi == 0.0
        assert isinstance(truncated_nfw.phi, float)

        assert truncated_nfw.kappa_s == 2.0
        assert isinstance(truncated_nfw.kappa_s, float)

        assert truncated_nfw.inner_slope == 1.0
        assert isinstance(truncated_nfw.inner_slope, float)

        assert truncated_nfw.scale_radius == 10.0
        assert isinstance(truncated_nfw.scale_radius, dim.Length)
        assert truncated_nfw.scale_radius.unit_length == 'arcsec'

        assert truncated_nfw.truncation_radius == pytest.approx(2.0*189.26967095554755, 1.0e-4)
        assert isinstance(truncated_nfw.truncation_radius, dim.Length)
        assert truncated_nfw.truncation_radius.unit_length == 'arcsec'


class TestNFW(object):

    def test__constructor_and_units(self):

        nfw = mp.EllipticalNFW(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, kappa_s=2.0, scale_radius=10.0)

        assert nfw.centre == (1.0, 2.0)
        assert isinstance(nfw.centre[0], dim.Length)
        assert isinstance(nfw.centre[1], dim.Length)
        assert nfw.centre[0].unit == 'arcsec'
        assert nfw.centre[1].unit == 'arcsec'

        assert nfw.axis_ratio == 0.5
        assert isinstance(nfw.axis_ratio, float)

        assert nfw.phi == 45.0
        assert isinstance(nfw.phi, float)

        assert nfw.kappa_s == 2.0
        assert isinstance(nfw.kappa_s, float)

        assert nfw.inner_slope == 1.0
        assert isinstance(nfw.inner_slope, float)

        assert nfw.scale_radius == 10.0
        assert isinstance(nfw.scale_radius, dim.Length)
        assert nfw.scale_radius.unit_length == 'arcsec'

        nfw = mp.SphericalNFW(centre=(1.0, 2.0), kappa_s=2.0, scale_radius=10.0)

        assert nfw.centre == (1.0, 2.0)
        assert isinstance(nfw.centre[0], dim.Length)
        assert isinstance(nfw.centre[1], dim.Length)
        assert nfw.centre[0].unit == 'arcsec'
        assert nfw.centre[1].unit == 'arcsec'

        assert nfw.axis_ratio == 1.0
        assert isinstance(nfw.axis_ratio, float)

        assert nfw.phi == 0.0
        assert isinstance(nfw.phi, float)

        assert nfw.kappa_s == 2.0
        assert isinstance(nfw.kappa_s, float)

        assert nfw.inner_slope == 1.0
        assert isinstance(nfw.inner_slope, float)

        assert nfw.scale_radius == 10.0
        assert isinstance(nfw.scale_radius, dim.Length)
        assert nfw.scale_radius.unit_length == 'arcsec'

    def test__convergence_correct_values(self):
        # r = 2.0 (> 1.0)
        # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
        # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.263600141, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[0.5, 0.0]])) == pytest.approx(1.388511, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[0.5, 0.0]])) == pytest.approx(2.0 * 1.388511, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)
        assert nfw.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(1.388511, 1e-3)

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, kappa_s=1.0, scale_radius=1.0)
        assert nfw.convergence_from_grid(grid=np.array([[0.25, 0.0]])) == pytest.approx(1.388511, 1e-3)

    def test__potential_correct_values(self):
        nfw = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        assert nfw.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(0.03702, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        assert nfw.potential_from_grid(grid=np.array([[0.1875, 0.1625]])) == pytest.approx(0.03702, 1e-3)

        nfw = mp.EllipticalNFW(centre=(0.3, 0.2), axis_ratio=0.7, phi=6.0, kappa_s=2.5, scale_radius=4.0)
        assert nfw.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.05380, 1e-3)

    def test__potential__spherical_and_elliptical_are_same(self):
        nfw_spherical = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        nfw_elliptical = mp.EllipticalNFW(centre=(0.3, 0.2), axis_ratio=1.0, phi=0.0, kappa_s=2.5, scale_radius=4.0)

        potential_spherical = nfw_spherical.potential_from_grid(grid=np.array([[0.1875, 0.1625]]))
        potential_elliptical = nfw_elliptical.potential_from_grid(grid=np.array([[0.1875, 0.1625]]))

        assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)

        potential_spherical = nfw_spherical.potential_from_grid(grid=np.array([[50.0, 50.0]]))
        potential_elliptical = nfw_elliptical.potential_from_grid(grid=np.array([[50.0, 50.0]]))

        assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)

        potential_spherical = nfw_spherical.potential_from_grid(grid=np.array([[-50.0, -50.0]]))
        potential_elliptical = nfw_elliptical.potential_from_grid(grid=np.array([[-50.0, -50.0]]))

        assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)

    def test__deflections_correct_values(self):
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.56194, 1e-3)
        assert defls[0, 1] == pytest.approx(0.56194, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
        defls = nfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-2.08909, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.69636, 1e-3)

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, kappa_s=1.0, scale_radius=1.0)
        defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.56194, 1e-3)
        assert defls[0, 1] == pytest.approx(0.56194, 1e-3)

        nfw = mp.EllipticalNFW(centre=(0.3, 0.2), axis_ratio=0.7, phi=6.0, kappa_s=2.5, scale_radius=4.0)
        defls = nfw.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-2.59480, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.44204, 1e-3)

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        nfw = mp.EllipticalNFW(centre=(-0.7, 0.5), axis_ratio=0.9, phi=45.0, kappa_s=1.0, scale_radius=8.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = nfw.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = nfw.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__dont_use_interpolate_and_cache_decorators(self):
        nfw = mp.SphericalNFW(centre=(-0.7, 0.5), kappa_s=1.0, scale_radius=8.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = nfw.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = nfw.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y != interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:, 1]).all()


class TestSersic(object):

    def test__constructor_and_units(self):
    
        sersic = mp.EllipticalSersic(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                     effective_radius=0.6, sersic_index=4.0, mass_to_light_ratio=10.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = mp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                    mass_to_light_ratio=10.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=3.0, effective_radius=2.0, sersic_index=2.0,
                                    mass_to_light_ratio=1.0)
        assert sersic.convergence_from_grid(grid=np.array([[0.0, 1.5]])) == pytest.approx(4.90657319276, 1e-3)

        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=6.0, effective_radius=2.0, sersic_index=2.0,
                                    mass_to_light_ratio=1.0)
        assert sersic.convergence_from_grid(grid=np.array([[0.0, 1.5]])) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=3.0, effective_radius=2.0, sersic_index=2.0,
                                    mass_to_light_ratio=2.0)
        assert sersic.convergence_from_grid(grid=np.array([[0.0, 1.5]])) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                     sersic_index=2.0, mass_to_light_ratio=1.0)
        assert sersic.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(5.38066670129, 1e-3)

    def test__deflections_correct_values(self):
        sersic = mp.EllipticalSersic(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                     effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0)
        defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(1.1446, 1e-3)
        assert defls[0, 1] == pytest.approx(0.79374, 1e-3)

        sersic = mp.EllipticalSersic(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                     effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0)
        defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(1.1446, 1e-3)
        assert defls[0, 1] == pytest.approx(0.79374, 1e-3)
        assert defls[1, 0] == pytest.approx(1.1446, 1e-3)
        assert defls[1, 1] == pytest.approx(0.79374, 1e-3)

    def test__surfce_density__change_geometry(self):
        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(1.0, 1.0))
        assert sersic_0.convergence_from_grid(grid=np.array([[1.0, 1.0]])) == sersic_1.convergence_from_grid(
            grid=np.array([[0.0, 0.0]]))

        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(0.0, 0.0))
        assert sersic_0.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == sersic_1.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]))

        sersic_0 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert sersic_0.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == sersic_1.convergence_from_grid(
            grid=np.array([[0.0, 1.0]]))

    def test__deflections__change_geometry(self):
        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(1.0, 1.0))
        defls_0 = sersic_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        defls_1 = sersic_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert defls_0[0, 0] == pytest.approx(-defls_1[0, 0], 1e-5)
        assert defls_0[0, 1] == pytest.approx(-defls_1[0, 1], 1e-5)

        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(0.0, 0.0))
        defls_0 = sersic_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = sersic_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

        sersic_0 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        defls_0 = sersic_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = sersic_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         effective_radius=1.0, sersic_index=4.0,
                                         mass_to_light_ratio=1.0)

        spherical = mp.SphericalSersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0,
                                       sersic_index=4.0, mass_to_light_ratio=1.0)

        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        sersic = mp.EllipticalSersic(centre=(-0.7, 0.5), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                     effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = sersic.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(self):
        sersic = mp.SphericalSersic(centre=(-0.7, 0.5), intensity=5.0, effective_radius=0.2, sersic_index=2.0,
                                    mass_to_light_ratio=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = sersic.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()


class TestExponential(object):

    def test__constructor_and_units(self):

        exponential = mp.EllipticalExponential(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                               effective_radius=0.6, mass_to_light_ratio=10.0)

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == 'arcsec'
        assert exponential.centre[1].unit == 'arcsec'

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == 'eps'

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == 'arcsec'

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == 'angular / eps'

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = mp.SphericalExponential(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6,
                                              mass_to_light_ratio=10.0)

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == 'arcsec'
        assert exponential.centre[1].unit == 'arcsec'

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == 'eps'

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == 'arcsec'

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == 'angular / eps'

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(4.9047, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(4.8566, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=4.0, effective_radius=3.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                               mass_to_light_ratio=2.0)
        assert exponential.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(4.8566, 1e-3)

    def test__deflections_correct_values(self):
        exponential = mp.EllipticalExponential(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                               effective_radius=0.2, mass_to_light_ratio=1.0)
        defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.90493, 1e-3)
        assert defls[0, 1] == pytest.approx(0.62569, 1e-3)

        exponential = mp.EllipticalExponential(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                               effective_radius=0.2, mass_to_light_ratio=1.0)
        defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.90493, 1e-3)
        assert defls[0, 1] == pytest.approx(0.62569, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                              effective_radius=1.0, mass_to_light_ratio=1.0)

        spherical = mp.SphericalExponential(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0,
                                            mass_to_light_ratio=1.0)

        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        exponential = mp.EllipticalExponential(centre=(-0.7, 0.5), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                               effective_radius=0.2, mass_to_light_ratio=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = exponential.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(self):
        exponential = mp.SphericalExponential(centre=(-0.7, 0.5), intensity=5.0, effective_radius=0.2,
                                              mass_to_light_ratio=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = exponential.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()


class TestDevVaucouleurs(object):

    def test__constructor_and_units(self):

        dev_vaucouleurs = mp.EllipticalDevVaucouleurs(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                                      effective_radius=0.6, mass_to_light_ratio=10.0)

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == 'arcsec'
        assert dev_vaucouleurs.centre[1].unit == 'arcsec'

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == 'eps'

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == 'arcsec'

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, dim.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == 'angular / eps'

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = mp.SphericalDevVaucouleurs(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6,
                                                     mass_to_light_ratio=10.0)

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == 'arcsec'
        assert dev_vaucouleurs.centre[1].unit == 'arcsec'

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == 'eps'

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == 'arcsec'

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, dim.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == 'angular / eps'

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                          mass_to_light_ratio=1.0)
        assert dev.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(5.6697, 1e-3)

        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                          mass_to_light_ratio=1.0)
        assert dev.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(7.4455, 1e-3)

        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=4.0, effective_radius=3.0,
                                          mass_to_light_ratio=1.0)
        assert dev.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 7.4455, 1e-3)

        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                          mass_to_light_ratio=2.0)
        assert dev.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 7.4455, 1e-3)

        sersic = mp.SphericalDevVaucouleurs(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.6,
                                            mass_to_light_ratio=1.0)
        assert sersic.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.351797, 1e-3)

    def test__deflections_correct_values(self):
        dev = mp.EllipticalDevVaucouleurs(centre=(0.4, 0.2), axis_ratio=0.9, phi=10.0, intensity=2.0,
                                          effective_radius=0.8, mass_to_light_ratio=3.0)
        defls = dev.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-24.528, 1e-3)
        assert defls[0, 1] == pytest.approx(-3.37605, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalDevVaucouleurs(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                 effective_radius=1.0, mass_to_light_ratio=1.0)

        spherical = mp.SphericalDevVaucouleurs(centre=(0.0, 0.0), intensity=1.0,
                                               effective_radius=1.0, mass_to_light_ratio=1.0)

        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)

        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        dev = mp.EllipticalDevVaucouleurs(centre=(-0.7, 0.5), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                          effective_radius=0.2, mass_to_light_ratio=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = dev.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(self):
        dev = mp.SphericalDevVaucouleurs(centre=(-0.7, 0.5), intensity=5.0, effective_radius=0.2,
                                         mass_to_light_ratio=1.0)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = dev.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()


class TestSersicMassRadialGradient(object):

    def test__constructor_and_units(self):

        sersic = mp.EllipticalSersicRadialGradient(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                     effective_radius=0.6, sersic_index=4.0, mass_to_light_ratio=10.0,
                                                   mass_to_light_gradient=-1.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = mp.SphericalSersicRadialGradient(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0,
                                    mass_to_light_ratio=10.0, mass_to_light_gradient=-1.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == 'angular / eps'

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
        sersic = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                   effective_radius=0.6, sersic_index=4.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=1.0)
        assert sersic.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.6 * 0.351797, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75
        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=-1.0)
        assert sersic.convergence_from_grid(grid=np.array([[1.5, 0.0]])) == pytest.approx(0.75 * 4.90657319276,
                                                                                          1e-3)

        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=1.0, phi=0.0, intensity=6.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=-1.0)
        assert sersic.convergence_from_grid(grid=np.array([[1.5, 0.0]])) == pytest.approx(
            2.0 * 0.75 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=2.0,
                                                   mass_to_light_gradient=-1.0)
        assert sersic.convergence_from_grid(grid=np.array([[1.5, 0.0]])) == pytest.approx(
            2.0 * 0.75 * 4.90657319276, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836
        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=1.0)
        assert sersic.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(
            2.836879 * 5.38066670129, abs=2e-01)

    def test__deflections_correct_values(self):
        sersic = mp.EllipticalSersicRadialGradient(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                                   effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=1.0)
        defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(3.60324873535244, 1e-3)
        assert defls[0, 1] == pytest.approx(2.3638898009652, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                                   effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=-1.0)
        defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.97806399756448, 1e-3)
        assert defls[0, 1] == pytest.approx(0.725459334118341, 1e-3)

    def test__compare_to_sersic(self):
        sersic = mp.EllipticalSersicRadialGradient(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                                   effective_radius=0.2, sersic_index=1.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=0.0)
        sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        exponential = mp.EllipticalExponential(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                               effective_radius=0.2, mass_to_light_ratio=1.0)
        exponential_defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        assert sersic_defls[0, 0] == exponential_defls[0, 0] == pytest.approx(0.90493, 1e-3)
        assert sersic_defls[0, 1] == exponential_defls[0, 1] == pytest.approx(0.62569, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(centre=(0.4, 0.2), axis_ratio=0.9, phi=10.0, intensity=2.0,
                                                   effective_radius=0.8, sersic_index=4.0, mass_to_light_ratio=3.0,
                                                   mass_to_light_gradient=0.0)
        sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        dev = mp.EllipticalDevVaucouleurs(centre=(0.4, 0.2), axis_ratio=0.9, phi=10.0, intensity=2.0,
                                          effective_radius=0.8, mass_to_light_ratio=3.0)

        dev_defls = dev.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        assert sersic_defls[0, 0] == dev_defls[0, 0] == pytest.approx(-24.528, 1e-3)
        assert sersic_defls[0, 1] == dev_defls[0, 1] == pytest.approx(-3.37605, 1e-3)

        sersic_grad = mp.EllipticalSersicRadialGradient(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                                        effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0,
                                                        mass_to_light_gradient=0.0)
        sersic_grad_defls = sersic_grad.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        sersic = mp.EllipticalSersic(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                     effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0)
        sersic_defls = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        assert sersic_grad_defls[0, 0] == sersic_defls[0, 0] == pytest.approx(1.1446, 1e-3)
        assert sersic_grad_defls[0, 1] == sersic_defls[0, 1] == pytest.approx(0.79374, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                       effective_radius=1.0, sersic_index=4.0,
                                                       mass_to_light_ratio=1.0, mass_to_light_gradient=1.0)
        spherical = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0,
                                                      sersic_index=4.0, mass_to_light_ratio=1.0,
                                                      mass_to_light_gradient=1.0)
        assert (elliptical.convergence_from_grid(grid) == spherical.convergence_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        assert (elliptical.deflections_from_grid(grid) == spherical.deflections_from_grid(grid)).all()

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(self):
        sersic = mp.EllipticalSersicRadialGradient(centre=(-0.7, 0.5), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                                   effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=1.5)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = sersic.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(self):
        sersic = mp.SphericalSersicRadialGradient(centre=(-0.7, 0.5), intensity=5.0, effective_radius=0.2,
                                                  sersic_index=2.0,
                                                  mass_to_light_ratio=1.0, mass_to_light_gradient=1.5)

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, True, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.RegularGrid.from_mask(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5)

        interp_deflections_values = sersic.deflections_from_grid(grid=interpolator.interp_grid)

        interp_deflections_manual_y = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 0])
        interp_deflections_manual_x = \
            interpolator.interpolated_values_from_values(values=interp_deflections_values[:, 1])

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()


class TestMassSheet(object):

    def test__constructor_and_units(self):
        
        mass_sheet = mp.MassSheet(centre=(1.0, 2.0), kappa=2.0)

        assert mass_sheet.centre == (1.0, 2.0)
        assert isinstance(mass_sheet.centre[0], dim.Length)
        assert isinstance(mass_sheet.centre[1], dim.Length)
        assert mass_sheet.centre[0].unit == 'arcsec'
        assert mass_sheet.centre[1].unit == 'arcsec'

        assert mass_sheet.kappa == 2.0
        assert isinstance(mass_sheet.kappa, float)

    def test__convergence__correct_values(self):
        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        convergence = mass_sheet.convergence_from_grid(grid=np.array([[1.0, 0.0]]))

        assert convergence[0] == pytest.approx(1.0, 1e-3)

        convergence = mass_sheet.convergence_from_grid(grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]]))

        assert convergence[0] == pytest.approx(1.0, 1e-3)
        assert convergence[1] == pytest.approx(1.0, 1e-3)
        assert convergence[2] == pytest.approx(1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=-3.0)

        convergence = mass_sheet.convergence_from_grid(grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]]))

        assert convergence[0] == pytest.approx(-3.0, 1e-3)
        assert convergence[1] == pytest.approx(-3.0, 1e-3)
        assert convergence[2] == pytest.approx(-3.0, 1e-3)

    def test__deflections__correct_values(self):
        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=-1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(-2.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(4.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        # The radial coordinate at (1.0, 1.0) is sqrt(2)
        # This is decomposed into (y,x) angles of sin(45) = cos(45) = sqrt(2) / 2.0
        # Thus, for a mass sheet, the deflection angle is (sqrt(2) * sqrt(2) / 2.0) = 1.0

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(2.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 2.0]]))
        assert deflections[0, 0] == pytest.approx(4.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(4.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        # The radial coordinate at (2.0, 1.0) is sqrt(5)
        # This gives an angle of 26.5650512 degrees between the 1.0 and np.sqrt(5) of the triangle
        # This is decomposed into y angle of cos(26.5650512 degrees) = 0.8944271
        # This is decomposed into x angle of sin(26.5650512 degrees) = 0.4472135
        # Thus, for a mass sheet, the deflection angles are:
        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(0.8944271 * np.sqrt(5), 1e-3)
        assert deflections[0, 1] == pytest.approx(0.4472135 * np.sqrt(5), 1e-3)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[-1.0, -1.0]]))
        assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(-1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(1.0, 2.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(1.0, 2.0), kappa=-1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(-1.0, 1e-3)

    def test__deflections__change_geometry(self):
        mass_sheet_0 = mp.MassSheet(centre=(0.0, 0.0))
        mass_sheet_1 = mp.MassSheet(centre=(1.0, 1.0))
        defls_0 = mass_sheet_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        defls_1 = mass_sheet_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert defls_0[0, 0] == pytest.approx(-defls_1[0, 0], 1e-5)
        assert defls_0[0, 1] == pytest.approx(-defls_1[0, 1], 1e-5)

        mass_sheet_0 = mp.MassSheet(centre=(0.0, 0.0))
        mass_sheet_1 = mp.MassSheet(centre=(0.0, 0.0))
        defls_0 = mass_sheet_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = mass_sheet_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

    def test__multiple_coordinates_in__multiple_coordinates_out(self):
        mass_sheet = mp.MassSheet(centre=(1.0, 2.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)
        assert deflections[1, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[1, 1] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        assert deflections[1, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[1, 1] == pytest.approx(2.0, 1e-3)

        assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 1] == pytest.approx(1.0, 1e-3)

        assert deflections[3, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[3, 1] == pytest.approx(2.0, 1e-3)


class TestExternalShear(object):

    def test__constructor_and_units(self):

        shear = mp.ExternalShear(magnitude=0.05, phi=45.0)

        assert shear.magnitude == 0.05
        assert isinstance(shear.magnitude, float)

        assert shear.phi == 45.0
        assert isinstance(shear.phi, float)

    def test__convergence_returns_zeros(self):

        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        convergence = shear.convergence_from_grid(grid=np.array([0.1]))
        assert (convergence == np.array([0.0])).all()

        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        convergence = shear.convergence_from_grid(grid=np.array([0.1, 0.2, 0.3]))
        assert (convergence == np.array([0.0, 0.0, 0.0])).all()

    def test__potential_returns_zeros(self):
        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        potential = shear.potential_from_grid(grid=np.array([0.1]))
        assert (potential == np.array([0.0])).all()

        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        potential = shear.potential_from_grid(grid=np.array([0.1, 0.2, 0.3]))
        assert (potential == np.array([0.0, 0.0, 0.0])).all()

    def test__deflections_correct_values(self):
        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        defls = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.01625, 1e-3)
        assert defls[0, 1] == pytest.approx(0.01625, 1e-3)

        shear = mp.ExternalShear(magnitude=0.2, phi=75.0)
        defls = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.04439, 1e-3)
        assert defls[0, 1] == pytest.approx(-0.011895, 1e-3)


class TestEinsteinRadiusMass(object):

    def test__radius_of_critical_curve_and_einstein_radius__radius_unit_conversions(self):

        cosmology = MockCosmology(kpc_per_arcsec=2.0)

        sis_arcsec = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                            einstein_radius=dim.Length(2.0, 'arcsec'))

        radius = sis_arcsec.average_convergence_of_1_radius_in_units(unit_length='arcsec', cosmology=cosmology)
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_arcsec.einstein_radius_in_units(unit_length='arcsec')
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_arcsec.einstein_radius_in_units(unit_length='kpc', redshift_profile=0.5, cosmology=cosmology)
        assert radius == pytest.approx(4.0, 1e-4)
        
        sis_kpc = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                         einstein_radius=dim.Length(2.0, 'kpc'))

        radius = sis_kpc.average_convergence_of_1_radius_in_units(unit_length='kpc')
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_kpc.einstein_radius_in_units(unit_length='kpc')
        assert radius == pytest.approx(2.0, 1e-4)

        radius = sis_kpc.einstein_radius_in_units(unit_length='arcsec', redshift_profile=0.5, cosmology=cosmology)
        assert radius == pytest.approx(1.0, 1e-4)

        nfw_arcsec = mp.SphericalNFW(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                     kappa_s=0.5, scale_radius=dim.Length(5.0, 'arcsec'))

        radius = nfw_arcsec.average_convergence_of_1_radius_in_units(unit_length='arcsec')
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_arcsec.einstein_radius_in_units(unit_length='arcsec', cosmology=cosmology)
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_arcsec.einstein_radius_in_units(unit_length='kpc', redshift_profile=0.5, cosmology=cosmology)
        assert radius == pytest.approx(2.0*2.76386, 1e-4)

        nfw_kpc = mp.SphericalNFW(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                  kappa_s=0.5, scale_radius=dim.Length(5.0, 'kpc'))

        radius = nfw_kpc.average_convergence_of_1_radius_in_units(unit_length='kpc', redshift_profile=0.5, cosmology=cosmology)
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_kpc.einstein_radius_in_units(unit_length='kpc')
        assert radius == pytest.approx(2.76386, 1e-4)

        radius = nfw_kpc.einstein_radius_in_units(unit_length='arcsec', redshift_profile=0.5, cosmology=cosmology)
        assert radius == pytest.approx(0.5*2.76386, 1e-4)

    def test__einstein_mass__radius_unit_conversions(self):

        cosmology = MockCosmology(kpc_per_arcsec=2.0, critical_surface_density=2.0)


        sis_arcsec = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                            einstein_radius=dim.Length(1.0, 'arcsec'))

        mass = sis_arcsec.einstein_mass_in_units(unit_mass='angular', cosmology=cosmology)
        assert mass == pytest.approx(np.pi, 1e-4)

        mass = sis_arcsec.einstein_mass_in_units(unit_mass='solMass', redshift_profile=0.5, redshift_source=1.0,
                                                 cosmology=cosmology)
        assert mass == pytest.approx(2.0*np.pi, 1e-4)


        sis_kpc = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                         einstein_radius=dim.Length(2.0, 'kpc'))

        mass = sis_kpc.einstein_mass_in_units(unit_mass='angular', redshift_profile=0.5, cosmology=cosmology)
        assert mass == pytest.approx(4.0*np.pi, 1e-4)

        mass = sis_kpc.einstein_mass_in_units(unit_mass='solMass', redshift_profile=0.5, redshift_source=1.0,
                                              cosmology=cosmology)
        assert mass == pytest.approx(2.0*np.pi, 1e-4)


        nfw_arcsec = mp.SphericalNFW(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                     kappa_s=0.5, scale_radius=dim.Length(5.0, 'arcsec'))

        mass = nfw_arcsec.einstein_mass_in_units(unit_mass='angular', cosmology=cosmology)
        assert mass == pytest.approx(np.pi * 2.76386**2.0, 1e-4)

        mass = nfw_arcsec.einstein_mass_in_units(unit_mass='solMass', redshift_profile=0.5, redshift_source=1.0,
                                                 cosmology=cosmology)
        assert mass == pytest.approx(2.0 * np.pi * 2.76386**2.0, 1e-4)


        nfw_kpc = mp.SphericalNFW(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                  kappa_s=0.5, scale_radius=dim.Length(5.0, 'kpc'))

        mass = nfw_kpc.einstein_mass_in_units(unit_mass='angular', redshift_profile=0.5, cosmology=cosmology)
        assert mass == pytest.approx(np.pi * 2.76386**2.0, 1e-4)

        mass = nfw_kpc.einstein_mass_in_units(unit_mass='solMass', redshift_profile=0.5, redshift_source=1.0,
                                              cosmology=cosmology)
        assert mass == pytest.approx(0.5 * np.pi * 2.76386**2.0, 1e-4)


def mass_within_radius_of_profile_from_grid_calculation(radius, profile):

    mass_total = 0.0

    xs = np.linspace(-radius*1.5, radius*1.5, 40)
    ys = np.linspace(-radius*1.5, radius*1.5, 40)

    edge = xs[1] - xs[0]
    area = edge ** 2

    for x in xs:
        for y in ys:

            eta = profile.grid_to_elliptical_radii(np.array([[x,y]]))

            if eta < radius:
                mass_total += profile.convergence_func(eta) * area

    return mass_total


class TestMassWithinCircle(object):

    def test__mass_in_angular_units__singular_isothermal_sphere__compare_to_analytic(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        radius = dim.Length(2.0, 'arcsec')

        mass = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                               unit_mass='angular')
        assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

        sis = mp.SphericalIsothermal(einstein_radius=4.0)

        radius = dim.Length(4.0, 'arcsec')

        mass = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                               unit_mass='angular')
        assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

    def test__mass_in_angular_units__singular_isothermal__compare_to_grid(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        radius = dim.Length(1.0, 'arcsec')

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(radius=radius, profile=sis)

        mass = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                               unit_mass='angular')

        assert mass_grid == pytest.approx(mass, 0.02)

    def test__radius_units_conversions__mass_profile_updates_units_and_computes_correct_mass(self):

        cosmology = MockCosmology(kpc_per_arcsec=2.0)

        # arcsec -> arcsec

        sis_arcsec = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                            einstein_radius=dim.Length(2.0, 'arcsec'))

        radius = dim.Length(2.0, 'arcsec')
        mass = sis_arcsec.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                      unit_mass='angular')
        assert math.pi * sis_arcsec.einstein_radius * radius == pytest.approx(mass, 1e-3)

        # arcsec -> kpc

        radius = dim.Length(2.0, 'kpc')
        mass = sis_arcsec.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                      unit_mass='angular', cosmology=cosmology)
        assert math.pi * sis_arcsec.einstein_radius * 1.0 == pytest.approx(mass, 1e-3)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, 'arcsec')
        mass_arcsec = sis_arcsec.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                             unit_mass='angular', cosmology=cosmology)
        radius = dim.Length(4.0, 'kpc')
        mass_kpc = sis_arcsec.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                          unit_mass='angular', cosmology=cosmology)
        assert mass_arcsec == mass_kpc

        # kpc -> kpc

        sis_kpc = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                         einstein_radius=dim.Length(2.0,'kpc'))

        radius = dim.Length(2.0, 'kpc')
        mass = sis_kpc.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                   unit_mass='angular', cosmology=cosmology)
        assert math.pi * sis_kpc.einstein_radius * radius == pytest.approx(mass, 1e-3)

        # kpc -> arcsec

        radius = dim.Length(2.0, 'arcsec')
        mass = sis_kpc.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                   unit_mass='angular', cosmology=cosmology)
        assert 2.0 * math.pi * sis_kpc.einstein_radius * radius == pytest.approx(mass, 1e-3)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, 'arcsec')
        mass_arcsec = sis_kpc.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                          unit_mass='angular', cosmology=cosmology)
        radius = dim.Length(4.0, 'kpc')
        mass_kpc = sis_kpc.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                       unit_mass='angular')
        assert mass_arcsec == mass_kpc

    def test__mass_units_conversions__multiplies_by_critical_surface_density_factor(self):

        cosmology = MockCosmology(critical_surface_density=2.0)

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        radius = dim.Length(2.0, 'arcsec')

        mass = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                               unit_mass='angular', cosmology=cosmology)
        assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

        mass = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                               unit_mass='solMass', cosmology=cosmology)
        assert 2.0 * math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)

        mass = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                               unit_mass='solMass', cosmology=cosmology)
        assert 2.0 * math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)


class TestMassWithinEllipse(object):
    
    def test__mass_in_angular_units__singular_isothermal_sphere__compare_circle_and_ellipse(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        radius = dim.Length(2.0)
        mass_circle = sis.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                      unit_mass='angular')
        mass_ellipse = sis.mass_within_ellipse_in_units(major_axis=radius, redshift_profile=0.5, redshift_source=1.0,
                                                        unit_mass='angular')
        assert mass_circle == mass_ellipse

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)
        radius = dim.Length(2.0)
        mass_circle = sie.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                      unit_mass='angular')
        mass_ellipse = sie.mass_within_ellipse_in_units(major_axis=radius, redshift_profile=0.5, redshift_source=1.0,
                                                        unit_mass='angular')
        assert mass_circle == mass_ellipse * 2.0

    def test__mass_in_angular_units__singular_isothermal_ellipsoid__compare_to_grid(self):

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

        radius = dim.Length(0.5)

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(radius=radius, profile=sie)

        mass = sie.mass_within_ellipse_in_units(major_axis=radius, redshift_profile=0.5, redshift_source=1.0,
                                                unit_mass='angular')

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert mass_grid == pytest.approx(mass, 0.1)

    def test__radius_units_conversions__mass_profile_updates_units_and_computes_correct_mass(self):

        cosmology = MockCosmology(kpc_per_arcsec=2.0)

        # arcsec -> arcsec

        sie_arcsec = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')),
                                            einstein_radius=dim.Length(2.0, 'arcsec'))

        major_axis = dim.Length(0.5, 'arcsec')

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(radius=major_axis, profile=sie_arcsec)

        mass = sie_arcsec.mass_within_ellipse_in_units(major_axis=major_axis, redshift_profile=0.5, redshift_source=1.0,
                                                       unit_mass='angular')
        assert mass_grid == pytest.approx(mass, 0.1)

        # arcsec -> kpc

        major_axis = dim.Length(0.5, 'kpc')
        mass = sie_arcsec.mass_within_ellipse_in_units(major_axis=major_axis, redshift_profile=0.5, redshift_source=1.0,
                                                       unit_mass='angular', cosmology=cosmology)
        assert 0.5 * mass_grid == pytest.approx(mass, 0.1)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, 'arcsec')
        mass_arcsec = sie_arcsec.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                             unit_mass='angular', cosmology=cosmology)
        radius = dim.Length(4.0, 'kpc')
        mass_kpc = sie_arcsec.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                          unit_mass='angular', cosmology=cosmology)
        assert mass_arcsec == mass_kpc

        # kpc -> kpc

        sie_kpc = mp.SphericalIsothermal(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                         einstein_radius=dim.Length(2.0,'kpc'))

        major_axis = dim.Length(0.5, 'kpc')
        mass = sie_kpc.mass_within_ellipse_in_units(major_axis=major_axis, redshift_profile=0.5, redshift_source=1.0,
                                                    unit_mass='angular')
        assert mass_grid == pytest.approx(mass, 0.1)

        # kpc -> arcsec

        major_axis = dim.Length(0.5, 'arcsec')
        mass = sie_kpc.mass_within_ellipse_in_units(major_axis=major_axis, redshift_profile=0.5, redshift_source=1.0,
                                                    unit_mass='angular', cosmology=cosmology)
        assert 2.0 * mass_grid == pytest.approx(mass, 0.1)

        # 2.0 arcsec = 4.0 kpc, same masses.

        radius = dim.Length(2.0, 'arcsec')
        mass_arcsec = sie_kpc.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                          unit_mass='angular', cosmology=cosmology)
        radius = dim.Length(4.0, 'kpc')
        mass_kpc = sie_kpc.mass_within_circle_in_units(radius=radius, redshift_profile=0.5, redshift_source=1.0,
                                                       unit_mass='angular')
        assert mass_arcsec == mass_kpc

    def test__mass_unit_conversions__compare_to_grid__mutliplies_by_critical_surface_density(self):

        cosmology = MockCosmology(critical_surface_density=2.0)

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

        radius = dim.Length(2.0, 'arcsec')

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(radius=radius, profile=sie)
        mass = sie.mass_within_ellipse_in_units(major_axis=radius, redshift_profile=0.5, redshift_source=1.0,
                                                unit_mass='angular', cosmology=cosmology)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert mass_grid == pytest.approx(radius * sie.axis_ratio * mass, 0.1)

        critical_surface_density = dim.MassOverLength2(2.0, 'arcsec', 'solMass')
        mass = sie.mass_within_ellipse_in_units(major_axis=radius, redshift_profile=0.5, redshift_source=1.0,
                                                unit_mass='solMass', cosmology=cosmology)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert mass_grid == pytest.approx(0.5 * radius * sie.axis_ratio * mass, 0.1)


class TestDensityBetweenAnnuli(object):

    def test__circular_annuli__sis__analyic_density_agrees(self):

        cosmology = MockCosmology(kpc_per_arcsec=2.0, critical_surface_density=2.0)

        einstein_radius = 1.0
        sis_arcsec = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=einstein_radius)

        inner_annuli_radius = dim.Length(2.0, 'arcsec')
        outer_annuli_radius = dim.Length(3.0, 'arcsec')

        inner_mass = math.pi * einstein_radius * inner_annuli_radius
        outer_mass = math.pi * einstein_radius * outer_annuli_radius

        density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius, outer_annuli_radius=outer_annuli_radius,
            unit_length='arcsec', unit_mass='angular', redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (np.pi * inner_annuli_radius ** 2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(density_between_annuli, 1e-4)

        density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius, outer_annuli_radius=outer_annuli_radius,
            unit_length='arcsec', unit_mass='solMass', redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (np.pi * inner_annuli_radius ** 2.0)

        assert (2.0*outer_mass - 2.0*inner_mass) / annuli_area == pytest.approx(density_between_annuli, 1e-4)

        density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius, outer_annuli_radius=outer_annuli_radius,
            unit_length='kpc', unit_mass='angular', redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

        inner_mass = math.pi * 2.0 * einstein_radius * inner_annuli_radius
        outer_mass = math.pi * 2.0 * einstein_radius * outer_annuli_radius

        annuli_area = (np.pi * 2.0*outer_annuli_radius ** 2.0) - (np.pi * 2.0*inner_annuli_radius ** 2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(density_between_annuli, 1e-4)

    def test__circular_annuli__nfw_profile__compare_to_manual_mass(self):

        cosmology = MockCosmology(kpc_per_arcsec=2.0, critical_surface_density=2.0)

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, kappa_s=1.0)

        inner_mass = nfw.mass_within_circle_in_units(radius=dim.Length(1.0), redshift_profile=0.5, redshift_source=1.0,
                                                     unit_mass='angular')

        outer_mass = nfw.mass_within_circle_in_units(radius=dim.Length(2.0), redshift_profile=0.5, redshift_source=1.0,
                                                     unit_mass='angular')

        density_between_annuli = nfw.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=dim.Length(1.0), outer_annuli_radius=dim.Length(2.0),
            unit_length='arcsec', unit_mass='angular', redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology)

        annuli_area = (np.pi * 2.0 ** 2.0) - (np.pi * 1.0 ** 2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(density_between_annuli, 1e-4)

