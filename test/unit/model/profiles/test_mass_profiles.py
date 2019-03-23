import math

import numpy as np
import pytest
from astropy import cosmology

from autolens.data.array import mask as msk
from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestPointMass(object):

    def test_constructor(self):

        point_mass = mp.PointMass(centre=(1.0, 1.0), einstein_radius=2.0)

        assert point_mass.centre == (1.0, 1.0)
        assert point_mass.einstein_radius == 2.0

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
        assert deflections[0, 0] == pytest.approx(8.0/97.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(18.0/97.0, 1e-3)

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()
    # def test__mass(self):
    #
    #     point_mass = mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.91716)
    #
    #     print(point_mass.mass)
    #
    #     assert point_mass.mass == pytest.approx(1.3332e11, 1e9)


class TestCoredPowerLaw(object):

    def test__constructor(self):
        power_law = mp.EllipticalCoredPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                               einstein_radius=1.0, slope=2.2, core_radius=0.1)

        assert power_law.centre == (1.0, 1.0)
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.2
        assert power_law.core_radius == 0.1
        # (3 - slope) / (1 + axis_ratio) * (1.0) = (3 - 2) / (1 + 1) * (1.1)**1.2 = 0.5
        assert power_law.einstein_radius_rescaled == pytest.approx(0.4, 1e-3)

        power_law = mp.SphericalCoredPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.2,
                                              core_radius=0.1)

        assert power_law.centre == (1.0, 1.0)
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 0.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.2
        assert power_law.core_radius == 0.1
        assert power_law.einstein_radius_rescaled == pytest.approx(0.4, 1e-3)

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()


class TestPowerLaw(object):

    def test__constructor(self):
        power_law = mp.EllipticalPowerLaw(centre=(1, 1), axis_ratio=1.0, phi=45.0, einstein_radius=1.0, slope=2.0)

        assert power_law.centre == (1.0, 1.0)
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 45.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.0
        assert power_law.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

        power_law = mp.SphericalPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.0)

        assert power_law.centre == (1.0, 1.0)
        assert power_law.axis_ratio == 1.0
        assert power_law.phi == 0.0
        assert power_law.einstein_radius == 1.0
        assert power_law.slope == 2.0
        assert power_law.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()


class TestCoredIsothermal(object):

    def test__constructor(self):
        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                                        einstein_radius=1.0, core_radius=0.2)

        assert cored_isothermal.centre == (1.0, 1.0)
        assert cored_isothermal.axis_ratio == 1.0
        assert cored_isothermal.phi == 45.0
        assert cored_isothermal.einstein_radius == 1.0
        assert cored_isothermal.slope == 2.0
        assert cored_isothermal.core_radius == 0.2
        # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5
        assert cored_isothermal.einstein_radius_rescaled == 0.5

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(1, 1),
                                                       einstein_radius=1.0, core_radius=0.2)

        assert cored_isothermal.centre == (1.0, 1.0)
        assert cored_isothermal.axis_ratio == 1.0
        assert cored_isothermal.phi == 0.0
        assert cored_isothermal.einstein_radius == 1.0
        assert cored_isothermal.slope == 2.0
        assert cored_isothermal.core_radius == 0.2
        # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5
        assert cored_isothermal.einstein_radius_rescaled == 0.5

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()


class TestIsothermal(object):

    def test__constructor(self):
        isothermal = mp.EllipticalIsothermal(centre=(1, 1), axis_ratio=1.0, phi=45.0,
                                             einstein_radius=1.0)

        assert isothermal.centre == (1.0, 1.0)
        assert isothermal.axis_ratio == 1.0
        assert isothermal.phi == 45.0
        assert isothermal.einstein_radius == 1.0
        assert isothermal.slope == 2.0
        assert isothermal.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

        isothermal = mp.SphericalIsothermal(centre=(1, 1), einstein_radius=1.0)

        assert isothermal.centre == (1.0, 1.0)
        assert isothermal.axis_ratio == 1.0
        assert isothermal.phi == 0.0
        assert isothermal.einstein_radius == 1.0
        assert isothermal.slope == 2.0
        assert isothermal.einstein_radius_rescaled == 0.5  # (3 - slope) / (1 + axis_ratio) = (3 - 2) / (1 + 1) = 0.5

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
                                                     einstein_radius=1.0, slope=2.0, core_radius=0.0)

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

        assert sis.radius_where_average_convergence_in_circle_is_one == pytest.approx(2.0, 1e-4)

        sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.8, phi=0.0)
        assert sie.radius_where_average_convergence_in_circle_is_one == pytest.approx(1.0, 1e-4)

        sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=3.0, axis_ratio=0.5, phi=0.0)
        assert sie.radius_where_average_convergence_in_circle_is_one == pytest.approx(3.0, 1e-4)

        sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=8.0, axis_ratio=0.2, phi=0.0)
        assert sie.radius_where_average_convergence_in_circle_is_one == pytest.approx(8.0, 1e-4)

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()


class TestGeneralizedNFW(object):

    def test__constructor(self):

        # gnfw = mp.EllipticalGeneralizedNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=45.0,
        #                                    kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)
        #
        # assert gnfw.centre == (0.7, 1.0)
        # assert gnfw.axis_ratio == 0.7
        # assert gnfw.phi == 45.0
        # assert gnfw.kappa_s == 2.0
        # assert gnfw.inner_slope == 1.5
        # assert gnfw.scale_radius == 10.0

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.7, 1.0),
                                          kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)

        assert gnfw.centre == (0.7, 1.0)
        assert gnfw.axis_ratio == 1.0
        assert gnfw.phi == 0.0
        assert gnfw.kappa_s == 2.0
        assert gnfw.inner_slope == 1.5
        assert gnfw.scale_radius == 10.0

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

    def test__constructor(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.7, 1.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        assert truncated_nfw.centre == (0.7, 1.0)
        assert truncated_nfw.axis_ratio == 1.0
        assert truncated_nfw.phi == 0.0
        assert truncated_nfw.kappa_s == 2.0
        assert truncated_nfw.scale_radius == 10.0
        assert truncated_nfw.truncation_radius == 3.0

    def test__coord_function_f__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        # r > 1

        assert truncated_nfw.coord_func_f(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([0.604599, 0.435209]), 1.0e-4)

        # r < 1
        assert truncated_nfw.coord_func_f(grid_radius=np.array([0.5, 1.0/3.0])) == \
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

        assert truncated_nfw.coord_func_g(grid_radius=np.array([0.5, 1.0/3.0])) == \
               pytest.approx(np.array([0.69425, 0.97838]), 1.0e-4)

        # r == 1
        assert (truncated_nfw.coord_func_g(grid_radius=np.array([1.0, 1.0])) == np.array([1.0/3.0, 1.0/3.0])).all()

    def test__coord_function_h__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        assert truncated_nfw.coord_func_h(grid_radius=np.array([0.5, 3.0])) == \
               pytest.approx(np.array([0.134395, 0.840674]), 1.0e-4)

    def test__coord_function_k__correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.coord_func_k(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([-0.88137, -0.62514]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=4.0)

        assert truncated_nfw.coord_func_k(grid_radius=np.array([2.0, 3.0])) == \
               pytest.approx(np.array([-1.44363, -1.09861]), 1.0e-4)

    def test__coord_function_l__correct_values(self):

        # f_r = self.coord_func_f(r=r)
        # g_r = self.coord_func_g(r=r)
        # k_r = self.coord_func_k(r=r)
        #
        # coeff = np.divide(self.truncation_radius**2.0, (self.truncation_radius**2.0 + 1.0)**2.0)
        # term_1 = (self.truncation_radius**2.0 + 1.0)*g_r
        # term_2 = 2*f_r
        # term_3 = np.pi/(np.sqrt(self.truncation_radius ** 2.0 + r ** 2.0))
        # term_4 = ((self.truncation_radius**2.0 - 1.0) /
        #           (self.truncation_radius * (np.sqrt(self.truncation_radius ** 2.0 + r ** 2.0)))) * k_r

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        coeff = 0.16
        term_1 = 0.6590003532
        term_2 = 1.20919957615
        term_3 = 1.1107207345
        term_4 = -0.4674189301

        value = coeff*(term_1 + term_2 - term_3 + term_4)

        assert truncated_nfw.coord_func_l(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([value, value]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        coeff = 0.09
        term_1 = 1.3180007
        term_2 = 1.20919957615
        term_3 = 0.87132
        term_4 = -0.883647

        value = coeff*(term_1 + term_2 - term_3 + term_4)

        assert truncated_nfw.coord_func_l(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([value, value]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)
        coeff = 0.09
        term_1 = 0.705987
        term_2 = 0.870419
        term_3 = 0.74048
        term_4 = -0.553977

        value = coeff*(term_1 + term_2 - term_3 + term_4)

        assert truncated_nfw.coord_func_l(grid_radius=np.array([3.0, 3.0])) == \
               pytest.approx(np.array([value, value]), 1.0e-4)

    def test__coord_function_m__correct_values(self):

        # f_r = self.coord_func_f(r=r)
        # k_r = self.coord_func_k(r=r)
        #
        # coeff = (self.truncation_radius**2.0 / (self.truncation_radius**2.0 + 1.0) ** 2.0
        # term_1 = ((self.truncation_radius**2.0 + 2.0*r**2.0 - 1.0)*f_r)
        # term_2 = (np.pi*self.truncation_radius)
        # term_3 = ((self.truncation_radius**2.0 - 1.0) * np.log(self.truncation_radius))
        # term_4 = (np.sqrt(r**2.0 + self.truncation_radius**2.0)*((
        # (self.truncation_radius**2.0 - 1.0)/self.truncation_radius)*k_r - np.pi))

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=2.0)

        coeff = 0.16
        term_1 = 6.650597
        term_2 = 6.283185
        term_3 = 2.079441
        term_4 = -12.62511

        value = coeff*(term_1 + term_2 + term_3 + term_4)

        assert truncated_nfw.coord_func_m(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([value, value]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        coeff = 0.09
        term_1 = 9.673596
        term_2 = 9.4247779
        term_3 = 8.788898
        term_4 = -22.81458

        value = coeff*(term_1 + term_2 + term_3 + term_4)

        assert truncated_nfw.coord_func_m(grid_radius=np.array([2.0, 2.0])) == \
               pytest.approx(np.array([value, value]), 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0,
                                                 truncation_radius=3.0)

        coeff = 0.09
        term_1 = 11.31545
        term_2 = 9.4247779
        term_3 = 8.788898
        term_4 = -23.30025

        value = coeff*(term_1 + term_2 + term_3 + term_4)

        assert truncated_nfw.coord_func_m(grid_radius=np.array([3.0, 3.0])) == \
               pytest.approx(np.array([value, value]), 1.0e-4)

    def test__convergence_correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == \
               pytest.approx(2.0*0.046409642, 1.0e-4)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[1.0, 1.0]])) == \
               pytest.approx(2.0*0.10549515, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == \
               pytest.approx(6.0*0.046409642, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=5.0,
                                                 truncation_radius=2.0)

        assert truncated_nfw.convergence_from_grid(grid=np.array([[2.0, 0.0]])) == \
               pytest.approx(6.0*0.7042266, 1.0e-4)

    def test__deflections_correct_values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        # factor = (4.0 * kappa_s * scale_radius / (r / scale_radius))

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 1.0 * 1.0) / (2.0 / 1.0)
        assert deflections[0,0] == pytest.approx(factor*0.38209715, 1.0e-4)
        assert deflections[0,1] == pytest.approx(0.0, 1.0e-4)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[0.0, 2.0]]))

        assert deflections[0,0] == pytest.approx(0.0, 1.0e-4)
        assert deflections[0,1] == pytest.approx(factor*0.38209715, 1.0e-4)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

        factor = (4.0 * 1.0 * 1.0) / (np.sqrt(2) / 1.0)
        assert deflections[0,0] == pytest.approx((1.0/np.sqrt(2))*factor*0.3125838, 1.0e-4)
        assert deflections[0,1] == pytest.approx((1.0/np.sqrt(2))*factor*0.3125838, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0,
                                                 truncation_radius=2.0)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 2.0 * 1.0) / (2.0 / 1.0)
        assert deflections[0,0] == pytest.approx(factor*0.38209715, 1.0e-4)
        assert deflections[0,1] == pytest.approx(0.0, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0,
                                                 truncation_radius=2.0)

        deflections = truncated_nfw.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        factor = (4.0 * 1.0 * 4.0) / (2.0 / 4.0)
        assert deflections[0,0] == pytest.approx(factor*0.116951813, 1.0e-4)
        assert deflections[0,1] == pytest.approx(0.0, 1.0e-4)

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()

    def test__mass_at_truncation_radius__values(self):

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=1.0)

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(critical_surface_mass_density_arcsec=1.0,
                                                                            cosmic_average_mass_density_arcsec=1.0)

        assert mass_at_truncation_radius == pytest.approx(0.00009792581, 1.0e-5)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
                                                 truncation_radius=1.0)

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(critical_surface_mass_density_arcsec=2.0,
                                                                            cosmic_average_mass_density_arcsec=3.0)

        assert mass_at_truncation_radius == pytest.approx(0.00008789978, 1.0e-5)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0,
                                                 truncation_radius=1.0)

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(critical_surface_mass_density_arcsec=2.0,
                                                                            cosmic_average_mass_density_arcsec=3.0)

        assert mass_at_truncation_radius == pytest.approx(0.0000418378, 1.0e-5)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=8.0,
                                                 truncation_radius=4.0)

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(critical_surface_mass_density_arcsec=2.0,
                                                                            cosmic_average_mass_density_arcsec=3.0)

        assert mass_at_truncation_radius == pytest.approx(0.0000421512, 1.0e-4)

        truncated_nfw = mp.SphericalTruncatedNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=8.0,
                                                 truncation_radius=4.0)

        mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(critical_surface_mass_density_arcsec=2.0,
                                                                            cosmic_average_mass_density_arcsec=3.0)

        assert mass_at_truncation_radius == pytest.approx(0.00033636625, 1.0e-4)


class TestNFW(object):

    def test__constructor(self):

        nfw = mp.EllipticalNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=60.0, kappa_s=2.0,
                               scale_radius=10.0)

        assert nfw.centre == (0.7, 1.0)
        assert nfw.axis_ratio == 0.7
        assert nfw.phi == 60.0
        assert nfw.kappa_s == 2.0
        assert nfw.scale_radius == 10.0

        nfw = mp.SphericalNFW(centre=(0.7, 1.0), kappa_s=2.0, scale_radius=10.0)

        assert nfw.centre == (0.7, 1.0)
        assert nfw.axis_ratio == 1.0
        assert nfw.phi == 0.0
        assert nfw.kappa_s == 2.0
        assert nfw.scale_radius == 10.0

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

        nfw = mp.EllipticalNFW(centre=(0.3, 0.2), axis_ratio=0.7, phi=6.0, kappa_s=2.5, scale_radius=4.0)
        assert nfw.potential_from_grid(grid=np.array([[0.1625, 0.1625]])) == pytest.approx(0.05380, 1e-3)

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

    def test__radius_of_critical_curve_and_einstein_radius(self):

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=0.5, scale_radius=5.0)
        assert nfw.radius_where_average_convergence_in_circle_is_one == pytest.approx(2.76386, 1e-4)
        assert nfw.einstein_radius == pytest.approx(2.76386, 1e-4)

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), kappa_s=0.5, scale_radius=5.0, axis_ratio=0.8)
        assert nfw.radius_where_average_convergence_in_circle_is_one == pytest.approx(2.48747, 1e-4)
        assert nfw.einstein_radius == pytest.approx(2.48747, 1e-4)

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y != interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x != interp_deflections[:,1]).all()

    def test__rho_scale_radius_value(self):

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.rho_scale_radius(critical_surface_mass_density_arcsec=1.0) == pytest.approx(1.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
        assert nfw.rho_scale_radius(critical_surface_mass_density_arcsec=1.0) == pytest.approx(3.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
        assert nfw.rho_scale_radius(critical_surface_mass_density_arcsec=1.0) == pytest.approx(0.25, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.rho_scale_radius(critical_surface_mass_density_arcsec=5.0) == pytest.approx(5.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=3.0)
        assert nfw.rho_scale_radius(critical_surface_mass_density_arcsec=6.0) == pytest.approx(4.0, 1e-3)

    def test__delta_concentration_value(self):

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=1.0,
                                       cosmic_average_mass_density_arcsec=1.0) == pytest.approx(1.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=1.0,
                                       cosmic_average_mass_density_arcsec=1.0) == pytest.approx(3.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=1.0,
                                       cosmic_average_mass_density_arcsec=1.0) == pytest.approx(0.25, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=1.0) == pytest.approx(5.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=1.0) == pytest.approx(10.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=20.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=1.0) == pytest.approx(0.5, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=2.0) == pytest.approx(2.5, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=5.0) == pytest.approx(2.0, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=20.0)
        assert nfw.delta_concentration(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=5.0) == pytest.approx(0.1, 1e-3)

    def test__solve_concentration(self):

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        critical_surface_mass_density_arcsec = 1.0
        cosmic_average_mass_density_arcsec = 1.0

        concentration = nfw.concentration(critical_surface_mass_density_arcsec, cosmic_average_mass_density_arcsec)

        assert concentration == pytest.approx(0.0074263, 1.0e-4)

    def test__radius_at_200_times_cosmic_average_density(self):

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        concentration = nfw.concentration(critical_surface_mass_density_arcsec=5.0,
                                          cosmic_average_mass_density_arcsec=5.0)

        radius_200 = nfw.radius_at_200(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=5.0)

        assert radius_200 == concentration * 1.0

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=3.0)

        concentration = nfw.concentration(critical_surface_mass_density_arcsec=5.0,
                                          cosmic_average_mass_density_arcsec=8.0)

        radius_200 = nfw.radius_at_200(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=8.0)

        assert radius_200 == concentration * 3.0

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=6.0, scale_radius=2.0)

        concentration = nfw.concentration(critical_surface_mass_density_arcsec=5.0,
                                          cosmic_average_mass_density_arcsec=2.0)

        radius_200 = nfw.radius_at_200(critical_surface_mass_density_arcsec=5.0,
                                       cosmic_average_mass_density_arcsec=2.0)

        assert radius_200 == concentration * 2.0

    def test__mass_at_200(self):

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        mass_at_200 = nfw.mass_at_200(critical_surface_mass_density_arcsec=5.0,
                                      cosmic_average_mass_density_arcsec=8.0)

        # radius_200 = 0.004658
        # mass_200 = 200.0 * ((4*pi)/3)  * (0.004658 ** 3.0)

        assert mass_at_200 == pytest.approx(0.00067757, 1.0e-5)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=4.0)

        mass_at_200 = nfw.mass_at_200(critical_surface_mass_density_arcsec=50.0,
                                      cosmic_average_mass_density_arcsec=4.0)

        # radius_200 = 0.004658
        # mass_200 = 200.0 * ((4*pi)/3)  * (0.004658 ** 3.0)

        assert mass_at_200 == pytest.approx(18.57133, 1.0e-5)


class TestSersic(object):

    def test__constructor(self):
        sersic = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                     effective_radius=0.6, sersic_index=2.0, mass_to_light_ratio=1.0)

        assert sersic.centre == (0.0, 0.0)
        assert sersic.axis_ratio == 1.0
        assert sersic.phi == 0.0
        assert sersic.intensity == 1.0
        assert sersic.effective_radius == 0.6
        assert sersic.sersic_index == 2.0
        assert sersic.sersic_constant == pytest.approx(3.67206, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6
        assert sersic.mass_to_light_ratio == 1.0

        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=1.0,
                                    effective_radius=0.6, sersic_index=2.0, mass_to_light_ratio=1.0)

        assert sersic.centre == (0.0, 0.0)
        assert sersic.axis_ratio == 1.0
        assert sersic.phi == 0.0
        assert sersic.intensity == 1.0
        assert sersic.effective_radius == 0.6
        assert sersic.sersic_index == 2.0
        assert sersic.sersic_constant == pytest.approx(3.67206, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6
        assert sersic.mass_to_light_ratio == 1.0

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()


class TestExponential(object):

    def test__constructor(self):
        exponential = mp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                               effective_radius=0.6, mass_to_light_ratio=1.0)

        assert exponential.centre == (0.0, 0.0)
        assert exponential.axis_ratio == 1.0
        assert exponential.phi == 0.0
        assert exponential.intensity == 1.0
        assert exponential.effective_radius == 0.6
        assert exponential.sersic_index == 1.0
        assert exponential.sersic_constant == pytest.approx(1.678388, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6
        assert exponential.mass_to_light_ratio == 1.0

        exponential = mp.SphericalExponential(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.6,
                                              mass_to_light_ratio=1.0)

        assert exponential.centre == (0.0, 0.0)
        assert exponential.axis_ratio == 1.0
        assert exponential.phi == 0.0
        assert exponential.intensity == 1.0
        assert exponential.effective_radius == 0.6
        assert exponential.sersic_index == 1.0
        assert exponential.sersic_constant == pytest.approx(1.678388, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6
        assert exponential.mass_to_light_ratio == 1.0

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()


class TestDevVaucouleurs(object):

    def test__constructor(self):
        dev = mp.EllipticalDevVaucouleurs(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                          intensity=1.0,
                                          effective_radius=0.6, mass_to_light_ratio=1.0)

        assert dev.centre == (0.0, 0.0)
        assert dev.axis_ratio == 1.0
        assert dev.phi == 0.0
        assert dev.intensity == 1.0
        assert dev.effective_radius == 0.6
        assert dev.sersic_index == 4.0
        assert dev.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert dev.elliptical_effective_radius == 0.6
        assert dev.mass_to_light_ratio == 1.0

        dev = mp.SphericalDevVaucouleurs(centre=(0.0, 0.0), intensity=1.0,
                                         effective_radius=0.6, mass_to_light_ratio=1.0)

        assert dev.centre == (0.0, 0.0)
        assert dev.axis_ratio == 1.0
        assert dev.phi == 0.0
        assert dev.intensity == 1.0
        assert dev.effective_radius == 0.6
        assert dev.sersic_index == 4.0
        assert dev.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert dev.elliptical_effective_radius == 0.6
        assert dev.mass_to_light_ratio == 1.0

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()


class TestSersicMassRadialGradient(object):

    def test__constructor(self):
        sersic = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                   effective_radius=0.6, sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=2.0)

        assert sersic.centre == (0.0, 0.0)
        assert sersic.axis_ratio == 1.0
        assert sersic.phi == 0.0
        assert sersic.intensity == 1.0
        assert sersic.effective_radius == 0.6
        assert sersic.sersic_index == 2.0
        assert sersic.sersic_constant == pytest.approx(3.67206, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6
        assert sersic.mass_to_light_ratio == 1.0
        assert sersic.mass_to_light_gradient == 2.0

        sersic = mp.SphericalSersicRadialGradient(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.6,
                                                  sersic_index=2.0, mass_to_light_ratio=1.0, mass_to_light_gradient=2.0)

        assert sersic.centre == (0.0, 0.0)
        assert sersic.axis_ratio == 1.0
        assert sersic.phi == 0.0
        assert sersic.intensity == 1.0
        assert sersic.effective_radius == 0.6
        assert sersic.sersic_index == 2.0
        assert sersic.sersic_constant == pytest.approx(3.67206, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6
        assert sersic.mass_to_light_ratio == 1.0
        assert sersic.mass_to_light_gradient == 2.0

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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(self):

        sersic = mp.SphericalSersicRadialGradient(centre=(-0.7, 0.5), intensity=5.0, effective_radius=0.2, sersic_index=2.0,
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

        assert (interp_deflections_manual_y == interp_deflections[:,0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:,1]).all()


class TestMassSheet(object):

    def test_constructor(self):

        mass_sheet = mp.MassSheet(centre=(1.0, 1.0), kappa=2.0)

        assert mass_sheet.centre == (1.0, 1.0)
        assert mass_sheet.kappa == 2.0

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


class TestMassIntegral(object):

    def test__within_circle__no_critical_surface_mass_density__singular_isothermal_sphere__compare_to_analytic(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 2.0
        dimensionless_mass_integral = sis.mass_within_circle_in_angular_units(radius=integral_radius)
        assert math.pi * sis.einstein_radius * integral_radius == pytest.approx(dimensionless_mass_integral, 1e-3)

        sis = mp.SphericalIsothermal(einstein_radius=4.0)
        integral_radius = 4.0
        dimensionless_mass_integral = sis.mass_within_circle_in_angular_units(radius=integral_radius)
        assert math.pi * sis.einstein_radius * integral_radius == pytest.approx(dimensionless_mass_integral, 1e-3)

    def test__within_circle__no_critical_surface_mass_density__singular_isothermal__compare_to_grid(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)

        integral_radius = 1.0
        dimensionless_mass_total = 0.0

        xs = np.linspace(-1.5, 1.5, 40)
        ys = np.linspace(-1.5, 1.5, 40)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = math.sqrt(x ** 2 + y ** 2)

                if eta < integral_radius:
                    dimensionless_mass_total += sis.convergence_func(eta) * area

        dimensionless_mass_integral = sis.mass_within_circle_in_angular_units(radius=integral_radius)

        assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

    def test__mass_within_circle__critical_surface_mass_density_factor_multiplies(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 2.0
        mass_integral = sis.mass_within_circle_in_mass_units(radius=integral_radius, critical_surface_mass_density=2.0)
        assert 2.0 * math.pi * sis.einstein_radius * integral_radius == pytest.approx(mass_integral, 1e-3)

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 4.0
        mass_integral = sis.mass_within_circle_in_mass_units(radius=integral_radius, critical_surface_mass_density=8.0)
        assert 8.0 * math.pi * sis.einstein_radius * integral_radius == pytest.approx(mass_integral, 1e-3)

    def test__within_ellipse__no_critical_surface_mass_density__singular_isothermal_sphere__compare_circle_and_ellipse(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 2.0
        dimensionless_mass_integral_circle = sis.mass_within_circle_in_angular_units(radius=integral_radius)
        dimensionless_mass_integral_ellipse = sis.mass_within_ellipse_in_angular_units(major_axis=integral_radius)
        assert dimensionless_mass_integral_circle == dimensionless_mass_integral_ellipse

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)
        integral_radius = 2.0
        dimensionless_mass_integral_circle = sie.mass_within_circle_in_angular_units(radius=integral_radius)
        dimensionless_mass_integral_ellipse = sie.mass_within_ellipse_in_angular_units(major_axis=integral_radius)
        assert dimensionless_mass_integral_circle == dimensionless_mass_integral_ellipse * 2.0

    def test__within_ellipse__no_critical_surface_mass_density__singular_isothermal_ellipsoid__compare_to_grid(self):

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

        integral_radius = 0.5
        dimensionless_mass_tot = 0.0

        xs = np.linspace(-1.0, 1.0, 40)
        ys = np.linspace(-1.0, 1.0, 40)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = sie.grid_to_elliptical_radii(np.array([[x, y]]))

                if eta < integral_radius:
                    dimensionless_mass_tot += sie.convergence_func(eta) * area

        dimensionless_mass_integral = sie.mass_within_ellipse_in_angular_units(major_axis=integral_radius)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert dimensionless_mass_tot == pytest.approx(dimensionless_mass_integral, 0.1)

    def test__mass_within_ellipse__compare_to_grid__uses_critical_surface_mass_density(self):

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)

        integral_radius = 0.5
        dimensionless_mass_tot = 0.0

        xs = np.linspace(-1.0, 1.0, 40)
        ys = np.linspace(-1.0, 1.0, 40)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = sie.grid_to_elliptical_radii(np.array([[x, y]]))

                if eta < integral_radius:
                    dimensionless_mass_tot += sie.convergence_func(eta) * area

        mass_integral = sie.mass_within_ellipse_in_mass_units(major_axis=integral_radius,
                                                              critical_surface_mass_density=2.0)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert dimensionless_mass_tot == pytest.approx(0.5 * mass_integral, 0.1)

        mass_integral = sie.mass_within_ellipse_in_mass_units(major_axis=integral_radius,
                                                              critical_surface_mass_density=8.0)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert dimensionless_mass_tot == pytest.approx(0.125 * mass_integral, 0.1)


class TestDensityBetweenAnnuli(object):

    def test__circular_annuli__sis__analyic_density_agrees(self):

        einstein_radius=1.0
        sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=einstein_radius)

        inner_annuli_radius = 2.0
        inner_mass = math.pi * einstein_radius * inner_annuli_radius
        outer_annuli_radius = 3.0
        outer_mass = math.pi * einstein_radius * outer_annuli_radius

        density_between_annuli = sis.density_between_circular_annuli_in_angular_units(
            inner_annuli_radius=inner_annuli_radius, outer_annuli_radius=outer_annuli_radius)

        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (np.pi * inner_annuli_radius **2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(density_between_annuli, 1e-4)

    def test__circular_annuli__nfw_profile__compare_to_manual_mass_integrals(self):

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, kappa_s=1.0)

        inner_mass = nfw.mass_within_circle_in_angular_units(radius=1.0)
        outer_mass = nfw.mass_within_circle_in_angular_units(radius=2.0)

        density_between_annuli = nfw.density_between_circular_annuli_in_angular_units(inner_annuli_radius=1.0, outer_annuli_radius=2.0)

        annuli_area = (np.pi * 2.0 ** 2.0) - (np.pi * 1.0 **2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(density_between_annuli, 1e-4)