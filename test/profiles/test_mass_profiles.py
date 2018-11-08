import math

import numpy as np
import pytest

from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


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

    def test__surface_density_correct_values(self):
        cored_power_law = mp.SphericalCoredPowerLaw(centre=(1, 1), einstein_radius=1.0, slope=2.2, core_radius=0.1)
        assert cored_power_law.surface_density_func(radius=1.0) == pytest.approx(0.39762, 1e-4)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0,
                                                     slope=2.3, core_radius=0.2)
        assert cored_power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.45492, 1e-3)

        cored_power_law = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=2.0,
                                                     slope=1.7, core_radius=0.2)
        assert cored_power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(1.3887, 1e-3)

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
        assert cored_power_law_0.surface_density_from_grid(
            grid=np.array([[1.0, 1.0]])) == cored_power_law_1.surface_density_from_grid(grid=np.array([[0.0, 0.0]]))

        cored_power_law_0 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        cored_power_law_1 = mp.SphericalCoredPowerLaw(centre=(0.0, 0.0))
        assert cored_power_law_0.surface_density_from_grid(
            grid=np.array([[1.0, 0.0]])) == cored_power_law_1.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

        cored_power_law_0 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        cored_power_law_1 = mp.EllipticalCoredPowerLaw(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert cored_power_law_0.surface_density_from_grid(
            grid=np.array([[1.0, 0.0]])) == cored_power_law_1.surface_density_from_grid(grid=np.array([[0.0, 1.0]]))

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
        assert cored_power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0], [0.0, 1.0]]))[0] == pytest.approx(
            0.45492, 1e-3)
        assert cored_power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0], [0.0, 1.0]]))[1] == pytest.approx(
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

        assert elliptical.surface_density_from_grid(grid) == pytest.approx(spherical.surface_density_from_grid(grid),
                                                                           1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)


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

    def test__surface_density_correct_values(self):
        isothermal = mp.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0)
        assert isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(0.5, 1e-3)

        isothermal = mp.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.2)
        assert isothermal.surface_density_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.4, 1e-3)

        power_law = mp.SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.2)
        assert power_law.surface_density_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.4, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                          einstein_radius=1.0, slope=2.3)
        assert power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.466666, 1e-3)

        power_law = mp.EllipticalPowerLaw(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                          einstein_radius=2.0, slope=1.7)
        assert power_law.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(1.4079, 1e-3)

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

        assert elliptical.surface_density_from_grid(grid) == pytest.approx(spherical.surface_density_from_grid(grid),
                                                                           1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)


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

    def test__surface_density_correct_values(self):
        cored_isothermal = mp.SphericalCoredIsothermal(centre=(1, 1), einstein_radius=1., core_radius=0.1)
        assert cored_isothermal.surface_density_func(radius=1.0) == pytest.approx(0.49752, 1e-4)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(1, 1), einstein_radius=1.0, core_radius=0.1)
        assert cored_isothermal.surface_density_func(radius=1.0) == pytest.approx(0.49752, 1e-4)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.2)
        assert cored_isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(0.49029, 1e-3)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(0.0, 0.0), einstein_radius=2.0, core_radius=0.2)
        assert cored_isothermal.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(2.0 * 0.49029,
                                                                                                        1e-3)

        cored_isothermal = mp.SphericalCoredIsothermal(centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.2)
        assert cored_isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.49029, 1e-3)

        # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.
        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0,
                                                        einstein_radius=1.0, core_radius=0.2)
        assert cored_isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
            0.49029 * 1.33333, 1e-3)

        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0,
                                                        einstein_radius=2.0, core_radius=0.2)
        assert cored_isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 0.49029,
                                                                                                        1e-3)

        # for axis_ratio = 1.0, the factor is 1/2
        # for axis_ratio = 0.5, the factor is 1/(1.5)
        # So the change in the value is 0.5 / (1/1.5) = 1.0 / 0.75
        # axis ratio changes only einstein_rescaled, so wwe can use the above value and times by 1.0/1.5.
        cored_isothermal = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0,
                                                        core_radius=0.2)
        assert cored_isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
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

        assert elliptical.surface_density_from_grid(grid) == pytest.approx(spherical.surface_density_from_grid(grid),
                                                                           1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)


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

    def test__surface_density__correct_values(self):
        # eta = 1.0
        # kappa = 0.5 * 1.0 ** 1.0
        isothermal = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)
        assert isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.5 * 2.0, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0)
        assert isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.5, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=2.0)
        assert isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.5 * 2.0, 1e-3)

        isothermal = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0)
        assert isothermal.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.66666, 1e-3)

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

        assert elliptical.surface_density_from_grid(grid) == pytest.approx(spherical.surface_density_from_grid(grid),
                                                                           1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)


class TestGeneralizedNFW(object):

    def test__constructor(self):
        gnfw = mp.EllipticalGeneralizedNFW(centre=(0.7, 1.0), axis_ratio=0.7, phi=45.0,
                                           kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)

        assert gnfw.centre == (0.7, 1.0)
        assert gnfw.axis_ratio == 0.7
        assert gnfw.phi == 45.0
        assert gnfw.kappa_s == 2.0
        assert gnfw.inner_slope == 1.5
        assert gnfw.scale_radius == 10.0

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.7, 1.0),
                                          kappa_s=2.0, inner_slope=1.5, scale_radius=10.0)

        assert gnfw.centre == (0.7, 1.0)
        assert gnfw.axis_ratio == 1.0
        assert gnfw.phi == 0.0
        assert gnfw.kappa_s == 2.0
        assert gnfw.inner_slope == 1.5
        assert gnfw.scale_radius == 10.0

    def test__coord_func_x_above_1(self):
        assert mp.EllipticalNFW.coord_func(2.0) == pytest.approx(0.60459, 1e-3)

        assert mp.EllipticalNFW.coord_func(0.5) == pytest.approx(1.5206919, 1e-3)

        assert mp.EllipticalNFW.coord_func(1.0) == 1.0

    def test__surface_density_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0)
        assert gnfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.30840, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, inner_slope=1.5, scale_radius=1.0)
        assert gnfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.30840 * 2, 1e-3)

        gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.5,
                                           phi=90.0, inner_slope=1.5, scale_radius=1.0)
        assert gnfw.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.30840, 1e-3)

        gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=2.0, axis_ratio=0.5,
                                           phi=90.0, inner_slope=1.5, scale_radius=1.0)
        assert gnfw.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.30840 * 2, 1e-3)

    def test__potential_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0)
        assert gnfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]])) == pytest.approx(0.00920, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=8.0)
        assert gnfw.potential_from_grid(grid=np.array([[0.1625, 0.1875]])) == pytest.approx(0.17448, 1e-3)

        gnfw = mp.EllipticalGeneralizedNFW(centre=(1.0, 1.0), kappa_s=5.0, axis_ratio=0.5,
                                           phi=100.0, inner_slope=1.0, scale_radius=10.0)
        assert gnfw.potential_from_grid(grid=np.array([[2.0, 2.0]])) == pytest.approx(2.4718, 1e-4)

    def test__deflections_correct_values(self):
        gnfw = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0)
        defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.43501, 1e-3)
        assert defls[0, 1] == pytest.approx(0.37701, 1e-3)

        gnfw = mp.SphericalGeneralizedNFW(centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0)
        defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-9.31254, 1e-3)
        assert defls[0, 1] == pytest.approx(-3.10418, 1e-3)

        gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), kappa_s=1.0, axis_ratio=0.3,
                                           phi=100.0, inner_slope=0.5, scale_radius=8.0)
        defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.26604, 1e-3)
        assert defls[0, 1] == pytest.approx(0.58988, 1e-3)

        gnfw = mp.EllipticalGeneralizedNFW(centre=(0.3, 0.2), kappa_s=2.5, axis_ratio=0.5,
                                           phi=100.0, inner_slope=1.5, scale_radius=4.0)
        defls = gnfw.deflections_from_grid(grid=np.array([[0.1875, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-5.99032, 1e-3)
        assert defls[0, 1] == pytest.approx(-4.02541, 1e-3)

    def test__surfce_density__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0))
        assert gnfw_0.surface_density_from_grid(grid=np.array([[1.0, 1.0]])) == gnfw_1.surface_density_from_grid(
            grid=np.array([[0.0, 0.0]]))

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        assert gnfw_0.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.surface_density_from_grid(
            grid=np.array([[0.0, 1.0]]))

        gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert gnfw_0.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.surface_density_from_grid(
            grid=np.array([[0.0, 1.0]]))

    def test__potential__change_geometry(self):
        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(1.0, 1.0))
        assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 1.0]])) == gnfw_1.potential_from_grid(
            grid=np.array([[0.0, 0.0]]))

        gnfw_0 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        gnfw_1 = mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))
        assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.potential_from_grid(
            grid=np.array([[0.0, 1.0]]))

        gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert gnfw_0.potential_from_grid(grid=np.array([[1.0, 0.0]])) == gnfw_1.potential_from_grid(
            grid=np.array([[0.0, 1.0]]))

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

        gnfw_0 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0,
                                             inner_slope=1.5, scale_radius=1.0)
        gnfw_1 = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, kappa_s=1.0,
                                             inner_slope=1.5, scale_radius=1.0)
        defls_0 = gnfw_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        defls_1 = gnfw_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert defls_0[0, 0] == pytest.approx(defls_1[0, 1], 1e-5)
        assert defls_0[0, 1] == pytest.approx(defls_1[0, 0], 1e-5)

    def test__compare_to_nfw(self):
        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0, scale_radius=5.0)
        gnfw = mp.EllipticalGeneralizedNFW(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, kappa_s=1.0,
                                           inner_slope=1.0, scale_radius=5.0)

        assert nfw.potential_from_grid(grid) == pytest.approx(gnfw.potential_from_grid(grid), 1e-3)
        assert nfw.potential_from_grid(grid) == pytest.approx(gnfw.potential_from_grid(grid), 1e-3)
        assert nfw.deflections_from_grid(grid) == pytest.approx(gnfw.deflections_from_grid(grid), 1e-3)
        assert nfw.deflections_from_grid(grid) == pytest.approx(gnfw.deflections_from_grid(grid), 1e-3)

    def test__spherical_and_elliptical_match(self):
        elliptical = mp.EllipticalGeneralizedNFW(centre=(0.1, 0.2), axis_ratio=1.0, phi=0.0, kappa_s=2.0,
                                                 inner_slope=1.5, scale_radius=3.0)
        spherical = mp.SphericalGeneralizedNFW(centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0)

        assert elliptical.surface_density_from_grid(grid) == pytest.approx(spherical.surface_density_from_grid(grid),
                                                                           1e-4)
        assert elliptical.potential_from_grid(grid) == pytest.approx(spherical.potential_from_grid(grid), 1e-4)
        assert elliptical.deflections_from_grid(grid) == pytest.approx(spherical.deflections_from_grid(grid), 1e-4)


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

    def test__surface_density_correct_values(self):
        # r = 2.0 (> 1.0)
        # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
        # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141
        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.surface_density_from_grid(grid=np.array([[2.0, 0.0]])) == pytest.approx(0.263600141, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
        assert nfw.surface_density_from_grid(grid=np.array([[0.5, 0.0]])) == pytest.approx(1.388511, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)
        assert nfw.surface_density_from_grid(grid=np.array([[0.5, 0.0]])) == pytest.approx(2.0 * 1.388511, 1e-3)

        nfw = mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)
        assert nfw.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(1.388511, 1e-3)

        nfw = mp.EllipticalNFW(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, kappa_s=1.0, scale_radius=1.0)
        assert nfw.surface_density_from_grid(grid=np.array([[0.25, 0.0]])) == pytest.approx(1.388511, 1e-3)

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

    def test__surface_density_correct_values(self):
        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=3.0, effective_radius=2.0, sersic_index=2.0,
                                    mass_to_light_ratio=1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]])) == pytest.approx(4.90657319276, 1e-3)

        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=6.0, effective_radius=2.0, sersic_index=2.0,
                                    mass_to_light_ratio=1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]])) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = mp.SphericalSersic(centre=(0.0, 0.0), intensity=3.0, effective_radius=2.0, sersic_index=2.0,
                                    mass_to_light_ratio=2.0)
        assert sersic.surface_density_from_grid(grid=np.array([[0.0, 1.5]])) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                     sersic_index=2.0, mass_to_light_ratio=1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(5.38066670129, 1e-3)

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
        assert sersic_0.surface_density_from_grid(grid=np.array([[1.0, 1.0]])) == sersic_1.surface_density_from_grid(
            grid=np.array([[0.0, 0.0]]))

        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(0.0, 0.0))
        assert sersic_0.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == sersic_1.surface_density_from_grid(
            grid=np.array([[0.0, 1.0]]))

        sersic_0 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert sersic_0.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == sersic_1.surface_density_from_grid(
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

        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))

    def test__from_light_profile(self):
        light_exponential = lp.EllipticalExponential(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0,
                                                     intensity=5.0, effective_radius=0.2)
        mass_exponential = mp.EllipticalExponential.from_exponential_light_profile(light_exponential,
                                                                                   mass_to_light_ratio=1.)
        defls = mass_exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(0.90493, 1e-3)
        assert defls[0, 1] == pytest.approx(0.62569, 1e-3)

        light_dev = lp.EllipticalDevVaucouleurs(centre=(0.4, 0.2), axis_ratio=0.9, phi=10.0, intensity=2.0,
                                                effective_radius=0.8)
        mass_dev = mp.EllipticalDevVaucouleurs.from_dev_vaucouleurs_light_profile(light_dev, mass_to_light_ratio=3.)
        defls = mass_dev.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(-24.528, 1e-3)
        assert defls[0, 1] == pytest.approx(-3.37605, 1e-3)


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

    def test__surface_density_correct_values(self):
        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(4.9047, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(4.8566, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=4.0, effective_radius=3.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                               mass_to_light_ratio=2.0)
        assert exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = mp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                               mass_to_light_ratio=1.0)
        assert exponential.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(4.8566, 1e-3)

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

        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))


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

    def test__surface_density_correct_values(self):
        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                          mass_to_light_ratio=1.0)
        assert dev.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(5.6697, 1e-3)

        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                          mass_to_light_ratio=1.0)
        assert dev.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(7.4455, 1e-3)

        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=4.0, effective_radius=3.0,
                                          mass_to_light_ratio=1.0)
        assert dev.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 7.4455, 1e-3)

        dev = mp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0,
                                          mass_to_light_ratio=2.0)
        assert dev.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 7.4455, 1e-3)

        sersic = mp.SphericalDevVaucouleurs(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.6,
                                            mass_to_light_ratio=1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.351797, 1e-3)

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

        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)

        np.testing.assert_almost_equal(elliptical.deflections_from_grid(grid), spherical.deflections_from_grid(grid))


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

    def test__surface_density_correct_values(self):
        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
        sersic = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                   effective_radius=0.6, sersic_index=4.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.6 * 0.351797, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75
        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=-1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[1.5, 0.0]])) == pytest.approx(0.75 * 4.90657319276,
                                                                                              1e-3)

        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=1.0, phi=0.0, intensity=6.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=-1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[1.5, 0.0]])) == pytest.approx(
            2.0 * 0.75 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=2.0,
                                                   mass_to_light_gradient=-1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[1.5, 0.0]])) == pytest.approx(
            2.0 * 0.75 * 4.90657319276, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836
        sersic = mp.EllipticalSersicRadialGradient(axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0,
                                                   sersic_index=2.0, mass_to_light_ratio=1.0,
                                                   mass_to_light_gradient=1.0)
        assert sersic.surface_density_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(
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

    def test__from_light_profile__deflection_angles_unchanged(self):
        light_sersic = lp.EllipticalSersic(centre=(-0.4, -0.2), axis_ratio=0.8, phi=110.0, intensity=5.0,
                                           effective_radius=0.2, sersic_index=2.0)
        mass_sersic = mp.EllipticalSersicRadialGradient.from_profile(light_sersic, mass_to_light_ratio=1.0,
                                                                     mass_to_light_gradient=0.0)
        defls = mass_sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert defls[0, 0] == pytest.approx(1.1446, 1e-3)
        assert defls[0, 1] == pytest.approx(0.79374, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                       effective_radius=1.0, sersic_index=4.0,
                                                       mass_to_light_ratio=1.0, mass_to_light_gradient=1.0)
        spherical = mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0,
                                                      sersic_index=4.0, mass_to_light_ratio=1.0,
                                                      mass_to_light_gradient=1.0)
        assert (elliptical.surface_density_from_grid(grid) == spherical.surface_density_from_grid(grid)).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        assert (elliptical.deflections_from_grid(grid) == spherical.deflections_from_grid(grid)).all()


class TestExternalShear(object):

    def test__surface_density_returns_zeros(self):
        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        surface_density = shear.surface_density_from_grid(grid=np.array([0.1]))
        assert (surface_density == np.array([0.0])).all()

        shear = mp.ExternalShear(magnitude=0.1, phi=45.0)
        surface_density = shear.surface_density_from_grid(grid=np.array([0.1, 0.2, 0.3]))
        assert (surface_density == np.array([0.0, 0.0, 0.0])).all()

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

    def test__within_circle__singular_isothermal_sphere__compare_to_analytic1(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 2.0
        dimensionless_mass_integral = sis.dimensionless_mass_within_circle(radius=integral_radius)
        assert math.pi * sis.einstein_radius * integral_radius == pytest.approx(dimensionless_mass_integral, 1e-3)

        sis = mp.SphericalIsothermal(einstein_radius=4.0)
        integral_radius = 4.0
        dimensionless_mass_integral = sis.dimensionless_mass_within_circle(radius=integral_radius)
        assert math.pi * sis.einstein_radius * integral_radius == pytest.approx(dimensionless_mass_integral, 1e-3)

    def test__within_circle__singular_isothermal__compare_to_grid(self):

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
                    dimensionless_mass_total += sis.surface_density_func(eta) * area

        dimensionless_mass_integral = sis.dimensionless_mass_within_circle(radius=integral_radius)

        assert dimensionless_mass_total == pytest.approx(dimensionless_mass_integral, 0.02)

    def test__mass_within_circle__conversion_factor_multiplies(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 2.0
        mass_integral = sis.mass_within_circle(radius=integral_radius, conversion_factor=2.0)
        assert 2.0 * math.pi * sis.einstein_radius * integral_radius == pytest.approx(mass_integral, 1e-3)

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 4.0
        mass_integral = sis.mass_within_circle(radius=integral_radius, conversion_factor=8.0)
        assert 8.0 * math.pi * sis.einstein_radius * integral_radius == pytest.approx(mass_integral, 1e-3)

    def test__within_ellipse__singular_isothermal_sphere__compare_circle_and_ellipse(self):

        sis = mp.SphericalIsothermal(einstein_radius=2.0)
        integral_radius = 2.0
        dimensionless_mass_integral_circle = sis.dimensionless_mass_within_circle(radius=integral_radius)
        dimensionless_mass_integral_ellipse = sis.dimensionless_mass_within_ellipse(major_axis=integral_radius)
        assert dimensionless_mass_integral_circle == dimensionless_mass_integral_ellipse

        sie = mp.EllipticalIsothermal(einstein_radius=2.0, axis_ratio=0.5, phi=0.0)
        integral_radius = 2.0
        dimensionless_mass_integral_circle = sie.dimensionless_mass_within_circle(radius=integral_radius)
        dimensionless_mass_integral_ellipse = sie.dimensionless_mass_within_ellipse(major_axis=integral_radius)
        assert dimensionless_mass_integral_circle == dimensionless_mass_integral_ellipse * 2.0

    def test__within_ellipse__singular_isothermal_ellipsoid__compare_to_grid(self):

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
                    dimensionless_mass_tot += sie.surface_density_func(eta) * area

        dimensionless_mass_integral = sie.dimensionless_mass_within_ellipse(major_axis=integral_radius)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert dimensionless_mass_tot == pytest.approx(dimensionless_mass_integral, 0.1)

    def test__mass_within_ellipse__compare_to_grid__uses_conversion_factor(self):

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
                    dimensionless_mass_tot += sie.surface_density_func(eta) * area

        mass_integral = sie.mass_within_ellipse(major_axis=integral_radius, conversion_factor=2.0)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert dimensionless_mass_tot == pytest.approx(0.5 * mass_integral, 0.1)

        mass_integral = sie.mass_within_ellipse(major_axis=integral_radius, conversion_factor=8.0)

        # Large errors required due to cusp at center of SIE - can get to errors of 0.01 for a 400 x 400 grid.
        assert dimensionless_mass_tot == pytest.approx(0.125 * mass_integral, 0.1)