import math

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.model.profiles import mass_profiles as mp

from test.unit.mock.model import mock_cosmology

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestMassSheet(object):
    def test__constructor_and_units(self):

        mass_sheet = mp.MassSheet(centre=(1.0, 2.0), kappa=2.0)

        assert mass_sheet.centre == (1.0, 2.0)
        assert isinstance(mass_sheet.centre[0], dim.Length)
        assert isinstance(mass_sheet.centre[1], dim.Length)
        assert mass_sheet.centre[0].unit == "arcsec"
        assert mass_sheet.centre[1].unit == "arcsec"

        assert mass_sheet.kappa == 2.0
        assert isinstance(mass_sheet.kappa, float)

    def test__convergence__correct_values(self):
        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        convergence = mass_sheet.convergence_from_grid(grid=np.array([[1.0, 0.0]]))

        assert convergence[0] == pytest.approx(1.0, 1e-3)

        convergence = mass_sheet.convergence_from_grid(
            grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
        )

        assert convergence[0] == pytest.approx(1.0, 1e-3)
        assert convergence[1] == pytest.approx(1.0, 1e-3)
        assert convergence[2] == pytest.approx(1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=-3.0)

        convergence = mass_sheet.convergence_from_grid(
            grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
        )

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
        deflections_0 = mass_sheet_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        deflections_1 = mass_sheet_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert deflections_0[0, 0] == pytest.approx(-deflections_1[0, 0], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(-deflections_1[0, 1], 1e-5)

        mass_sheet_0 = mp.MassSheet(centre=(0.0, 0.0))
        mass_sheet_1 = mp.MassSheet(centre=(0.0, 0.0))
        deflections_0 = mass_sheet_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        deflections_1 = mass_sheet_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

    def test__multiple_coordinates_in__multiple_coordinates_out(self):
        mass_sheet = mp.MassSheet(centre=(1.0, 2.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(
            grid=np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
        )
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)
        assert deflections[1, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[1, 1] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(
            grid=np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]])
        )
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        assert deflections[1, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[1, 1] == pytest.approx(2.0, 1e-3)

        assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 1] == pytest.approx(1.0, 1e-3)

        assert deflections[3, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[3, 1] == pytest.approx(2.0, 1e-3)

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        mass_sheet = mp.MassSheet()

        convergence = mass_sheet.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        potential = mass_sheet.potential_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert potential.shape == (2, 2)

        deflections = mass_sheet.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


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
        deflections = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.01625, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.01625, 1e-3)

        shear = mp.ExternalShear(magnitude=0.2, phi=75.0)
        deflections = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        shear = mp.ExternalShear()

        convergence = shear.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        potential = shear.potential_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert potential.shape == (2, 2)

        deflections = shear.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)
