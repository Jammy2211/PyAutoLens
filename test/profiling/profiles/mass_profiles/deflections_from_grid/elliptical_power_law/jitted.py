import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools
from scipy.integrate import quad

from profiles import mass_profiles


class EllipticalPowerLaw(mass_profiles.EllipticalMassProfile, mass_profiles.MassProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (x,y) coordinates of the origin of the profile.
        axis_ratio : float
            Elliptical mass profile's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis
        einstein_radius : float
            Einstein radius of power-law mass profile.
        slope : float
            power-law density slope of mass profile.
        """

        super(EllipticalPowerLaw, self).__init__(centre, axis_ratio, phi)
        super(mass_profiles.MassProfile, self).__init__()

        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = 0.0

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(grid, npow, index, axis_ratio, einstein_radius_rescaled, slope, core_radius):
            deflection_grid = np.zeros(grid.shape[0])

            for i in range(grid.shape[0]):
                deflection_grid[i] = self.axis_ratio * grid[i, index] * quad(self.deflection_func, a=0.0, b=1.0,
                                                                             args=(grid[i, 0], grid[i, 1], npow,
                                                                                   axis_ratio,
                                                                                   einstein_radius_rescaled, slope,
                                                                                   core_radius))[0]

            return deflection_grid

        deflection_x = calculate_deflection_component(grid, 0.0, 0, self.axis_ratio, self.einstein_radius_rescaled,
                                                      self.slope, self.core_radius)
        deflection_y = calculate_deflection_component(grid, 1.0, 1, self.axis_ratio, self.einstein_radius_rescaled,
                                                      self.slope, self.core_radius)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_x, deflection_y)).T))

    def deflections_from_grid_jitted(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """

        def calculate_deflection_component(grid, npow, index, axis_ratio, einstein_radius_rescaled, slope, core_radius):
            deflection_grid = np.zeros(grid.shape[0])

            for i in range(grid.shape[0]):
                deflection_grid[i] = self.axis_ratio * grid[i, index] * quad(self.deflection_func, a=0.0, b=1.0,
                                                                             args=(grid[i, 0], grid[i, 1], npow,
                                                                                   axis_ratio,
                                                                                   einstein_radius_rescaled, slope,
                                                                                   core_radius))[0]

            return deflection_grid

        deflection_x = calculate_deflection_component(grid, 0.0, 0, self.axis_ratio, self.einstein_radius_rescaled,
                                                      self.slope, self.core_radius)
        deflection_y = calculate_deflection_component(grid, 1.0, 1, self.axis_ratio, self.einstein_radius_rescaled,
                                                      self.slope, self.core_radius)

        return self.rotate_grid_from_profile(np.multiply(1.0, np.vstack((deflection_x, deflection_y)).T))

    @staticmethod
    @mass_profiles.jit_integrand
    def deflection_func(u, x, y, npow, axis_ratio, einstein_radius_rescaled, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return einstein_radius_rescaled * eta_u ** (-(slope - 1)) / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


sis = EllipticalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.4, slope=2.0)

subgrd_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=subgrd_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=subgrd_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=subgrd_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=subgrd_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=subgrd_size)

assert (sis.deflections_from_grid(grid=lsst.coords.sub_grid_coords) ==
        pytest.approx(sis.deflections_from_grid_jitted(grid=lsst.coords.sub_grid_coords), 1e-4))


@tools.tick_toc_x1
def lsst_solution():
    sis.deflections_from_grid_jitted(grid=lsst.coords.sub_grid_coords)


@tools.tick_toc_x1
def euclid_solution():
    sis.deflections_from_grid_jitted(grid=euclid.coords.sub_grid_coords)


@tools.tick_toc_x1
def hst_solution():
    sis.deflections_from_grid_jitted(grid=hst.coords.sub_grid_coords)


@tools.tick_toc_x1
def hst_up_solution():
    sis.deflections_from_grid_jitted(grid=hst_up.coords.sub_grid_coords)


@tools.tick_toc_x1
def ao_solution():
    sis.deflections_from_grid_jitted(grid=ao.coords.sub_grid_coords)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
