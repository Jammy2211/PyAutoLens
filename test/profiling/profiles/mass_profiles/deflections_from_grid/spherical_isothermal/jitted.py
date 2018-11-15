import numba
import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools

from profiles import geometry_profiles


class SphericalIsothermal(geometry_profiles.SphericalProfile):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the origin of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(SphericalIsothermal, self).__init__(centre)
        self.einstein_radius = einstein_radius
        self.slope = 2.0
        self.axis_ratio = 1.0
        self.phi = 0.0

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
        return np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled)

    def deflections_from_grid_jitted(self, grid):
        """
        Calculate the deflection angles at a given set of gridded coordinates.

        Parameters
        ----------
        grid : masks.ImageGrid
            The grid of coordinates the deflection angles are computed on.
        """
        return self.deflections_from_grid_jit(grid, self.einstein_radius_rescaled)

    @staticmethod
    @numba.jit(nopython=True)
    def deflections_from_grid_jit(grid, einstein_radius_rescaled):
        deflections = np.zeros(grid.shape[0])
        for i in range(deflections.shape[0]):
            deflections[i] = 2.0 * einstein_radius_rescaled
        return deflections


sis = SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.4)

subgrd_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=subgrd_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=subgrd_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=subgrd_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=subgrd_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=subgrd_size)

assert (sis.deflections_from_grid(grid=lsst.coords.sub_grid_coords) ==
        pytest.approx(sis.deflections_from_grid_jitted(grid=lsst.coords.sub_grid_coords), 1e-4))


@tools.tick_toc_x10
def lsst_solution():
    sis.deflections_from_grid_jitted(grid=lsst.coords.sub_grid_coords)


@tools.tick_toc_x10
def euclid_solution():
    sis.deflections_from_grid_jitted(grid=euclid.coords.sub_grid_coords)


@tools.tick_toc_x10
def hst_solution():
    sis.deflections_from_grid_jitted(grid=hst.coords.sub_grid_coords)


@tools.tick_toc_x10
def hst_up_solution():
    sis.deflections_from_grid_jitted(grid=hst_up.coords.sub_grid_coords)


@tools.tick_toc_x10
def ao_solution():
    sis.deflections_from_grid_jitted(grid=ao.coords.sub_grid_coords)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
