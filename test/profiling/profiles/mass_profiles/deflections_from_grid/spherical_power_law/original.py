import numpy as np
from profiling import profiling_data
from profiling import tools

from profiles import geometry_profiles


class SphericalPowerLaw(geometry_profiles.SphericalProfile):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (x,y) coordinates of the origin of the profile.
        einstein_radius : float
            Einstein radius of power-law mass profiles
        slope : float
            power-law density slope of mass profiles
        """
        super(SphericalPowerLaw, self).__init__(centre)
        self.einstein_radius = einstein_radius
        self.slope = slope

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (self.slope - 1)

    def deflections_from_grid(self, grid):
        eta = self.grid_to_radius(grid)
        deflection_r = 2.0 * self.einstein_radius_rescaled * \
                       np.divide(np.power(eta, (3.0 - self.slope)), np.multiply((3.0 - self.slope), eta))
        return deflection_r


power_law = SphericalPowerLaw(centre=(0.0, 0.0), einstein_radius=1.4, slope=2.0)

subgrd_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=subgrd_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=subgrd_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=subgrd_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=subgrd_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=subgrd_size)


@tools.tick_toc_x10
def lsst_solution():
    power_law.deflections_from_grid(grid=lsst.coords.sub_grid_coords)


@tools.tick_toc_x10
def euclid_solution():
    power_law.deflections_from_grid(grid=euclid.coords.sub_grid_coords)


@tools.tick_toc_x10
def hst_solution():
    power_law.deflections_from_grid(grid=hst.coords.sub_grid_coords)


@tools.tick_toc_x10
def hst_up_solution():
    power_law.deflections_from_grid(grid=hst_up.coords.sub_grid_coords)


@tools.tick_toc_x10
def ao_solution():
    power_law.deflections_from_grid(grid=ao.coords.sub_grid_coords)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
