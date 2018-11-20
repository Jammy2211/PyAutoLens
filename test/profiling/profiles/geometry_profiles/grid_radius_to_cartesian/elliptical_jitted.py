import numba
import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools

from profiles import geometry_profiles


class EllipticalProfile(geometry_profiles.EllipticalProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """ Generic elliptical profiles class to contain functions shared by light and mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the origin of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalProfile, self).__init__(centre)
        self.axis_ratio = axis_ratio
        self.phi = phi

    def grid_angle_to_profile(self, theta_grid):
        theta_coordinate_to_profile = np.add(theta_grid, - self.phi_radians)
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

    def grid_radius_to_cartesian(self, grid, radius):
        theta_grid = np.arctan2(grid[:, 1], grid[:, 0])
        cos_theta, sin_theta = self.grid_angle_to_profile(theta_grid)
        return np.multiply(radius[:, None], np.vstack((cos_theta, sin_theta)).T)

    def grid_radius_to_cartesian_jitted(self, grid, radius):
        return self.grid_radius_to_cartesian_jit(grid, radius, self.phi_radians)

    @staticmethod
    @numba.jit(nopython=True)
    def grid_radius_to_cartesian_jit(grid, radius, phi_radians):
        cartesian = np.zeros(grid.shape)

        for i in range(grid.shape[0]):
            theta_coordinate_to_profile = np.arctan2(grid[i, 1], grid[i, 0]) - phi_radians
            cartesian[i, 0] = radius[i] * np.cos(theta_coordinate_to_profile)
            cartesian[i, 1] = radius[i] * np.sin(theta_coordinate_to_profile)

        return cartesian


geometry = EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)

sub_grid_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

lsst_radius = np.ones(lsst.coords.sub_grid_coords.shape[0])
euclid_radius = np.ones(euclid.coords.sub_grid_coords.shape[0])
hst_radius = np.ones(hst.coords.sub_grid_coords.shape[0])
hst_up_radius = np.ones(hst_up.coords.sub_grid_coords.shape[0])
ao_radius = np.ones(ao.coords.sub_grid_coords.shape[0])

assert (geometry.grid_radius_to_cartesian(grid=lsst.coords.sub_grid_coords, radius=lsst_radius) ==
        pytest.approx(geometry.grid_radius_to_cartesian_jitted(grid=lsst.coords.sub_grid_coords, radius=lsst_radius),
                      1e-4))


@tools.tick_toc_x20
def lsst_solution():
    geometry.grid_radius_to_cartesian_jitted(grid=lsst.coords.sub_grid_coords, radius=lsst_radius)


@tools.tick_toc_x20
def euclid_solution():
    geometry.grid_radius_to_cartesian_jitted(grid=euclid.coords.sub_grid_coords, radius=euclid_radius)


@tools.tick_toc_x20
def hst_solution():
    geometry.grid_radius_to_cartesian_jitted(grid=hst.coords.sub_grid_coords, radius=hst_radius)


@tools.tick_toc_x20
def hst_up_solution():
    geometry.grid_radius_to_cartesian_jitted(grid=hst_up.coords.sub_grid_coords, radius=hst_up_radius)


@tools.tick_toc_x20
def ao_solution():
    geometry.grid_radius_to_cartesian_jitted(grid=ao.coords.sub_grid_coords, radius=ao_radius)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
