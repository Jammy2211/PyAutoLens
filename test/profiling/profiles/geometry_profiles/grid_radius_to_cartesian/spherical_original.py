import numpy as np
from profiling import profiling_data
from profiling import tools

from profiles import geometry_profiles


class SphericalProfile(geometry_profiles.Profile):

    def __init__(self, centre=(0.0, 0.0)):
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
        super(SphericalProfile, self).__init__(centre)
        self.axis_ratio = 1.0
        self.phi = 0.0

    def grid_angle_to_profile(self, theta_grid):
        return np.cos(theta_grid), np.sin(theta_grid)

    def grid_radius_to_cartesian(self, grid, radius):
        theta_grid = np.arctan2(grid[:, 1], grid[:, 0])
        cos_theta, sin_theta = self.grid_angle_to_profile(theta_grid)
        return np.multiply(radius[:, None], np.vstack((cos_theta, sin_theta)).T)


geometry = SphericalProfile(centre=(0.0, 0.0))

sub_grid_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

lsst_radius = np.ones(lsst.grids.sub.shape[0])
euclid_radius = np.ones(euclid.grids.sub.shape[0])
hst_radius = np.ones(hst.grids.sub.shape[0])
hst_up_radius = np.ones(hst_up.grids.sub.shape[0])
ao_radius = np.ones(ao.grids.sub.shape[0])


@tools.tick_toc_x20
def lsst_solution():
    geometry.grid_radius_to_cartesian(grid=lsst.grids.sub, radius=lsst_radius)


@tools.tick_toc_x20
def euclid_solution():
    geometry.grid_radius_to_cartesian(grid=euclid.grids.sub, radius=euclid_radius)


@tools.tick_toc_x20
def hst_solution():
    geometry.grid_radius_to_cartesian(grid=hst.grids.sub, radius=hst_radius)


@tools.tick_toc_x20
def hst_up_solution():
    geometry.grid_radius_to_cartesian(grid=hst_up.grids.sub, radius=hst_up_radius)


@tools.tick_toc_x20
def ao_solution():
    geometry.grid_radius_to_cartesian(grid=ao.grids.sub, radius=ao_radius)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
