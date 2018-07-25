import sys
import numpy as np

from profiling import profiling_data
from profiles import geometry_profiles
import time
import numba
import pytest

class EllipticalProfile(geometry_profiles.EllipticalProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """ Generic elliptical profiles class to contain functions shared by light and mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalProfile, self).__init__(centre)
        self.axis_ratio = axis_ratio
        self.phi = phi

class SphericalProfile(geometry_profiles.Profile):

    def __init__(self, centre=(0.0, 0.0)):
        """ Generic elliptical profiles class to contain functions shared by light and mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The coordinates of the centre of the profiles
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

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=2)
geometry_elliptcal = EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)
geometry_spherical = SphericalProfile(centre=(0.0, 0.0))

lsst_radius = np.ones(lsst.coords.sub_grid_coords.shape[0])

assert geometry_elliptcal.grid_radius_to_cartesian(grid=lsst.coords.sub_grid_coords, radius=lsst_radius) == \
       pytest.approx(geometry_spherical.grid_radius_to_cartesian(grid=lsst.coords.sub_grid_coords, radius=lsst_radius), 1e-4)

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=2)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=2)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=2)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=2)

euclid_radius = np.ones(euclid.coords.sub_grid_coords.shape[0])
hst_radius = np.ones(hst.coords.sub_grid_coords.shape[0])
hst_up_radius = np.ones(hst_up.coords.sub_grid_coords.shape[0])
ao_radius = np.ones(ao.coords.sub_grid_coords.shape[0])

repeats = 1
def tick_toc(func):
    def wrapper():
        start = time.time()
        for _ in range(repeats):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))

    return wrapper

@tick_toc
def lsst_elliptical_solution():
    geometry_elliptcal.grid_radius_to_cartesian(grid=lsst.coords.sub_grid_coords, radius=lsst_radius)

@tick_toc
def lsst_spherical_solution():
    geometry_spherical.grid_radius_to_cartesian(grid=lsst.coords.sub_grid_coords, radius=lsst_radius)

@tick_toc
def euclid_elliptical_solution():
    geometry_elliptcal.grid_radius_to_cartesian(grid=euclid.coords.sub_grid_coords, radius=euclid_radius)

@tick_toc
def euclid_spherical_solution():
    geometry_spherical.grid_radius_to_cartesian(grid=euclid.coords.sub_grid_coords, radius=euclid_radius)

@tick_toc
def hst_elliptical_solution():
    geometry_elliptcal.grid_radius_to_cartesian(grid=hst.coords.sub_grid_coords, radius=hst_radius)

@tick_toc
def hst_spherical_solution():
    geometry_spherical.grid_radius_to_cartesian(grid=hst.coords.sub_grid_coords, radius=hst_radius)

@tick_toc
def hst_up_elliptical_solution():
    geometry_elliptcal.grid_radius_to_cartesian(grid=hst_up.coords.sub_grid_coords, radius=hst_up_radius)

@tick_toc
def hst_up_spherical_solution():
    geometry_spherical.grid_radius_to_cartesian(grid=hst_up.coords.sub_grid_coords, radius=hst_up_radius)

@tick_toc
def ao_elliptical_solution():
    geometry_elliptcal.grid_radius_to_cartesian(grid=ao.coords.sub_grid_coords, radius=ao_radius)

@tick_toc
def ao_spherical_solution():
    geometry_spherical.grid_radius_to_cartesian(grid=ao.coords.sub_grid_coords, radius=ao_radius)

if __name__ == "__main__":
    lsst_elliptical_solution()
    lsst_spherical_solution()

    print()

    euclid_elliptical_solution()
    euclid_spherical_solution()

    print()

    hst_elliptical_solution()
    hst_spherical_solution()

    print()

    hst_up_elliptical_solution()
    hst_up_spherical_solution()

    print()

    ao_elliptical_solution()
    ao_spherical_solution()