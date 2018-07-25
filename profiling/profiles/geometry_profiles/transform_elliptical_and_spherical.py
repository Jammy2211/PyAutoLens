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

    def transform_grid_to_reference_frame(self, grid):
        transformed = np.subtract(grid, self.centre)
        return transformed.view(geometry_profiles.TransformedGrid)


lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=2)
geometry_elliptcal = EllipticalProfile(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0)
geometry_spherical = SphericalProfile(centre=(0.0, 0.0))

assert geometry_elliptcal.transform_grid_to_reference_frame(grid=lsst.coords.sub_grid_coords) == \
       pytest.approx(geometry_spherical.transform_grid_to_reference_frame(grid=lsst.coords.sub_grid_coords), 1e-4)

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=2)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=2)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=2)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=2)

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
def lsst_original_solution():
    geometry_elliptcal.transform_grid_to_reference_frame(grid=lsst.coords.sub_grid_coords)

@tick_toc
def lsst_jit_solution():
    geometry_spherical.transform_grid_to_reference_frame(grid=lsst.coords.sub_grid_coords)

@tick_toc
def euclid_original_solution():
    geometry_elliptcal.transform_grid_to_reference_frame(grid=euclid.coords.sub_grid_coords)

@tick_toc
def euclid_jit_solution():
    geometry_spherical.transform_grid_to_reference_frame(grid=euclid.coords.sub_grid_coords)

@tick_toc
def hst_original_solution():
    geometry_elliptcal.transform_grid_to_reference_frame(grid=hst.coords.sub_grid_coords)

@tick_toc
def hst_jit_solution():
    geometry_spherical.transform_grid_to_reference_frame(grid=hst.coords.sub_grid_coords)

@tick_toc
def hst_up_original_solution():
    geometry_elliptcal.transform_grid_to_reference_frame(grid=hst_up.coords.sub_grid_coords)

@tick_toc
def hst_up_jit_solution():
    geometry_spherical.transform_grid_to_reference_frame(grid=hst_up.coords.sub_grid_coords)

@tick_toc
def ao_original_solution():
    geometry_elliptcal.transform_grid_to_reference_frame(grid=ao.coords.sub_grid_coords)

@tick_toc
def ao_jit_solution():
    geometry_spherical.transform_grid_to_reference_frame(grid=ao.coords.sub_grid_coords)

# @tick_toc
# def jitted_solution():
#     kernel_convolver.convolve_array_jit(data)

if __name__ == "__main__":
    lsst_original_solution()
    lsst_jit_solution()

    print()

    euclid_original_solution()
    euclid_jit_solution()

    print()

    hst_original_solution()
    hst_jit_solution()

    print()

    hst_up_original_solution()
    hst_up_jit_solution()

    print()

    ao_original_solution()
    ao_jit_solution()