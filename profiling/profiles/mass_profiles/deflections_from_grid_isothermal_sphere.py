import numpy as np

from profiling import profiling_data
from profiles import light_profiles
import time
import numba

class SphericalIsothermalOriginal(light_profiles.EllipticalLightProfile):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(SphericalIsothermalOriginal, self).__init__(centre, 1.0, 0.0, einstein_radius)
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
        grid : mask.CoordinateGrid
            The grid of coordinates the deflection angles are computed on.
        """
        return np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled)

class SphericalIsothermalJit(light_profiles.EllipticalLightProfile):

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        einstein_radius : float
            Einstein radius of power-law mass profiles
        """

        super(SphericalIsothermalJit, self).__init__(centre, 1.0, 0.0, einstein_radius)
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
        grid : mask.CoordinateGrid
            The grid of coordinates the deflection angles are computed on.
        """

        @numba.jit(nopython=True)
        def deflections_from_grid_jit(grid, einstein_radius_rescaled):

            deflections = np.zeros(grid.shape[0])

            for i in range(deflections.shape[0]):

                deflections[i] = 2.0 *  einstein_radius_rescaled

            return deflections

        return deflections_from_grid_jit(grid, self.einstein_radius_rescaled)

subgrd_size=4

sis_original = SphericalIsothermalOriginal(centre=(0.0, 0.0), einstein_radius=1.4)
sis_jit = SphericalIsothermalJit(centre=(0.0, 0.0), einstein_radius=1.4)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=subgrd_size)

assert (sis_original.deflections_from_grid(grid=lsst.coords.sub_grid_coords) ==
        sis_jit.deflections_from_grid(grid=lsst.coords.sub_grid_coords)).all()

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=subgrd_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=subgrd_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=subgrd_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=subgrd_size)

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
    sis_original.deflections_from_grid(grid=lsst.coords.sub_grid_coords)

@tick_toc
def lsst_jit_solution():
    sis_jit.deflections_from_grid(grid=lsst.coords.sub_grid_coords)

@tick_toc
def euclid_original_solution():
    sis_original.deflections_from_grid(grid=euclid.coords.sub_grid_coords)

@tick_toc
def euclid_jit_solution():
    sis_jit.deflections_from_grid(grid=euclid.coords.sub_grid_coords)

@tick_toc
def hst_original_solution():
    sis_original.deflections_from_grid(grid=hst.coords.sub_grid_coords)

@tick_toc
def hst_jit_solution():
    sis_jit.deflections_from_grid(grid=hst.coords.sub_grid_coords)

@tick_toc
def hst_up_original_solution():
    sis_original.deflections_from_grid(grid=hst_up.coords.sub_grid_coords)

@tick_toc
def hst_up_jit_solution():
    sis_jit.deflections_from_grid(grid=hst_up.coords.sub_grid_coords)

@tick_toc
def ao_original_solution():
    sis_original.deflections_from_grid(grid=ao.coords.sub_grid_coords)

@tick_toc
def ao_jit_solution():
    sis_jit.deflections_from_grid(grid=ao.coords.sub_grid_coords)

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