import sys
import numpy as np

from profiling import profiling_data
from profiles import light_profiles
import time
import numba

class EllipticalSersicOriginal(light_profiles.EllipticalLightProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, settings=light_profiles.LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        """
        super(EllipticalSersicOriginal, self).__init__(centre, axis_ratio, phi, settings)
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def sersic_constant(self):
        """

        Returns
        -------
        sersic_constant: float
            A parameter, derived from sersic_index, that ensures that effective_radius always contains 50% of the light.
        """
        return (2 * self.sersic_index) - (1. / 3.) + (4. / (405. * self.sersic_index)) + (
                46. / (25515. * self.sersic_index ** 2)) + (131. / (1148175. * self.sersic_index ** 3)) - (
                       2194697. / (30690717750. * self.sersic_index ** 4))

    def grid_to_eccentric_radii(self, grid):
        return np.multiply(np.sqrt(self.axis_ratio),
                           np.sqrt(np.add(np.square(grid[:, 0]),
                                          np.square(np.divide(grid[:, 1], self.axis_ratio))))).view(np.ndarray)

    def intensity_from_grid(self, grid):
        return self.intensity_at_grid_radii(self.grid_to_eccentric_radii(grid))

    def intensity_at_grid_radii(self, grid_radii):
        return np.multiply(self.intensity, np.exp(
            np.multiply(-self.sersic_constant,
                        np.add(np.power(np.divide(grid_radii, self.effective_radius), 1. / self.sersic_index), -1))))

class EllipticalSersicJit(light_profiles.EllipticalLightProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, settings=light_profiles.LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        """
        super(EllipticalSersicJit, self).__init__(centre, axis_ratio, phi, settings)
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def sersic_constant(self):
        """

        Returns
        -------
        sersic_constant: float
            A parameter, derived from sersic_index, that ensures that effective_radius always contains 50% of the light.
        """
        return (2 * self.sersic_index) - (1. / 3.) + (4. / (405. * self.sersic_index)) + (
                46. / (25515. * self.sersic_index ** 2)) + (131. / (1148175. * self.sersic_index ** 3)) - (
                       2194697. / (30690717750. * self.sersic_index ** 4))

    def intensity_from_grid(self, grid):

        @numba.jit(nopython=True)
        def intensity_jit(grid, axis_ratio, intensity, effective_radius, sersic_index, sersic_constant):

            intensities = np.zeros(grid.shape[0])

            for i in range(grid.shape[0]):

                radius = np.multiply(np.sqrt(axis_ratio), np.sqrt(np.add(np.square(grid[i, 0]),
                                                                         np.square(np.divide(grid[i, 1], axis_ratio)))))

                intensities[i] = intensity * np.exp(-sersic_constant * (((radius / effective_radius) **
                                                                         (1. / sersic_index)) - 1))

            return intensities

        return intensity_jit(grid, self.axis_ratio, self.intensity, self.effective_radius, self.sersic_index,
                             self.sersic_constant)

subgrd_size=2

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=subgrd_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=subgrd_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=subgrd_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=subgrd_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=subgrd_size)

sersic_original = EllipticalSersicOriginal(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)

sersic_jit = EllipticalSersicJit(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
sersic_jit.intensity_from_grid(grid=lsst.coords.sub_grid_coords) # Run jit functions so their set-up isn't included in profiling run-time

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
    sersic_original.intensity_from_grid(grid=lsst.coords.sub_grid_coords)

@tick_toc
def lsst_jit_solution():
    sersic_jit.intensity_from_grid(grid=lsst.coords.sub_grid_coords)

@tick_toc
def euclid_original_solution():
    sersic_original.intensity_from_grid(grid=euclid.coords.sub_grid_coords)

@tick_toc
def euclid_jit_solution():
    sersic_jit.intensity_from_grid(grid=euclid.coords.sub_grid_coords)

@tick_toc
def hst_original_solution():
    sersic_original.intensity_from_grid(grid=hst.coords.sub_grid_coords)

@tick_toc
def hst_jit_solution():
    sersic_jit.intensity_from_grid(grid=hst.coords.sub_grid_coords)

@tick_toc
def hst_up_original_solution():
    sersic_original.intensity_from_grid(grid=hst_up.coords.sub_grid_coords)

@tick_toc
def hst_up_jit_solution():
    sersic_jit.intensity_from_grid(grid=hst_up.coords.sub_grid_coords)

@tick_toc
def ao_original_solution():
    sersic_original.intensity_from_grid(grid=ao.coords.sub_grid_coords)

@tick_toc
def ao_jit_solution():
    sersic_jit.intensity_from_grid(grid=ao.coords.sub_grid_coords)

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