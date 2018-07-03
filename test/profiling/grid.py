import numpy as np
from src.analysis import galaxy
from src.profiles import mass_profiles
import time
import os
import numba

path = os.path.dirname(os.path.realpath(__file__))
grid = np.load("{}/grid.npy".format(path))
grid_result = np.load("{}/grid_result.npy".format(path))

lens_galaxy = galaxy.Galaxy(spherical_mass_profile=mass_profiles.EllipticalIsothermal(axis_ratio=0.9),
                            shear_mass_profile=mass_profiles.ExternalShear())

repeats = 10


def tick_toc(func):
    def wrapper():
        start = time.time()
        result = None
        for _ in range(repeats):
            result = func()
        assert (result == grid_result).all()
        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))

    return wrapper


@tick_toc
def current_solution():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = lens_galaxy.deflections_at_coordinates(coordinates=coordinate)

    return grid_values


@tick_toc
@numba.jit
def with_numba():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = lens_galaxy.deflections_at_coordinates(coordinates=coordinate)

    return grid_values


if __name__ == "__main__":
    current_solution()
    with_numba()
