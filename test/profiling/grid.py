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
        for _ in range(repeats):
            func()
        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))

    return wrapper


@tick_toc
def current_solution():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = lens_galaxy.deflections_at_coordinates(coordinates=coordinate)

    assert (grid_values == grid_result).all()


@tick_toc
@numba.jit
def with_numba():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = lens_galaxy.deflections_at_coordinates(coordinates=coordinate)

    assert (grid_values == grid_result).all()


if __name__ == "__main__":
    current_solution()
    with_numba()
