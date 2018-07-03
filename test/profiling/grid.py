import numpy as np
from src.analysis import galaxy
from src.profiles import mass_profiles
import time
import os
import numba

path = os.path.dirname(os.path.realpath(__file__))


def load(name):
    return np.load("{}/{}.npy".format(path, name))


grid = load("grid")
deflection_result = load("deflection_result")
transformed_coordinates = load("transformed_coords")

# print(transformed_coordinates)

mass_profile = mass_profiles.EllipticalIsothermal(axis_ratio=0.9)

lens_galaxy = galaxy.Galaxy(spherical_mass_profile=mass_profile,
                            shear_mass_profile=mass_profiles.ExternalShear())

repeats = 100


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

        assert (grid_values == deflection_result).all()


@tick_toc
@numba.jit
def with_numba():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = lens_galaxy.deflections_at_coordinates(coordinates=coordinate)

    assert (grid_values == deflection_result).all()


@tick_toc
def current_transform_to_reference_frame():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = mass_profile.transform_to_reference_frame(coordinates=coordinate)

    assert (grid_values == transformed_coordinates).all()


@tick_toc
def new_transform_to_reference_frame():
    result = mass_profile.transform_grid_to_reference_frame(grid)

    assert (result == transformed_coordinates).all()


if __name__ == "__main__":
    current_transform_to_reference_frame()
    new_transform_to_reference_frame()
