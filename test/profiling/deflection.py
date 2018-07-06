import numpy as np
from src.analysis import galaxy
from src.profiles import mass_profiles
import time
import os
import numba
from numpy.testing import assert_almost_equal

path = os.path.dirname(os.path.realpath(__file__))


def load(name):
    return np.load("{}/{}.npy".format(path, name))


grid = load("grid")
deflection_result = load("deflection_result")
transformed_coordinates = load("transformed_coords")
elliptical_isothermal_deflection_result = load("elliptical_isothermal_deflection_result")

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


@tick_toc
def current_transform_and_back():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = mass_profile.transform_from_reference_frame(
            mass_profile.transform_to_reference_frame(coordinates=coordinate))

    assert (grid_values == transformed_coordinates).all()


@tick_toc
def new_transform_and_back():
    result = mass_profile.transform_grid_from_reference_frame(mass_profile.transform_grid_to_reference_frame(grid))

    assert_almost_equal(result, grid)


@tick_toc
def current_deflection_with_grid_transformation():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = mass_profile.deflections_at_coordinates(coordinates=coordinate)

    assert (grid_values == elliptical_isothermal_deflection_result).all()


@tick_toc
def new_deflection_with_grid_transformation():
    result = mass_profile.deflections_from_coordinate_grid(grid)

    for i in range(result.shape[0]):
        print("")
        print(result[i])
        print(elliptical_isothermal_deflection_result[i])
        print(result[i] - elliptical_isothermal_deflection_result[i])

    assert (result == elliptical_isothermal_deflection_result).all()


if __name__ == "__main__":
    current_deflection_with_grid_transformation()
    new_deflection_with_grid_transformation()
