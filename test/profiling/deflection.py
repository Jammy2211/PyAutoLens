import numpy as np
from src.analysis import galaxy
from src.profiles import mass_profiles
import time
import os
import numba
from numpy.testing import assert_almost_equal
import inspect
import logging

path = os.path.dirname(os.path.realpath(__file__))


def load(name):
    return np.load("{}/data/{}.npy".format(path, name))


grid = load("grid")
deflection_result = load("deflection_result")
transformed_coordinates = load("transformed_coords")
elliptical_isothermal_deflection_result = load("elliptical_isothermal_deflection_result")

mass_profile = mass_profiles.EllipticalIsothermal(axis_ratio=0.9)

lens_galaxy = galaxy.Galaxy(spherical_mass_profile=mass_profile,
                            shear_mass_profile=mass_profiles.ExternalShear())

repeats = 1


def tick_toc(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        for _ in range(repeats):
            func(*args, **kwargs)

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
def current_deflection_elliptical_isothermal():
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = mass_profile.deflections_at_coordinates(coordinates=coordinate)

    assert (grid_values == elliptical_isothermal_deflection_result).all()


@tick_toc
def new_deflection_elliptical_isothermal():
    result = mass_profile.deflections_from_coordinate_grid(grid)

    assert (result == elliptical_isothermal_deflection_result).all()


def all_mass_profiles(func):
    mass_profile_classes = [value for value in mass_profiles.__dict__.values()
                            if inspect.isclass(value)
                            and issubclass(value, mass_profiles.MassProfile)
                            and value not in (mass_profiles.MassProfile, mass_profiles.EllipticalMassProfile)]

    def wrapper():
        instances = map(lambda cls: cls(), mass_profile_classes)
        for instance in instances:
            tick_toc(func)(instance)

    return wrapper


@all_mass_profiles
def test_deflections_at_coordinates(instance):
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = instance.deflections_at_coordinates(coordinates=coordinate)


@all_mass_profiles
def test_deflections_from_coordinate_grid(instance):
    name = instance.__class__.__name__
    example = load(name)
    try:
        result = instance.deflections_from_coordinate_grid(grid)
        if (result == example).all():
            logging.info("{} gives the correct result")
        else:
            logging.warning("{} does not give the correct result".format(name))

    except AttributeError:
        logging.warning("{} has no deflections_from_coordinate_grid function".format(name))
    except NotImplementedError as e:
        logging.exception(e)
    except ZeroDivisionError:
        logging.warning("{} throws a zero division error".format(name))


if __name__ == "__main__":
    test_deflections_from_coordinate_grid()
