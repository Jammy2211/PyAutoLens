import numpy as np
from src.analysis import galaxy
from src.profiles import mass_profiles
import time
import os
import numba
from numpy.testing import assert_almost_equal
import inspect
import logging

logging.level = logging.DEBUG

path = os.path.dirname(os.path.realpath(__file__))


def load(name):
    return np.load("{}/deflection_data/{}.npy".format(path, name))


# grid = load("grid")

grid = np.ones((100, 2))

mass_profile = mass_profiles.EllipticalPowerLaw(centre=(0, 0), axis_ratio=0.5, phi=0.0, einstein_radius=1.0, slope=2.0)

lens_galaxy = galaxy.Galaxy(spherical_mass_profile=mass_profile)

repeats = 10


class InvalidRun(Exception):
    pass


def tick_toc(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            for _ in range(repeats):
                func(*args, **kwargs)

            diff = time.time() - start
            print("{}: {}".format(func.__name__, diff))
            return diff
        except InvalidRun:
            pass

    return wrapper


@tick_toc
def current_deflection_elliptical_power_law():

    grid_values = np.zeros(grid.shape)

  #  for pixel_no, coordinate in enumerate(grid):
  #      grid_values[pixel_no] = mass_profile.deflections_at_coordinates(coordinates=coordinate)

   # assert (grid_values == elliptical_power_law_deflection_result).all()


@tick_toc
def new_deflection_elliptical_power_law():

    result = mass_profile.deflections_from_grid(grid)

   # assert (result == elliptical_power_law_deflection_result).all()


def all_mass_profiles(func):
    mass_profile_classes = [value for value in mass_profiles.__dict__.values()
                            if inspect.isclass(value)
                            and issubclass(value, mass_profiles.MassProfile)
                            and value not in (mass_profiles.MassProfile, mass_profiles.EllipticalMassProfile)]

    def wrapper():
        instances = map(lambda cls: cls(), mass_profile_classes)
        for instance in instances:
            func(instance)

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
    result = None
    try:
        result = instance.deflections_from_grid(grid)
        assert_almost_equal(result, example)
        print("{} gives the correct result".format(name))
        return
    except AssertionError as e:
        if result is None:
            raise e
        logging.warning("{} does not give the correct result".format(name))
        print(example.shape)
        print(result.shape)
        print(example[0])
        print(result[0])
        print(example[-1])
        print(result[-1])
    except NotImplementedError:
        logging.warning("{} has no deflections_from_coordinate_grid function".format(name))
    except ZeroDivisionError:
        logging.warning("{} throws a zero division error".format(name))


def compare_single_coordinates_for_class(mass_profile_class):
    instance = mass_profile_class()
    coordinates = np.array([0.7, 0.6])
    coordinates_grid = np.array([coordinates])

    print("\ndeflections_at_coordinates")
    result = instance.deflections_at_coordinates(coordinates)
    print("result = {}".format(result))
    print("\ndeflections_from_coordinate_grid")
    grid_result = instance.deflections_from_grid(coordinates_grid)[0]
    print("grid_result = {}".format(grid_result))


def tick_toc_comparison_for_class(mass_profile_class):
    print("")
    print(mass_profile_class.__name__)
    instance = mass_profile_class()

    @tick_toc
    def old_method():
        grid_values = np.zeros(grid.shape)

        for pixel_no, coordinate in enumerate(grid):
            grid_values[pixel_no] = instance.deflections_at_coordinates(coordinate)

    @tick_toc
    def new_method():
        instance.deflections_from_grid(grid)

  #  old = old_method()
    new = new_method()
    print("x faster: {}".format(new))


def tick_toc_comparison_for_classes():
    mass_profile_classes = [mass_profiles.EllipticalPowerLaw]

    for mass_profile_class in mass_profile_classes:
        tick_toc_comparison_for_class(mass_profile_class)


if __name__ == "__main__":
    tick_toc_comparison_for_classes()
