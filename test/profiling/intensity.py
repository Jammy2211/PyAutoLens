import inspect
import numpy as np
import os
import time
from src.profiles import light_profiles
import logging
from numpy.testing import assert_almost_equal

logging.level = logging.DEBUG

path = os.path.dirname(os.path.realpath(__file__))

repeats = 100


def load(name):
    return np.load("{}/intensity_data/{}.npy".format(path, name))


def save(name, array):
    return np.save("{}/intensity_data/{}.npy".format(path, name), array)


grid = load("grid")


def all_light_profiles(func):
    light_profile_classes = [value for value in light_profiles.__dict__.values()
                             if inspect.isclass(value)
                             and issubclass(value, light_profiles.LightProfile)
                             and value not in (light_profiles.LightProfile, light_profiles.EllipticalLightProfile)]

    def wrapper():
        instances = map(lambda cls: cls(), light_profile_classes)
        for instance in instances:
            func(instance)

    return wrapper


def tick_toc(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        for _ in range(repeats):
            func(*args, **kwargs)
        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))
        return diff

    return wrapper


def classic_method(func):
    grid_values = np.zeros(grid.shape[0])

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = func(coordinate)

    return grid_values


@all_light_profiles
def save_light_profile_intensity(profile):
    result = classic_method(profile.intensity_at_coordinates)
    save(profile.__class__.__name__, result)


@all_light_profiles
def test_deflections_from_coordinate_grid(instance):
    name = instance.__class__.__name__
    example = load(name)
    result = None
    try:
        result = instance.intensity_from_coordinate_grid(grid)
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


@all_light_profiles
def tick_toc_comparison_for_profile(instance):
    print("")
    print(instance.__class__.__name__)

    @tick_toc
    def old_method():
        grid_values = np.zeros(grid.shape)

        for pixel_no, coordinate in enumerate(grid):
            grid_values[pixel_no] = instance.intensity_at_coordinates(coordinate)

    @tick_toc
    def new_method():
        instance.intensity_from_coordinate_grid(grid)

    old = old_method()
    new = new_method()
    print("x faster: {}".format(old / new))


if __name__ == "__main__":
    tick_toc_comparison_for_profile()
