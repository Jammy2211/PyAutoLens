import inspect
import numpy as np
import os
import time
from src.profiles import light_profiles, geometry_profiles
from numpy.testing import assert_almost_equal

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
        result = None
        for _ in range(repeats):
            result = func(*args, **kwargs)
        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))
        return result

    return wrapper


def classic_method(func):
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = func(coordinate)

    return grid_values


def grid_to_eccentric_radius():
    profile = geometry_profiles.EllipticalProfile()

    @tick_toc
    def classic_grid_to_eccentric_radius():
        grid_values = np.zeros(grid.shape[0])

        for pixel_no, coordinate in enumerate(grid):
            grid_values[pixel_no] = profile.coordinates_to_eccentric_radius(coordinate)

        return grid_values

    @tick_toc
    def new_grid_to_eccentric_radius():
        return profile.grid_to_eccentric_radius(grid)

    classic = classic_grid_to_eccentric_radius()
    new = new_grid_to_eccentric_radius()

    assert_almost_equal(classic, new)


@all_light_profiles
def save_light_profile_intensity(profile):
    result = classic_method(profile.intensity_at_coordinates)
    save(profile.__class__.__name__, result)


if __name__ == "__main__":
    grid_to_eccentric_radius()
