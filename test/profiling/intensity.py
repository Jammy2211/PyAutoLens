import inspect
import numpy as np
import os
from src.profiles import light_profiles

path = os.path.dirname(os.path.realpath(__file__))


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


def classic_method(profile):
    grid_values = np.zeros(grid.shape)

    for pixel_no, coordinate in enumerate(grid):
        grid_values[pixel_no] = profile.intensity_at_coordinates(coordinate)

    return grid_values


@all_light_profiles
def save_light_profile_intensity(profile):
    result = classic_method(profile)
    save(profile.__class__.__name__, result)


if __name__ == "__main__":
    save_light_profile_intensity()
