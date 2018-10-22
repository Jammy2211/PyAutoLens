from autolens.profiles import light_profiles as lp

import numpy as np

# noinspection PyUnusedLocal
class MockLightProfile(lp.LightProfile):

    def intensities_from_grid(self, grid):
        return np.array([self.value])

    def __init__(self, value):
        self.value = value

    def intensities_from_grid_radii(self, grid_radii):
        return self.value

    def intensity_at_coordinates(self, coordinates):
        return self.value


class MockMassProfile(object):

    def __init__(self, value):
        self.value = value