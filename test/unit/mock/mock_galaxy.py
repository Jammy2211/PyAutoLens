import numpy as np


class MockGalaxy(object):

    def __init__(self, value, shape=1):
        self.value = value
        self.shape = shape

    def intensities_from_grid(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    def convergence_from_grid(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    def potential_from_grid(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    def deflections_from_grid(self, grid):
        return np.full(shape=(self.shape, 2), fill_value=self.value)


class MockHyperGalaxy(object):

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

    def contributions_from_model_image_and_galaxy_image(self, model_image, galaxy_image, minimum_value):
        contributions = galaxy_image / (model_image + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def hyper_noise_from_contributions(self, noise_map, contributions):
        return self.noise_factor * (noise_map * contributions) ** self.noise_power
