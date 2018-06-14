import numpy as np

# TODO : For each Galaxy, this class computes its contribution map and scaled noise map.
# TODO : Typically, we will have 2 galaxies (a lens and source), thus we will have 2 contribution maps and noise maps.
# TODO : For each Galaxy, we also have 3 hyper-parameters (contribution_factor, noise_factor, noise_power).

# TODO : Thus, our model mapper needs to be clever and adjust the number of free parameters based no the number of.
# TODO : galaxies, not the constructor arguments.

class HyperImage(object):

    def __init__(self, background_noise_scale=0.0, contribution_factors=(0.0, 0.0), noise_factors=(0.0, 0.0),
                 noise_powers=(1.0, 1.0)):
        """Class for scaling the noise of different components in an image, primarily the different galaxies (e.g. the \
        lens, source).

        Parameters
        -----------
        background_noise_scale : int
            The value by which the background noise is increased (electrons per second).
        contribution_factors : tuple
            Factors which adjust the profile of each galaxy contribution map.
        noise_factors : tuple
            The factor by which the noise is increased, for each galaxy.
        noise_powers : tuple
            The power to which each galaxy scaled noise map is raised.
        """

        self.background_noise_scale = background_noise_scale
        self.contribution_factors = contribution_factors
        self.noise_factors = noise_factors
        self.noise_powers = noise_powers

    def compute_all_galaxy_contributions(self, galaxy_images, minimum_values):
        """Compute the contribution map of all galaxies, which represent the fraction of flux in each pixel that \
        galaxy can be attributed to contain

        Parameters
        -----------
        galaxy_images : [ndarray]
            A model image of each galaxy (e.g the lens light profile or source reconstruction) computed from a
            previous analysis.
        minimum_values : [int]
            The minimum fractional flux a pixel must contain to not be rounded to 0.
        """

        model_image = np.sum(galaxy_images, axis=0)
        indexes = list(range(len(galaxy_images)))

        return list(map(lambda image, index, minimum_value :
                        self.compute_galaxy_contributions(model_image, image, index, minimum_value),
                        galaxy_images, indexes, minimum_values))

    def compute_galaxy_contributions(self, model_image, galaxy_image, galaxy_index, minimum_value):
        """Compute the contribution map of a galaxy, which represents the fraction of flux in each pixel that \
        galaxy can be attributed to contain.

        This is computed by dividing that galaxy's flux by the total flux in that pixel, and then scaling by the \
        maximum flux such that the contribution map ranges between 0 and 1.

        Parameters
        -----------
        galaxy_images : [ndarray]
            A model image of each galaxy (e.g the lens light profile or source reconstruction) computed from a
            previous analysis.
        minimum_values : [int]
            The minimum fractional flux a pixel must contain to not be rounded to 0.
        """

        contributions = galaxy_image / (model_image + self.contribution_factors[galaxy_index])
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def compute_scaled_noise(self, grid_baseline_noise, grid_background_noise, galaxy_contributions):
        """Compute a scaled noise map from the baseline noise map. This scales each galaxy component individually \
        using their galaxy contribution map and sums their scaled noise maps with the baseline and background noise maps.

        Parameters
        -----------
        grid_baseline_noise : ndarray or auto_lens.imaging.grids.GridBaselineNoise
            The 1D grid of baseline noise values.
        grid_background_noise : ndarray or auto_lens.imaging.grids.GridBackgroundNoise
            The 1D grid of background noise values.
        galaxy_contributions : ndarray
            The galaxy contribution map.
        """

        indexes = list(range(len(galaxy_contributions)))

        galaxy_scaled_noise = np.sum(np.asarray(list(map(lambda galaxy_contribution, index :
                                     self.compute_galaxy_scaled_noise(grid_baseline_noise, galaxy_contribution, index),
                                     galaxy_contributions, indexes))), axis=0)

        return grid_baseline_noise + self.background_noise_scale*grid_background_noise + galaxy_scaled_noise

    def compute_galaxy_scaled_noise(self, grid_baseline_noise, galaxy_contributions, galaxy_index):
        """Compute a scaled galaxy noise map from a baseline nosie map.

        This uses the galaxy contribution map with their noise scaling hyper-parameters.

        Parameters
        -----------
        grid_baseline_noise : ndarray or auto_lens.imaging.grids.GridBaselineNoise
            The 1D grid of baseline noise values.
        grid_background_noise : ndarray or auto_lens.imaging.grids.GridBackgroundNoise
            The 1D grid of background noise values.
        galaxy_contributions : ndarray
            The galaxy contribution map.
        """
        return self.noise_factors[galaxy_index] * (grid_baseline_noise * galaxy_contributions) ** \
               self.noise_powers[galaxy_index]