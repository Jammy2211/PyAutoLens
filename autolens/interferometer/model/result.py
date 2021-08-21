import numpy as np

import autoarray as aa
import autogalaxy as ag

from autolens.lens.model.result import ResultDataset


class ResultInterferometer(ResultDataset):
    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_interferometer_for_instance(instance=self.instance)

    @property
    def real_space_mask(self):
        return self.max_log_likelihood_fit.interferometer.real_space_mask

    @property
    def unmasked_model_visibilities(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_visibilities_of_planes(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_planes

    @property
    def unmasked_model_visibilities_of_planes_and_galaxies(self):
        fit = self.max_log_likelihood_fit
        return fit.unmasked_blurred_image_of_planes_and_galaxies

    def visibilities_for_galaxy(self, galaxy: ag.Galaxy) -> np.ndarray:
        """
        Parameters
        ----------
        galaxy
            A galaxy used in this search

        Returns
        -------
        ndarray or None
            A numpy arrays giving the model visibilities of that galaxy
        """
        return self.max_log_likelihood_fit.galaxy_model_visibilities_dict[galaxy]

    @property
    def visibilities_galaxy_dict(self) -> {str: ag.Galaxy}:
        """
        A dictionary associating galaxy names with model visibilities of those galaxies
        """
        return {
            galaxy_path: self.visibilities_for_galaxy(galaxy)
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @property
    def hyper_galaxy_visibilities_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy visibilities with their names.
        """

        hyper_galaxy_visibilities_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:
            hyper_galaxy_visibilities_path_dict[path] = self.visibilities_galaxy_dict[
                path
            ]

        return hyper_galaxy_visibilities_path_dict

    @property
    def hyper_model_visibilities(self):

        hyper_model_visibilities = aa.Visibilities.zeros(
            shape_slim=(self.max_log_likelihood_fit.visibilities.shape_slim,)
        )

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_visibilities += self.hyper_galaxy_visibilities_path_dict[path]

        return hyper_model_visibilities
