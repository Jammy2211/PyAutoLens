import numpy as np

import autofit as af
import autoarray as aa
from autoastro.galaxy import galaxy as g
from autolens.pipeline.phase import data


class Result(data.Result):
    @property
    def most_likely_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.constant
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.constant
        )

        return self.analysis.lens_imaging_fit_for_tracer(
            tracer=self.most_likely_tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def unmasked_model_image(self):
        return self.most_likely_fit.unmasked_blurred_profile_image

    @property
    def unmasked_model_image_of_planes(self):
        return self.most_likely_fit.unmasked_blurred_profile_image_of_planes

    @property
    def unmasked_model_image_of_planes_and_galaxies(self):
        fit = self.most_likely_fit
        return fit.unmasked_blurred_profile_image_of_planes_and_galaxies

    def image_2d_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
        """
        Parameters
        ----------
        galaxy
            A galaxy used in this phase

        Returns
        -------
        ndarray or None
            A numpy array giving the model image of that galaxy
        """
        return self.most_likely_fit.galaxy_model_image_2d_dict[galaxy]

    @property
    def image_galaxy_1d_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """

        image_1d_dict = {}

        for galaxy, galaxy_image_2d in self.image_galaxy_2d_dict.items():
            image_1d_dict[galaxy] = self.mask.mapping.array_1d_from_array_2d(
                array_2d=galaxy_image_2d
            )

        return image_1d_dict

    @property
    def image_galaxy_2d_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """
        return {
            galaxy_path: self.image_2d_for_galaxy(galaxy)
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @property
    def hyper_galaxy_image_1d_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy images with their names.
        """

        hyper_minimum_percent = af.conf.instance.general.get(
            "hyper", "hyper_minimum_percent", float
        )

        hyper_galaxy_image_1d_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            galaxy_image_1d = self.image_galaxy_1d_dict[path]

            if not np.all(galaxy_image_1d == 0):
                minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image_1d)
                galaxy_image_1d[
                    galaxy_image_1d < minimum_galaxy_value
                    ] = minimum_galaxy_value

            hyper_galaxy_image_1d_path_dict[path] = galaxy_image_1d

        return hyper_galaxy_image_1d_path_dict

    @property
    def hyper_galaxy_image_2d_path_dict(self):
        """
        A dictionary associating 2D hyper_galaxies galaxy images with their names.
        """

        hyper_galaxy_image_2d_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:
            hyper_galaxy_image_2d_path_dict[
                path
            ] = self.mask.mapping.scaled_array_2d_from_array_1d(
                array_1d=self.hyper_galaxy_image_1d_path_dict[path]
            )

        return hyper_galaxy_image_2d_path_dict

    def binned_image_1d_dict_from_binned_grid(self, binned_grid) -> {str: g.Galaxy}:
        """
        A dictionary associating 1D binned images with their names.
        """

        binned_image_1d_dict = {}

        for galaxy, galaxy_image_2d in self.image_galaxy_2d_dict.items():
            binned_image_2d = aa.util.binning.bin_array_2d_via_mean(
                array_2d=galaxy_image_2d, bin_up_factor=binned_grid.bin_up_factor
            )

            binned_image_1d_dict[
                galaxy
            ] = binned_grid.mask.mapping.array_1d_from_array_2d(
                array_2d=binned_image_2d
            )

        return binned_image_1d_dict

    def binned_hyper_galaxy_image_1d_path_dict(self, binned_grid):
        """
        A dictionary associating 1D hyper_galaxies galaxy binned images with their names.
        """

        if binned_grid is not None:

            hyper_minimum_percent = af.conf.instance.general.get(
                "hyper", "hyper_minimum_percent", float
            )

            binned_image_1d_galaxy_dict = self.binned_image_1d_dict_from_binned_grid(
                binned_grid=binned_grid
            )

            binned_hyper_galaxy_image_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                binned_galaxy_image_1d = binned_image_1d_galaxy_dict[path]

                minimum_hyper_value = hyper_minimum_percent * max(
                    binned_galaxy_image_1d
                )
                binned_galaxy_image_1d[
                    binned_galaxy_image_1d < minimum_hyper_value
                    ] = minimum_hyper_value

                binned_hyper_galaxy_image_path_dict[path] = binned_galaxy_image_1d

            return binned_hyper_galaxy_image_path_dict

    def binned_hyper_galaxy_image_2d_path_dict(self, binned_grid):
        """
        A dictionary associating "D hyper_galaxies galaxy images binned images with their names.
        """

        if binned_grid is not None:

            binned_hyper_galaxy_image_1d_path_dict = self.binned_hyper_galaxy_image_1d_path_dict(
                binned_grid=binned_grid
            )

            binned_hyper_galaxy_image_2d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                binned_hyper_galaxy_image_2d_path_dict[
                    path
                ] = binned_grid.mask.mapping.scaled_array_2d_from_array_1d(
                    array_1d=binned_hyper_galaxy_image_1d_path_dict[path]
                )

            return binned_hyper_galaxy_image_2d_path_dict

    @property
    def hyper_model_image_1d(self):

        hyper_model_image_1d = np.zeros(self.mask.pixels_in_mask)

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_image_1d += self.hyper_galaxy_image_1d_path_dict[path]

        return hyper_model_image_1d