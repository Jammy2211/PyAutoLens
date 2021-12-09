import numpy as np
from typing import Dict, Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.model.preloads import Preloads


class FitInterferometer(aa.FitInterferometer):
    def __init__(
        self,
        dataset,
        tracer,
        hyper_background_noise=None,
        use_hyper_scaling=True,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        tracer : Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        """

        self.tracer = tracer

        self.hyper_background_noise = hyper_background_noise
        self.use_hyper_scaling = use_hyper_scaling

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion

        self.preloads = preloads

        self.profiling_dict = profiling_dict

        if use_hyper_scaling:

            if hyper_background_noise is not None:
                noise_map = hyper_background_noise.hyper_noise_map_complex_from(
                    noise_map=dataset.noise_map
                )
            else:
                noise_map = dataset.noise_map

        else:

            noise_map = dataset.noise_map

        self.tracer = tracer

        self.profile_visibilities = self.tracer.visibilities_via_transformer_from(
            grid=dataset.grid, transformer=dataset.transformer
        )

        self.profile_subtracted_visibilities = (
            dataset.visibilities - self.profile_visibilities
        )

        if not tracer.has_pixelization:

            inversion = None
            model_visibilities = self.profile_visibilities

        else:

            inversion = tracer.inversion_interferometer_from(
                grid=dataset.grid_inversion,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=noise_map,
                transformer=dataset.transformer,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
                preloads=preloads,
            )

            model_visibilities = (
                self.profile_visibilities + inversion.mapped_reconstructed_data
            )

        fit = aa.FitDataComplex(
            data=dataset.visibilities,
            noise_map=noise_map,
            model_data=model_visibilities,
            inversion=inversion,
            use_mask_in_fit=False,
            profiling_dict=profiling_dict,
        )

        super().__init__(dataset=dataset, fit=fit, profiling_dict=profiling_dict)

    @property
    def grid(self):
        return self.interferometer.grid

    @property
    def galaxy_model_image_dict(self) -> {ag.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_image_2d_dict_from(grid=self.grid)

        for path, image in galaxy_model_image_dict.items():
            galaxy_model_image_dict[path] = image.binned

        # TODO : Extend to multiple inversioons across Planes

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            galaxy_model_image_dict.update(
                {
                    self.tracer.planes[plane_index].galaxies[
                        0
                    ]: self.inversion.mapped_reconstructed_image
                }
            )

        return galaxy_model_image_dict

    @property
    def galaxy_model_visibilities_dict(self) -> {ag.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_visibilities_dict = self.tracer.galaxy_visibilities_dict_via_transformer_from(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        # TODO : Extend to multiple inversioons across Planes

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            galaxy_model_visibilities_dict.update(
                {
                    self.tracer.planes[plane_index].galaxies[
                        0
                    ]: self.inversion.mapped_reconstructed_data
                }
            )

        return galaxy_model_visibilities_dict

    def model_visibilities_of_planes(self):

        model_visibilities_of_planes = self.tracer.visibilities_list_via_transformer_from(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_visibilities_of_planes[
                plane_index
            ] += self.inversion.mapped_reconstructed_image

        return model_visibilities_of_planes

    @property
    def total_mappers(self):
        return len(list(filter(None, self.tracer.regularization_pg_list)))

    def refit_with_new_preloads(self, preloads, settings_inversion=None):

        if self.profiling_dict is not None:
            profiling_dict = {}
        else:
            profiling_dict = None

        if settings_inversion is None:
            settings_inversion = self.settings_inversion

        return FitInterferometer(
            dataset=self.interferometer,
            tracer=self.tracer,
            hyper_background_noise=self.hyper_background_noise,
            use_hyper_scaling=self.use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )
