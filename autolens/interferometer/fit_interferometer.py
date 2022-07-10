import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

import autoarray as aa
import autogalaxy as ag

from autogalaxy.abstract_fit import AbstractFit

from autolens.analysis.preloads import Preloads
from autolens.lens.ray_tracing import Tracer


class FitInterferometer(aa.FitInterferometer, AbstractFit):
    def __init__(
        self,
        dataset: aa.Interferometer,
        tracer: Tracer,
        hyper_background_noise: ag.hyper_data.HyperBackgroundNoise = None,
        use_hyper_scaling: bool = True,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
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

        super().__init__(dataset=dataset, profiling_dict=profiling_dict)
        AbstractFit.__init__(
            self=self, model_obj=tracer, settings_inversion=settings_inversion
        )

    @property
    def noise_map(self):
        """
        Returns the interferometer's noise-map, which may have a hyper scaling performed which increase the noise in
        regions of the data that are poorly fitted in order to avoid overfitting.
        """
        if self.use_hyper_scaling and self.hyper_background_noise is not None:

            return self.hyper_background_noise.hyper_noise_map_complex_from(
                noise_map=self.dataset.noise_map
            )

        return self.dataset.noise_map

    @property
    def profile_visibilities(self):
        """
        Returns the visibilities of every light profile in the plane, which are computed by performing a Fourier
        transform to the sum of light profile images.
        """
        return self.tracer.visibilities_from(
            grid=self.dataset.grid, transformer=self.dataset.transformer
        )

    @property
    def profile_subtracted_visibilities(self):
        """
        Returns the interferomter dataset's visibilities with all transformed light profile images in the fit's
        plane subtracted.
        """
        return self.visibilities - self.profile_visibilities

    @cached_property
    def inversion(self):
        """
        If the plane has linear objects which are used to fit the data (e.g. a pixelization) this function returns
        the linear inversion.

        The image passed to this function is the dataset's image with all light profile images of the plane subtracted.
        """
        if self.perform_inversion:

            return self.tracer.to_inversion.inversion_interferometer_from(
                dataset=self.dataset,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=self.noise_map,
                w_tilde=self.w_tilde,
                settings_pixelization=self.settings_pixelization,
                settings_inversion=self.settings_inversion,
                preloads=self.preloads,
            )

    @property
    def model_data(self):
        """
        Returns the model-image that is used to fit the data.

        If the plane does not have any linear objects and therefore omits an inversion, the model image is the
        sum of all light profile images.

        If a inversion is included it is the sum of this sum and the inversion's reconstruction of the image.
        """

        if self.tracer.has(cls=aa.pix.Pixelization) or self.tracer.has(
            cls=ag.lp_linear.LightProfileLinear
        ):

            return self.profile_visibilities + self.inversion.mapped_reconstructed_data

        return self.profile_visibilities

    @property
    def grid(self):
        return self.interferometer.grid

    @property
    def galaxy_model_image_dict(self) -> Dict[ag.Galaxy, np.ndarray]:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_image_2d_dict_from(grid=self.grid)

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_image=True
        )

        return {**galaxy_model_image_dict, **galaxy_linear_obj_image_dict}

    @property
    def galaxy_model_visibilities_dict(self) -> Dict[ag.Galaxy, np.ndarray]:

        galaxy_model_visibilities_dict = self.tracer.galaxy_visibilities_dict_from(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        galaxy_linear_obj_visibilities_dict = self.galaxy_linear_obj_data_dict_from(
            use_image=False
        )

        return {**galaxy_model_visibilities_dict, **galaxy_linear_obj_visibilities_dict}

    @property
    def model_visibilities_of_planes_list(self):

        galaxy_model_visibilities_dict = self.galaxy_model_visibilities_dict

        model_visibilities_of_planes_list = [
            aa.Visibilities.zeros(shape_slim=(self.dataset.visibilities.shape_slim,))
            for i in range(self.tracer.total_planes)
        ]

        for plane_index, plane in enumerate(self.tracer.planes):
            for galaxy in plane.galaxies:
                model_visibilities_of_planes_list[
                    plane_index
                ] += galaxy_model_visibilities_dict[galaxy]

        return model_visibilities_of_planes_list

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
