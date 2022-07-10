import numpy as np
from typing import Dict, Optional

from autoconf import conf
from autoconf import cached_property

import autoarray as aa
import autogalaxy as ag

from autogalaxy.abstract_fit import AbstractFit

from autolens.analysis.preloads import Preloads
from autolens.lens.ray_tracing import Tracer


class FitImaging(aa.FitImaging, AbstractFit):
    def __init__(
        self,
        dataset: aa.Imaging,
        tracer: Tracer,
        hyper_image_sky: Optional[ag.hyper_data.HyperImageSky] = None,
        hyper_background_noise: Optional[ag.hyper_data.HyperBackgroundNoise] = None,
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

        super().__init__(dataset=dataset, profiling_dict=profiling_dict)
        AbstractFit.__init__(
            self=self, model_obj=tracer, settings_inversion=settings_inversion
        )

        self.tracer = tracer

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise
        self.use_hyper_scaling = use_hyper_scaling

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion

        self.preloads = preloads

    @property
    def data(self):
        """
        Returns the imaging data, which may have a hyper scaling performed which rescales the background sky level
        in order to account for uncertainty in the background sky subtraction.
        """
        if self.use_hyper_scaling:

            return hyper_image_from(
                image=self.dataset.image, hyper_image_sky=self.hyper_image_sky
            )

        return self.dataset.data

    @property
    def noise_map(self):
        """
        Returns the imaging noise-map, which may have a hyper scaling performed which increase the noise in regions of
        the data that are poorly fitted in order to avoid overfitting.
        """
        if self.use_hyper_scaling:

            return hyper_noise_map_from(
                noise_map=self.dataset.noise_map,
                tracer=self.tracer,
                hyper_background_noise=self.hyper_background_noise,
            )

        return self.dataset.noise_map

    @property
    def blurred_image(self):
        """
        Returns the image of all light profiles in the fit's tracer convolved with the imaging dataset's PSF.

        For certain lens models the blurred image does not change (for example when all light profiles in the tracer
        are fixed in the lens model). For faster run-times the blurred image can be preloaded.
        """

        if self.preloads.blurred_image is None:

            return self.tracer.blurred_image_2d_from(
                grid=self.dataset.grid,
                convolver=self.dataset.convolver,
                blurring_grid=self.dataset.blurring_grid,
            )
        return self.preloads.blurred_image

    @property
    def profile_subtracted_image(self):
        """
        Returns the dataset's image with all blurred light profile images in the fit's tracer subtracted.
        """
        return self.image - self.blurred_image

    @cached_property
    def inversion(self):
        """
        If the tracer has linear objects which are used to fit the data (e.g. a pixelization) this function returns
        the linear inversion.

        The image passed to this function is the dataset's image with all light profile images of the tracer subtracted.
        """
        if self.perform_inversion:

            return self.tracer.to_inversion.inversion_imaging_from(
                dataset=self.dataset,
                image=self.profile_subtracted_image,
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

        If the tracer does not have any linear objects and therefore omits an inversion, the model image is the
        sum of all light profile images.

        If a inversion is included it is the sum of this sum and the inversion's reconstruction of the image.
        """

        if self.tracer.has(cls=aa.pix.Pixelization) or self.tracer.has(
            cls=ag.lp_linear.LightProfileLinear
        ):

            return self.blurred_image + self.inversion.mapped_reconstructed_data

        return self.blurred_image

    @property
    def grid(self):
        return self.imaging.grid

    @property
    def galaxy_model_image_dict(self) -> Dict[ag.Galaxy, np.ndarray]:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_blurred_image_2d_dict = self.tracer.galaxy_blurred_image_2d_dict_from(
            grid=self.grid,
            convolver=self.imaging.convolver,
            blurring_grid=self.imaging.blurring_grid,
        )

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_image=True
        )

        return {**galaxy_blurred_image_2d_dict, **galaxy_linear_obj_image_dict}

    @property
    def model_images_of_planes_list(self):

        galaxy_model_image_dict = self.galaxy_model_image_dict

        model_images_of_planes_list = [
            aa.Array2D.manual_mask(
                array=np.zeros(self.dataset.grid.shape_slim), mask=self.dataset.mask
            )
            for i in range(self.tracer.total_planes)
        ]

        for plane_index, plane in enumerate(self.tracer.planes):
            for galaxy in plane.galaxies:
                model_images_of_planes_list[plane_index] += galaxy_model_image_dict[
                    galaxy
                ]

        return model_images_of_planes_list

    @property
    def subtracted_images_of_planes_list(self):

        subtracted_images_of_planes_list = []

        model_images_of_planes_list = self.model_images_of_planes_list

        for galaxy_index in range(len(self.tracer.planes)):

            other_planes_model_images = [
                model_image
                for i, model_image in enumerate(model_images_of_planes_list)
                if i != galaxy_index
            ]

            subtracted_image = self.image - sum(other_planes_model_images)

            subtracted_images_of_planes_list.append(subtracted_image)

        return subtracted_images_of_planes_list

    @property
    def unmasked_blurred_image(self):
        return self.tracer.unmasked_blurred_image_2d_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def unmasked_blurred_image_of_planes_list(self):
        return self.tracer.unmasked_blurred_image_2d_list_from(
            grid=self.grid, psf=self.imaging.psf
        )

    def refit_with_new_preloads(self, preloads, settings_inversion=None):

        profiling_dict = {} if self.profiling_dict is not None else None

        settings_inversion = (
            self.settings_inversion
            if settings_inversion is None
            else settings_inversion
        )

        return FitImaging(
            dataset=self.imaging,
            tracer=self.tracer,
            hyper_image_sky=self.hyper_image_sky,
            hyper_background_noise=self.hyper_background_noise,
            use_hyper_scaling=self.use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )


def hyper_image_from(image, hyper_image_sky):

    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from(image=image)
    else:
        return image


def hyper_noise_map_from(noise_map, tracer, hyper_background_noise):

    hyper_noise_map = tracer.hyper_noise_map_from(noise_map=noise_map)

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from(noise_map=noise_map)

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map
        noise_map_limit = conf.instance["general"]["hyper"]["hyper_noise_limit"]
        noise_map[noise_map > noise_map_limit] = noise_map_limit

    return noise_map
