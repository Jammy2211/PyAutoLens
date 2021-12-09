import numpy as np
from typing import Dict, Optional

from autoconf import conf
import autoarray as aa
import autogalaxy as ag

from autolens.lens.model.preloads import Preloads


class FitImaging(aa.FitImaging):
    def __init__(
        self,
        dataset,
        tracer,
        hyper_image_sky=None,
        hyper_background_noise=None,
        use_hyper_scaling=True,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=Preloads(),
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

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise
        self.use_hyper_scaling = use_hyper_scaling

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion

        self.preloads = preloads

        self.profiling_dict = profiling_dict

        if use_hyper_scaling:

            image = hyper_image_from(
                image=dataset.image, hyper_image_sky=hyper_image_sky
            )

            noise_map = hyper_noise_map_from(
                noise_map=dataset.noise_map,
                tracer=tracer,
                hyper_background_noise=hyper_background_noise,
            )

        else:

            image = dataset.image
            noise_map = dataset.noise_map

        if preloads.blurred_image is None:

            self.blurred_image = self.tracer.blurred_image_2d_via_convolver_from(
                grid=dataset.grid,
                convolver=dataset.convolver,
                blurring_grid=dataset.blurring_grid,
            )

        else:

            self.blurred_image = preloads.blurred_image

        self.profile_subtracted_image = image - self.blurred_image

        if not tracer.has_pixelization:

            inversion = None
            model_image = self.blurred_image

        else:

            inversion = tracer.inversion_imaging_from(
                grid=dataset.grid_inversion,
                image=self.profile_subtracted_image,
                noise_map=noise_map,
                convolver=dataset.convolver,
                w_tilde=dataset.w_tilde,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
                preloads=preloads,
            )

            model_image = self.blurred_image + inversion.mapped_reconstructed_image

        fit = aa.FitData(
            data=image,
            noise_map=noise_map,
            model_data=model_image,
            mask=dataset.mask,
            inversion=inversion,
            use_mask_in_fit=False,
            profiling_dict=profiling_dict,
        )

        super().__init__(dataset=dataset, fit=fit, profiling_dict=profiling_dict)

    @property
    def grid(self):
        return self.imaging.grid

    @property
    def galaxy_model_image_dict(self) -> {ag.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_blurred_image_2d_dict_via_convolver_from(
            grid=self.grid,
            convolver=self.imaging.convolver,
            blurring_grid=self.imaging.blurring_grid,
        )

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
    def model_images_of_planes(self):

        model_images_of_planes = self.tracer.blurred_image_2d_list_via_psf_from(
            grid=self.grid,
            psf=self.imaging.psf,
            blurring_grid=self.imaging.blurring_grid,
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_images_of_planes[
                plane_index
            ] += self.inversion.mapped_reconstructed_image

        return model_images_of_planes

    @property
    def subtracted_images_of_planes(self):

        subtracted_images_of_planes = []

        model_images_of_planes = self.model_images_of_planes

        for galaxy_index in range(len(self.tracer.planes)):

            other_planes_model_images = [
                model_image
                for i, model_image in enumerate(model_images_of_planes)
                if i != galaxy_index
            ]

            subtracted_image = self.image - sum(other_planes_model_images)

            subtracted_images_of_planes.append(subtracted_image)

        return subtracted_images_of_planes

    @property
    def unmasked_blurred_image(self):
        return self.tracer.unmasked_blurred_image_2d_via_psf_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def unmasked_blurred_image_of_planes(self):
        return self.tracer.unmasked_blurred_image_2d_list_via_psf_from(
            grid=self.grid, psf=self.imaging.psf
        )

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
