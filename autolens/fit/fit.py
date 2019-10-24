import numpy as np

from autoarray.fit import fit
from autoastro.galaxy import galaxy as g


class ImagingFit(fit.ImagingFit):
    def __init__(
        self,
        masked_imaging,
            tracer,
            hyper_image_sky=None,
            hyper_background_noise=None,
    ):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens simulate's hyper_galaxies.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D array.
        """

        self.tracer = tracer
        self.masked_imaging = masked_imaging

        image = hyper_image_from_image_and_hyper_image_sky(
            image=masked_imaging.image, hyper_image_sky=hyper_image_sky
        )

        noise_map = hyper_noise_map_from_noise_map_tracer_and_hyper_backkground_noise(
            noise_map=masked_imaging.noise_map,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        self.blurred_profile_image = tracer.blurred_profile_image_from_grid_and_convolver(
            grid=masked_imaging.grid,
            convolver=masked_imaging.convolver,
            blurring_grid=masked_imaging.blurring_grid,
        )

        self.profile_subtracted_image = image - self.blurred_profile_image

        if not tracer.has_pixelization:

            inversion = None
            model_image = self.blurred_profile_image

        else:

            inversion = tracer.inversion_imaging_from_grid_and_data(
                grid=masked_imaging.grid,
                image=self.profile_subtracted_image,
                noise_map=noise_map,
                convolver=masked_imaging.convolver,
                inversion_uses_border=masked_imaging.inversion_uses_border,
                preload_pixelization_grids_of_planes=masked_imaging.preload_pixelization_grids_of_planes,
            )

            model_image = self.blurred_profile_image + inversion.mapped_reconstructed_image

        super().__init__(
            mask=masked_imaging.mask,
            image=image, noise_map=noise_map,
            model_image=model_image,
            inversion=inversion,
        )

    @property
    def grid(self):
        return self.masked_imaging.grid

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_blurred_profile_image_dict_from_grid_and_convolver(
            grid=self.grid, convolver=self.masked_imaging.convolver, blurring_grid=self.masked_imaging.blurring_grid
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

        model_images_of_planes = self.tracer.blurred_profile_images_of_planes_from_grid_and_psf(
            grid=self.grid, psf=self.masked_imaging.psf, blurring_grid=self.masked_imaging.blurring_grid
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_images_of_planes[
                plane_index
            ] += self.inversion.mapped_reconstructed_image

        return model_images_of_planes

    @property
    def total_inversions(self):
        return len(list(filter(None, self.tracer.regularizations_of_planes)))


def hyper_image_from_image_and_hyper_image_sky(image, hyper_image_sky):

    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from_image(image=image)
    else:
        return image


def hyper_noise_map_from_noise_map_tracer_and_hyper_backkground_noise(
    noise_map, tracer, hyper_background_noise
):

    hyper_noise_map = tracer.hyper_noise_map_from_noise_map(
        noise_map=noise_map
    )

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
            noise_map=noise_map
        )

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map

    return noise_map