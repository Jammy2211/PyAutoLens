import numpy as np

from autoarray.fit import fit
from autoastro.galaxy import galaxy as g


class ImagingFit(fit.ImagingFit):
    def __init__(
        self,
        tracer,
        grid,
        blurring_grid,
        image,
        noise_map,
        mask,
        inversion,
        model_image,
        convolver,
        positions=None,
    ):

        self.tracer = tracer
        self.grid = grid
        self.blurring_grid = blurring_grid
        self.psf = convolver.psf
        self.convolver = convolver
        self.positions = positions

        super().__init__(
            image=image,
            noise_map=noise_map,
            mask=mask,
            model_image=model_image,
            inversion=inversion,
        )

    @classmethod
    def from_masked_data_and_tracer(
        cls, lens_data, tracer, hyper_image_sky=None, hyper_background_noise=None
    ):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens data's hyper_galaxies.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D array.
        """

        image_1d = hyper_image_from_image_and_hyper_image_sky(
            image=lens_data, hyper_image_sky=hyper_image_sky
        )

        noise_map_1d = hyper_noise_map_from_noise_map_tracer_and_hyper_backkground_noise(
            lens_data=lens_data,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        blurred_profile_image_1d = tracer.blurred_profile_image_from_grid_and_convolver(
            grid=lens_data.grid,
            convolver=lens_data.convolver,
            blurring_grid=lens_data.blurring_grid,
        )

        profile_subtracted_image_1d = image_1d - blurred_profile_image_1d

        if not tracer.has_pixelization:

            inversion = None
            model_image_1d = blurred_profile_image_1d

        else:

            inversion = tracer.inversion_imaging_from_grid_and_data(
                grid=lens_data.grid,
                image=profile_subtracted_image_1d,
                noise_map=noise_map_1d,
                convolver=lens_data.convolver,
                inversion_uses_border=lens_data.inversion_uses_border,
                preload_pixelization_grids_of_planes=lens_data.preload_pixelization_grids_of_planes,
            )

            model_image_1d = blurred_profile_image_1d + inversion.mapped_reconstructed_image

        return cls(
            tracer=tracer,
            image=image_1d,
            noise_map=noise_map_1d,
            mask=lens_data._mask_1d,
            model_image=model_image_1d,
            grid=lens_data.grid,
            blurring_grid=lens_data.blurring_grid,
            convolver=lens_data.convolver,
            inversion=inversion,
            positions=lens_data.positions,
        )

    def blurred_profile_image(self):
        return self.tracer.blurred_profile_image_from_grid_and_psf(
            grid=self.grid, psf=self.psf, blurring_grid=self.blurring_grid
        )

    def profile_subtracted_image(self):
        return self.image - self.blurred_profile_image

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_blurred_profile_image_dict_from_grid_and_convolver(
            grid=self.grid, convolver=self.convolver, blurring_grid=self.blurring_grid
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

    def model_images_of_planes(self):

        model_images_of_planes = self.tracer.blurred_profile_images_of_planes_from_grid_and_psf(
            grid=self.grid, psf=self.psf, blurring_grid=self.blurring_grid
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
        return image.image.in_1d


def hyper_noise_map_from_noise_map_tracer_and_hyper_backkground_noise(
    lens_data, tracer, hyper_background_noise
):

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
            noise_map=lens_data.noise_map
        )
    else:
        noise_map = lens_data.noise_map

    hyper_noise_map = tracer.hyper_noise_map_from_noise_map(
        noise_map=lens_data.noise_map
    )

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map
        if lens_data.hyper_noise_map_max is not None:
            noise_map[
                noise_map > lens_data.hyper_noise_map_max
            ] = lens_data.hyper_noise_map_max

    return noise_map
