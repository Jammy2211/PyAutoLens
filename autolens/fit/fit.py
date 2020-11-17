import numpy as np

from autoarray.fit import fit as aa_fit
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.galaxy import galaxy as g


class FitImaging(aa_fit.FitImaging):
    def __init__(
        self,
        masked_imaging,
        tracer,
        hyper_image_sky=None,
        hyper_background_noise=None,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
    ):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D arrays.
        """

        self.tracer = tracer

        image = hyper_image_from_image_and_hyper_image_sky(
            image=masked_imaging.image, hyper_image_sky=hyper_image_sky
        )

        noise_map = hyper_noise_map_from_noise_map_tracer_and_hyper_background_noise(
            noise_map=masked_imaging.noise_map,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        if (
            tracer.has_hyper_galaxy
            or hyper_image_sky is not None
            or hyper_background_noise is not None
        ):

            masked_imaging = masked_imaging.modify_image_and_noise_map(
                image=image, noise_map=noise_map
            )

        self.blurred_image = tracer.blurred_image_from_grid_and_convolver(
            grid=masked_imaging.grid,
            convolver=masked_imaging.convolver,
            blurring_grid=masked_imaging.blurring_grid,
        )

        self.profile_subtracted_image = image - self.blurred_image

        if not tracer.has_pixelization:

            inversion = None
            model_image = self.blurred_image

        else:

            inversion = tracer.inversion_imaging_from_grid_and_data(
                grid=masked_imaging.grid_inversion,
                image=self.profile_subtracted_image,
                noise_map=noise_map,
                convolver=masked_imaging.convolver,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
            )

            model_image = self.blurred_image + inversion.mapped_reconstructed_image

        super().__init__(
            masked_imaging=masked_imaging,
            model_image=model_image,
            inversion=inversion,
            use_mask_in_fit=False,
        )

    @property
    def grid(self):
        return self.masked_imaging.grid

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = (
            self.tracer.galaxy_blurred_image_dict_from_grid_and_convolver(
                grid=self.grid,
                convolver=self.masked_imaging.convolver,
                blurring_grid=self.masked_imaging.blurring_grid,
            )
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

        model_images_of_planes = self.tracer.blurred_images_of_planes_from_grid_and_psf(
            grid=self.grid,
            psf=self.masked_imaging.psf,
            blurring_grid=self.masked_imaging.blurring_grid,
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_images_of_planes[
                plane_index
            ] += self.inversion.mapped_reconstructed_image

        return model_images_of_planes

    @property
    def unmasked_blurred_image(self):
        return self.tracer.unmasked_blurred_image_from_grid_and_psf(
            grid=self.grid, psf=self.masked_imaging.psf
        )

    @property
    def unmasked_blurred_image_of_planes(self):
        return self.tracer.unmasked_blurred_image_of_planes_from_grid_and_psf(
            grid=self.grid, psf=self.masked_imaging.psf
        )

    @property
    def unmasked_blurred_image_of_planes_and_galaxies(self):
        return (
            self.tracer.unmasked_blurred_image_of_planes_and_galaxies_from_grid_and_psf(
                grid=self.grid, psf=self.masked_imaging.psf
            )
        )

    @property
    def total_inversions(self):
        return len(list(filter(None, self.tracer.regularizations_of_planes)))


class FitInterferometer(aa_fit.FitInterferometer):
    def __init__(
        self,
        masked_interferometer,
        tracer,
        hyper_background_noise=None,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
    ):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D arrays.
        """

        if hyper_background_noise is not None:
            noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
                noise_map=masked_interferometer.noise_map
            )
        else:
            noise_map = masked_interferometer.noise_map

        if hyper_background_noise is not None:

            masked_interferometer = masked_interferometer.modify_noise_map(
                noise_map=noise_map
            )

        self.tracer = tracer

        self.profile_visibilities = (
            tracer.profile_visibilities_from_grid_and_transformer(
                grid=masked_interferometer.grid,
                transformer=masked_interferometer.transformer,
            )
        )

        self.profile_subtracted_visibilities = (
            masked_interferometer.visibilities - self.profile_visibilities
        )

        if not tracer.has_pixelization:

            inversion = None
            model_visibilities = self.profile_visibilities

        else:

            inversion = tracer.inversion_interferometer_from_grid_and_data(
                grid=masked_interferometer.grid_inversion,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=noise_map,
                transformer=masked_interferometer.transformer,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
            )

            model_visibilities = (
                self.profile_visibilities + inversion.mapped_reconstructed_visibilities
            )

        super().__init__(
            masked_interferometer=masked_interferometer,
            model_visibilities=model_visibilities,
            inversion=inversion,
            use_mask_in_fit=False,
        )

    @property
    def grid(self):
        return self.masked_interferometer.grid

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_image_dict_from_grid(
            grid=self.grid
        )

        for path, image in galaxy_model_image_dict.items():
            if hasattr(image, "in_1d_binned"):
                galaxy_model_image_dict[path] = image.in_1d_binned

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
    def galaxy_model_visibilities_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_visibilities_dict = (
            self.tracer.galaxy_profile_visibilities_dict_from_grid_and_transformer(
                grid=self.masked_interferometer.grid,
                transformer=self.masked_interferometer.transformer,
            )
        )

        # TODO : Extend to multiple inversioons across Planes

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            galaxy_model_visibilities_dict.update(
                {
                    self.tracer.planes[plane_index].galaxies[
                        0
                    ]: self.inversion.mapped_reconstructed_visibilities
                }
            )

        return galaxy_model_visibilities_dict

    def model_visibilities_of_planes(self):

        model_visibilities_of_planes = (
            self.tracer.profile_visibilities_of_planes_from_grid_and_transformer(
                grid=self.masked_interferometer.grid,
                transformer=self.masked_interferometer.transformer,
            )
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_visibilities_of_planes[
                plane_index
            ] += self.inversion.mapped_reconstructed_image

        return model_visibilities_of_planes

    @property
    def total_inversions(self):
        return len(list(filter(None, self.tracer.regularizations_of_planes)))


def hyper_image_from_image_and_hyper_image_sky(image, hyper_image_sky):

    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from_image(image=image)
    else:
        return image


def hyper_noise_map_from_noise_map_tracer_and_hyper_background_noise(
    noise_map, tracer, hyper_background_noise
):

    hyper_noise_map = tracer.hyper_noise_map_from_noise_map(noise_map=noise_map)

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
            noise_map=noise_map
        )

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map

    return noise_map
