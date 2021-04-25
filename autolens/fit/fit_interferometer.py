import numpy as np

from autoarray.fit import fit as aa_fit
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.galaxy import galaxy as g
from autoarray import preloads as pload


class FitInterferometer(aa_fit.FitInterferometer):
    def __init__(
        self,
        interferometer,
        tracer,
        hyper_background_noise=None,
        use_hyper_scaling=True,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        preloads=pload.Preloads(),
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

        if use_hyper_scaling:

            if hyper_background_noise is not None:
                noise_map = hyper_background_noise.hyper_noise_map_from_complex_noise_map(
                    noise_map=interferometer.noise_map
                )
            else:
                noise_map = interferometer.noise_map

            if hyper_background_noise is not None:

                interferometer = interferometer.modify_noise_map(noise_map=noise_map)

        else:

            noise_map = interferometer.noise_map

        self.tracer = tracer

        self.profile_visibilities = tracer.profile_visibilities_from_grid_and_transformer(
            grid=interferometer.grid, transformer=interferometer.transformer
        )

        self.profile_subtracted_visibilities = (
            interferometer.visibilities - self.profile_visibilities
        )

        if not tracer.has_pixelization:

            inversion = None
            model_visibilities = self.profile_visibilities

        else:

            inversion = tracer.inversion_interferometer_from_grid_and_data(
                grid=interferometer.grid_inversion,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=noise_map,
                transformer=interferometer.transformer,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
                preloads=preloads,
            )

            model_visibilities = (
                self.profile_visibilities + inversion.mapped_reconstructed_visibilities
            )

        super().__init__(
            interferometer=interferometer,
            model_visibilities=model_visibilities,
            inversion=inversion,
            use_mask_in_fit=False,
        )

    @property
    def grid(self):
        return self.interferometer.grid

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.tracer.galaxy_image_dict_from_grid(
            grid=self.grid
        )

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
    def galaxy_model_visibilities_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_visibilities_dict = self.tracer.galaxy_profile_visibilities_dict_from_grid_and_transformer(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
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

        model_visibilities_of_planes = self.tracer.profile_visibilities_of_planes_from_grid_and_transformer(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_visibilities_of_planes[
                plane_index
            ] += self.inversion.mapped_reconstructed_image

        return model_visibilities_of_planes

    @property
    def total_inversions(self):
        return len(list(filter(None, self.tracer.regularizations_of_planes)))
