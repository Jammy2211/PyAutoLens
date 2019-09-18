import numpy as np

import autofit as af
from autolens.model.galaxy import galaxy as g

from autolens.array.mapping import reshape_returned_array


class UVPlaneFit(af.DataFit):
    def __init__(
        self, visibilities, noise_map, mask, model_visibilities, mapping, inversion
    ):

        super().__init__(
            data=visibilities,
            noise_map=noise_map,
            mask=mask,
            model_data=model_visibilities,
        )

        self.mapping = mapping
        self.inversion = inversion

    def visibilities(self):
        return self._data

    def noise_map(self):
        return self._noise_map

    def mask(self):
        return self.mapping.mask

    def signal_to_noise_map(self):
        return self._signal_to_noise_map

    def model_visibilities(self):
        return self._model_data

    def residual_map(self):
        return self._residual_map

    def normalized_residual_map(self):
        return self._normalized_residual_map

    def chi_squared_map(self):
        return self._chi_squared_map

    # @property
    # def likelihood_with_regularization(self):
    #     if self.inversion is not None:
    #         return likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
    #             chi_squared=self.chi_squared,
    #             regularization_term=self.inversion.regularization_term,
    #             noise_normalization=self.noise_normalization,
    #         )
    #
    # @property
    # def evidence(self):
    #     if self.inversion is not None:
    #         return evidence_from_inversion_terms(
    #             chi_squared=self.chi_squared,
    #             regularization_term=self.inversion.regularization_term,
    #             log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
    #             log_regularization_term=self.inversion.log_det_regularization_matrix_term,
    #             noise_normalization=self.noise_normalization,
    #         )

    @property
    def figure_of_merit(self):
        if self.inversion is None:
            return self.likelihood
        # else:
        #     return self.evidence


class LensUVPlaneFit(UVPlaneFit):
    def __init__(
        self,
        tracer,
        grid,
        visibilities,
        noise_map,
        mask,
        inversion,
        model_visibilities,
        transformer,
        positions=None,
    ):

        self.tracer = tracer
        self.grid = grid
        self.transformer = transformer
        self.positions = positions

        super().__init__(
            visibilities=visibilities,
            noise_map=noise_map,
            mask=mask,
            model_visibilities=model_visibilities,
            mapping=grid.mapping,
            inversion=inversion,
        )

    @classmethod
    def from_lens_data_and_tracer(cls, lens_data, tracer):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens data's hyper_galaxies.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D array.
        """

        profile_visibilities = tracer.profile_visibilities_from_grid_and_transformer(
            grid=lens_data.grid, transformer=lens_data.transformer
        )

        # profile_subtracted_visibilities_1d = visibilities_1d - blurred_profile_visibilities_1d
        #
        # if not tracer.has_pixelization:

        inversion = None
        model_visibilities = profile_visibilities

        # else:
        #
        #     inversion = tracer.inversion_from_grid_visibilities_1d_noise_map_1d_and_convolver(
        #         grid=lens_uv_plane_data.grid,
        #         visibilities_1d=profile_subtracted_visibilities_1d,
        #         noise_map_1d=noise_map_1d,
        #         convolver=lens_uv_plane_data.convolver,
        #         inversion_uses_border=lens_uv_plane_data.inversion_uses_border,
        #         preload_pixelization_grids_of_planes=lens_uv_plane_data.preload_pixelization_grids_of_planes,
        #     )
        #
        #     model_visibilities_1d = blurred_profile_visibilities_1d + inversion.reconstructed_data_1d

        return cls(
            tracer=tracer,
            visibilities=lens_data.visibilities(),
            noise_map=lens_data.noise_map(return_x2=True),
            mask=lens_data.visibilities_mask,
            model_visibilities=model_visibilities,
            grid=lens_data.grid,
            transformer=lens_data.transformer,
            inversion=inversion,
            positions=lens_data.positions,
        )

    def profile_visibilities(self):
        return self.tracer.tracer.profile_visibilities_from_grid_and_transformer(
            grid=self.grid, transformer=self.transformer, return_in_2d=False
        )

    @reshape_returned_array
    def profile_subtracted_visibilities(self):
        return self.visibilities() - self.profile_visibilities()

    # @property
    # def galaxy_visibilities_1d_dict(self) -> {g.Galaxy: np.ndarray}:
    #     """
    #     A dictionary associating galaxies with their corresponding model visibilities
    #     """
    #     galaxy_visibilities_dict = self.tracer.galaxy_visibilities_dict_from_grid_and_convolver(
    #         grid=self.grid, convolver=self.convolver
    #     )
    #
    #     # TODO : Extend to multiple inversioons across Planes
    #
    #     for plane_index in self.tracer.plane_indexes_with_pixelizations:
    #
    #         galaxy_visibilities_dict.update(
    #             {
    #                 self.tracer.planes[plane_index].galaxies[
    #                     0
    #                 ]: self.inversion.reconstructed_data_1d
    #             }
    #         )
    #
    #     return galaxy_visibilities_dict
    #
    # @property
    # def galaxy_visibilities_2d_dict(self) -> {g.Galaxy: np.ndarray}:
    #     """
    #     A dictionary associating galaxies with their corresponding model visibilities
    #     """
    #
    #     galaxy_visibilities_2d_dict = {}
    #
    #     for galalxy, galaxy_visibilities in self.galaxy_visibilities_1d_dict.items():
    #
    #         galaxy_visibilities_2d_dict[galalxy] = self.grid.mapping.scaled_array_2d_from_array_1d(
    #             array_1d=galaxy_visibilities
    #         )
    #
    #     return galaxy_visibilities_2d_dict

    def model_visibilities_of_planes(self):

        model_visibilities_of_planes = self.tracer.profile_visibilities_of_planes_from_grid_and_transformer(
            grid=self.grid, transformer=self.transformer
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            model_visibilities_of_planes[
                plane_index
            ] += self.inversion.reconstructed_data_1d

        return model_visibilities_of_planes

    @property
    def total_inversions(self):
        return len(list(filter(None, self.tracer.regularizations_of_planes)))
