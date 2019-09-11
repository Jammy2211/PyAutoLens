import numpy as np

import autofit as af
from autolens.model.galaxy import galaxy as g

from autolens.array.grids import reshape_array


class LensImageFit(af.DataFit1D):

    def __init__(self, image_1d, noise_map_1d, mask_1d, model_image_1d, inversion):

        super().__init__(
            data_1d=image_1d,
            noise_map_1d=noise_map_1d,
            mask_1d=mask_1d,
            model_data_1d=model_image_1d,
        )

        self.inversion = inversion

    @property
    def image_1d(self):
        return self.data_1d

    @property
    def model_image_1d(self):
        return self.model_data_1d

    @reshape_array
    def image(self, return_in_2d=True):
        return self.image_1d

    @reshape_array
    def noise_map(self, return_in_2d=True):
        return self.noise_map_1d

    @reshape_array
    def signal_to_noise_map(self, return_in_2d=True):
        return self.signal_to_noise_map_1d

    @reshape_array
    def model_image(self, return_in_2d=True):
        return self.model_image_1d

    @reshape_array
    def residual_map(self, return_in_2d=True):
        return self.residual_map_1d

    @reshape_array
    def normalized_residual_map(self, return_in_2d=True):
        return self.normalized_residual_map_1d

    @reshape_array
    def chi_squared_map(self, return_in_2d=True):
        return self.chi_squared_map_1d

    @property
    def likelihood_with_regularization(self):
        if self.inversion is not None:
            return likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def evidence(self):
        if self.inversion is not None:
            return evidence_from_inversion_terms(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def figure_of_merit(self):
        if self.inversion is None:
            return self.likelihood
        else:
            return self.evidence

class LensTracerImageFit(LensImageFit):

    def __init__(self, tracer, grid, image_1d, noise_map_1d, model_image_1d, convolver, inversion, positions=None):

        self.tracer = tracer
        self.grid = grid
        self.psf = convolver.psf
        self.convolver = convolver
        self.positions = positions

        super().__init__(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            mask_1d=np.full(fill_value=False, shape=(len(image_1d,))),
            model_image_1d=model_image_1d,
            inversion=inversion,
        )

    @classmethod
    def from_lens_data_and_tracer(cls, lens_data, tracer, hyper_image_sky=None, hyper_background_noise=None):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens data's hyper_galaxies.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D array.
        """

        image_1d = image_1d_from_lens_data_and_hyper_image_sky(
            lens_data=lens_data, hyper_image_sky=hyper_image_sky
        )

        noise_map_1d = noise_map_1d_from_lens_data_tracer_and_hyper_backkground_noise(
            lens_data=lens_data,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        blurred_profile_image_1d = tracer.blurred_profile_image_from_grid_and_convolver(
            grid=lens_data.grid,
            convolver=lens_data.convolver,
            preload_blurring_grid=lens_data.preload_blurring_grid,
            return_in_2d=False,
        )

        profile_subtracted_image_1d = image_1d - blurred_profile_image_1d

        if not tracer.has_pixelization:

            inversion=None
            model_image_1d = blurred_profile_image_1d

        else:

            inversion = tracer.inversion_from_grid_image_1d_noise_map_1d_and_convolver(
                grid=lens_data.grid,
                image_1d=profile_subtracted_image_1d,
                noise_map_1d=noise_map_1d,
                convolver=lens_data.convolver,
                inversion_uses_border=lens_data.inversion_uses_border,
                preload_pixelization_grids_of_planes=lens_data.preload_pixelization_grids_of_planes,
            )

            model_image_1d = blurred_profile_image_1d + inversion.reconstructed_data_1d

        return cls(tracer=tracer, image_1d=image_1d, noise_map_1d=noise_map_1d, model_image_1d=model_image_1d,
                   grid=lens_data.grid, convolver=lens_data.convolver,
                   inversion=inversion, positions=lens_data.positions)

    def mask(self, return_in_2d=True):
        if return_in_2d:
            return self.grid.mask
        else:
            return self.mask_1d

    @reshape_array
    def blurred_profile_image(self, return_in_2d=True):
        return self.tracer.blurred_profile_image_from_grid_and_psf(
            grid=self.grid,
            psf=self.psf,
            return_in_2d=False,
        )

    @reshape_array
    def profile_subtracted_image(self, return_in_2d=True):
        return self.image_1d - self.blurred_profile_image(return_in_2d=False)

    @property
    def galaxy_image_1d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_image_dict = self.tracer.galaxy_image_dict_from_grid_and_convolver(
            grid=self.grid,
            convolver=self.convolver,
        )

        # TODO : Extend to multiple inversioons across Planes

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            galaxy_image_dict.update(
                {self.tracer.planes[plane_index].galaxies[0]: self.inversion.reconstructed_data_1d}
            )

        return galaxy_image_dict

    @property
    def galaxy_image_2d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_image_2d_dict = {}

        for galalxy, galaxy_image in self.galaxy_image_1d_dict.items():

            galaxy_image_2d_dict[galalxy] = self.grid.scaled_array_2d_from_array_1d(
                array_1d=galaxy_image
            )

        return galaxy_image_2d_dict

    def model_images_of_planes(self, return_in_2d=True):

        model_images_of_planes = self.tracer.blurred_profile_images_of_planes_from_grid_and_psf(
            grid=self.grid,
            psf=self.psf,
            return_in_2d=return_in_2d,
        )

        for plane_index in self.tracer.plane_indexes_with_pixelizations:

            if return_in_2d:
                model_images_of_planes[plane_index] += self.inversion.reconstructed_data_2d
            else:
                model_images_of_planes[plane_index] += self.inversion.reconstructed_data_1d

        return model_images_of_planes

    @property
    def total_inversions(self):
        return len(list(filter(None, self.tracer.regularizations_of_planes)))

    @property
    def unmasked_blurred_profile_image(self):
        return self.tracer.unmasked_blurred_profile_image_from_grid_and_psf(
            grid=self.grid, psf=self.psf
        )

    @property
    def unmasked_blurred_profile_image_of_planes(self):
        return self.tracer.unmasked_blurred_profile_image_of_planes_from_grid_and_psf(
            grid=self.grid, psf=self.psf
        )

    @property
    def unmasked_blurred_profile_image_of_planes_and_galaxies(self):
        return self.tracer.unmasked_blurred_profile_image_of_planes_and_galaxies_from_grid_and_psf(
            grid=self.grid, psf=self.psf
        )


class LensPositionFit(object):
    def __init__(self, positions, noise_map):
        """A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher likelihood.

        Parameters
        -----------
        positions : [[]]
            The (y,x) arc-second coordinates of positions which the maximum distance and likelihood is computed using.
        noise_map : ndarray | float
            The noise-value assumed when computing the likelihood.
        """
        self.positions = positions
        self.noise_map = noise_map

    @property
    def chi_squared_map(self):
        return np.square(np.divide(self.maximum_separations, self.noise_map))

    @property
    def figure_of_merit(self):
        return -0.5 * sum(self.chi_squared_map)

    def maximum_separation_within_threshold(self, threshold):
        return max(self.maximum_separations) <= threshold

    @property
    def maximum_separations(self):
        return list(
            map(
                lambda positions: self.max_separation_of_grid(grid=positions),
                self.positions,
            )
        )

    @staticmethod
    def max_separation_of_grid(grid):
        rdist_max = np.zeros((grid.shape[0]))
        for i in range(grid.shape[0]):
            xdists = np.square(np.subtract(grid[i, 0], grid[:, 0]))
            ydists = np.square(np.subtract(grid[i, 1], grid[:, 1]))
            rdist_max[i] = np.max(np.add(xdists, ydists))
        return np.max(np.sqrt(rdist_max))


def image_1d_from_lens_data_and_hyper_image_sky(lens_data, hyper_image_sky):

    if hyper_image_sky is not None:
        return hyper_image_sky.image_scaled_sky_from_image(image=lens_data.image_1d)
    else:
        return lens_data.image_1d


def noise_map_1d_from_lens_data_tracer_and_hyper_backkground_noise(
    lens_data, tracer, hyper_background_noise
):

    if hyper_background_noise is not None:
        noise_map_1d = hyper_background_noise.noise_map_scaled_noise_from_noise_map(
            noise_map=lens_data.noise_map_1d
        )
    else:
        noise_map_1d = lens_data.noise_map_1d

    hyper_noise_map_1d = tracer.hyper_noise_map_1d_from_noise_map_1d(
        noise_map_1d=lens_data.noise_map_1d
    )

    if hyper_noise_map_1d is not None:
        noise_map_1d = noise_map_1d + hyper_noise_map_1d
        if lens_data.hyper_noise_map_max is not None:
            noise_map_1d[
                noise_map_1d > lens_data.hyper_noise_map_max
            ] = lens_data.hyper_noise_map_max

    return noise_map_1d


def likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
    chi_squared, regularization_term, noise_normalization
):
    """Compute the likelihood of an inversion's fit to the datas, including a regularization term which \
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + noise_normalization)


def evidence_from_inversion_terms(
    chi_squared,
    regularization_term,
    log_curvature_regularization_term,
    log_regularization_term,
    noise_normalization,
):
    """Compute the evidence of an inversion's fit to the datas, where the evidence includes a number of \
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_curvature_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (
        chi_squared
        + regularization_term
        + log_curvature_regularization_term
        - log_regularization_term
        + noise_normalization
    )
