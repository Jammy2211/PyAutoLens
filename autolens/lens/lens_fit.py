import numpy as np

import autofit as af
from autolens import exc
from autolens.lens.util import lens_fit_util as util
from autolens.model.galaxy import galaxy as g


class LensDataFit(af.DataFit1D):

    def __init__(self, image_1d, noise_map_1d, mask_1d, model_image_1d, scaled_array_2d_from_array_1d):

        super().__init__(
            data_1d=image_1d,
            noise_map_1d=noise_map_1d,
            mask_1d=mask_1d,
            model_data_1d=model_image_1d)

        self.scaled_array_2d_from_array_1d = scaled_array_2d_from_array_1d

    @property
    def image_1d(self):
        return self.data_1d

    @property
    def image_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.image_1d)

    @property
    def noise_map_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.noise_map_1d)

    @property
    def signal_to_noise_map_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.signal_to_noise_map_1d)

    @property
    def model_image_1d(self):
        return self.model_data_1d

    @property
    def model_image_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.model_image_1d)

    @property
    def residual_map_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.residual_map_1d)

    @property
    def normalized_residual_map_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.normalized_residual_map_1d)

    @property
    def chi_squared_map_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.chi_squared_map_1d)

    @property
    def figure_of_merit(self):
        return self.likelihood

    @classmethod
    def for_data_and_tracer(cls, lens_data, tracer, padded_tracer=None,
                            hyper_image_sky=None, hyper_noise_background=None):
        """Fit lens data with a model tracer, automatically determining the type of fit based on the \
        properties of the galaxies in the tracer.

        Parameters
        -----------
        lens_data : lens_data.LensData or lens_data.LensDataHyper
            The lens-images that is fitted.
        tracer : ray_tracing.TracerNonStack
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens data's \
            padded grid_stack such that unmasked model-images can be computed.
        """

        if tracer.has_light_profile and not tracer.has_pixelization:
            return LensProfileFit(
                lens_data=lens_data, tracer=tracer, padded_tracer=padded_tracer,
                hyper_image_sky=hyper_image_sky, hyper_noise_background=hyper_noise_background)
        elif not tracer.has_light_profile and tracer.has_pixelization:
            return LensInversionFit(
                lens_data=lens_data, tracer=tracer, hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background)
        elif tracer.has_light_profile and tracer.has_pixelization:
            return LensProfileInversionFit(
                lens_data=lens_data, tracer=tracer, padded_tracer=None,
                hyper_image_sky=hyper_image_sky, hyper_noise_background=hyper_noise_background)
        else:
            raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                       'properties of the tracer')


class LensTracerFit(LensDataFit):

    def __init__(self, image_1d, noise_map_1d, mask_1d, model_image_1d, tracer, psf, scaled_array_2d_from_array_1d,
                 mask_2d, padded_tracer=None):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens data's hyper.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.TracerNonStack or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens data's \
            padded grid_stack such that unmasked model-images can be computed.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper to its unmasked 2D array.
        """

        self.mask_2d = mask_2d

        super().__init__(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            mask_1d=mask_1d,
            model_image_1d=model_image_1d,
            scaled_array_2d_from_array_1d=scaled_array_2d_from_array_1d)

        self.tracer = tracer
        self.padded_tracer = padded_tracer
        self.psf = psf

    @property
    def galaxy_image_1d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        raise NotImplementedError()

    @property
    def galaxy_image_2d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_image_2d_dict = {}

        for galalxy, galaxy_image in self.galaxy_image_1d_dict.items():

            galaxy_image_2d_dict[galalxy] = self.scaled_array_2d_from_array_1d(array_1d=galaxy_image)

        return galaxy_image_2d_dict

    @property
    def total_inversions(self):
        return len(self.tracer.mappers_of_planes)

    @property
    def unmasked_blurred_image_plane_image(self):
        return self.padded_tracer.unmasked_blurred_profile_image_plane_image_from_psf(psf=self.psf)

    @property
    def unmasked_blurred_image_plane_image_of_planes(self):
        return self.padded_tracer.unmasked_blurred_profile_image_plane_image_of_planes_from_psf(psf=self.psf)

    @property
    def unmasked_blurred_image_plane_image_of_planes_and_galaxies(self):
        return self.padded_tracer.unmasked_blurred_profile_image_plane_image_of_plane_and_galaxies_from_psf(
            psf=self.psf)


class LensProfileFit(LensTracerFit):

    def __init__(self, lens_data, tracer, padded_tracer=None, hyper_image_sky=None, hyper_noise_background=None):
        """ An  lens profile fitter, which generates the image-plane image of all galaxies (with light \
        profiles) in the tracer and blurs it with the lens data's PSF.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lens_data : lens_data.LensData
            The lens-image that is fitted.
        tracer : ray_tracing.AbstractTracerData
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens data's \
            padded grid_stack such that unmasked model-images can be computed.
        """

        if hyper_image_sky is not None:
            image_1d = hyper_image_sky.image_scaled_sky_from_image(image=lens_data.image_1d)
        else:
            image_1d = lens_data.image_1d

        if hyper_noise_background is not None:
            noise_map_1d = hyper_noise_background.noise_map_scaled_noise_from_noise_map(noise_map=lens_data.noise_map_1d)
        else:
            noise_map_1d = lens_data.noise_map_1d

        hyper_noise_map_1d = tracer.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=lens_data.noise_map_1d)

        if hyper_noise_map_1d is not None:
            noise_map_1d = noise_map_1d + hyper_noise_map_1d

        blurred_profile_image_1d = \
            tracer.blurred_profile_image_plane_image_1d_from_convolver_image(
                convolver_image=lens_data.convolver_image)

        super(LensProfileFit, self).__init__(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            mask_1d=lens_data.mask_1d,
            model_image_1d=blurred_profile_image_1d,
            mask_2d=lens_data.mask_2d,
            tracer=tracer,
            padded_tracer=padded_tracer,
            psf=lens_data.psf,
            scaled_array_2d_from_array_1d=lens_data.scaled_array_2d_from_array_1d)

        self.convolver_image = lens_data.convolver_image

    @property
    def galaxy_image_1d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        return self.tracer.galaxy_image_dict_from_convolver_image(convolver_image=self.convolver_image)

    @property
    def model_image_2d_of_planes(self):
        return self.tracer.blurred_profile_image_plane_image_2d_of_planes_from_convolver_image(
            convolver_image=self.convolver_image)

    @property
    def figure_of_merit(self):
        return self.likelihood


class InversionFit(LensTracerFit):

    def __init__(self, lens_data, image_1d, noise_map_1d, model_image_1d, tracer, inversion, padded_tracer=None):

        super().__init__(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            mask_1d=lens_data.mask_1d,
            model_image_1d=model_image_1d,
            mask_2d=lens_data.mask_2d,
            psf=lens_data.psf,
            tracer=tracer,
            padded_tracer=padded_tracer,
            scaled_array_2d_from_array_1d=lens_data.scaled_array_2d_from_array_1d)

        self.inversion = inversion

        self.likelihood_with_regularization = \
            util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                chi_squared=self.chi_squared, regularization_term=inversion.regularization_term,
                noise_normalization=self.noise_normalization)

        self.evidence = util.evidence_from_inversion_terms(
            chi_squared=self.chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=self.noise_normalization)


class LensInversionFit(InversionFit):

    def __init__(self, lens_data, tracer, hyper_image_sky=None, hyper_noise_background=None):
        """ An  lens inversion fitter, which fits the lens data an inversion using the mapper(s) and \
        regularization(s) in the galaxies of the tracer.

        This inversion use's the lens-image, its PSF and an input noise-map.

        Parameters
        -----------
        lens_data : lens_data.LensData
            The lens-image that is fitted.
        tracer : ray_tracing.AbstractTracerData
            The tracer, which describes the ray-tracing and strong lens configuration.
        """

        if hyper_image_sky is not None:
            image_1d = hyper_image_sky.image_scaled_sky_from_image(image=lens_data.image_1d)
        else:
            image_1d = lens_data.image_1d

        if hyper_noise_background is not None:
            noise_map_1d = hyper_noise_background.noise_map_scaled_noise_from_noise_map(noise_map=lens_data.noise_map_1d)
        else:
            noise_map_1d = lens_data.noise_map_1d

        hyper_noise_map_1d = tracer.hyper_noise_map_1d_from_noise_map_1d(
            noise_map_1d=lens_data.noise_map_1d)

        if hyper_noise_map_1d is not None:
            noise_map_1d = noise_map_1d + hyper_noise_map_1d

        inversion = tracer.inversion_from_image_1d_noise_map_1d_and_convolver_mapping_matrix(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            convolver_mapping_matrix=lens_data.convolver_mapping_matrix)

        super().__init__(
            lens_data=lens_data,
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            model_image_1d=inversion.reconstructed_data_1d,
            tracer=tracer,
            inversion=inversion)

    @property
    def galaxy_image_1d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_image_dict = self.tracer.galaxy_image_dict_blank_images
        galaxy_image_dict.update({self.tracer.planes[-1].galaxies[0] : self.inversion.reconstructed_data_1d})
        return galaxy_image_dict

    @property
    def figure_of_merit(self):
        return self.evidence

    @property
    def model_image_2d_of_planes(self):
        return [np.zeros(self.image_2d.shape), self.inversion.reconstructed_data_2d]


class LensProfileInversionFit(InversionFit):

    def __init__(self, lens_data, tracer, padded_tracer=None, hyper_image_sky=None, hyper_noise_background=None):
        """ An  lens profile and inversion fitter, which first generates and subtracts the image-plane \
        image of all galaxies (with light profiles) in the tracer, blurs it with the PSF and fits the residual image \
        with an inversion using the mapper(s) and regularization(s) in the galaxy's of the tracer.

        This inversion use's the lens-image, its PSF and an input noise-map.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lens_data : lens_data.LensData
            The lens-image that is fitted.
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.TracerNonStack or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens data's \
            padded grid_stack such that unmasked model-images can be computed.
        """

        if hyper_image_sky is not None:
            image_1d = hyper_image_sky.image_scaled_sky_from_image(image=lens_data.image_1d)
        else:
            image_1d = lens_data.image_1d

        if hyper_noise_background is not None:
            noise_map_1d = hyper_noise_background.noise_map_scaled_noise_from_noise_map(noise_map=lens_data.noise_map_1d)
        else:
            noise_map_1d = lens_data.noise_map_1d

        hyper_noise_map_1d = tracer.hyper_noise_map_1d_from_noise_map_1d(
            noise_map_1d=lens_data.noise_map_1d)

        if hyper_noise_map_1d is not None:
            noise_map_1d = noise_map_1d + hyper_noise_map_1d

        self.blurred_profile_image_1d = \
            tracer.blurred_profile_image_plane_image_1d_from_convolver_image(
                convolver_image=lens_data.convolver_image)

        self.profile_subtracted_image_1d = image_1d - self.blurred_profile_image_1d

        inversion = tracer.inversion_from_image_1d_noise_map_1d_and_convolver_mapping_matrix(
            image_1d=self.profile_subtracted_image_1d,
            noise_map_1d=noise_map_1d,
            convolver_mapping_matrix=lens_data.convolver_mapping_matrix)

        model_image = self.blurred_profile_image_1d + inversion.reconstructed_data_1d

        super(LensProfileInversionFit, self).__init__(
            tracer=tracer,
            padded_tracer=padded_tracer,
            lens_data=lens_data,
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            model_image_1d=model_image,
            inversion=inversion)

        self.convolver_image = lens_data.convolver_image

    @property
    def galaxy_image_1d_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_image_dict = self.tracer.galaxy_image_dict_from_convolver_image(convolver_image=self.convolver_image)
        galaxy_image_dict.update({self.tracer.planes[-1].galaxies[0] : self.inversion.reconstructed_data_1d})
        return galaxy_image_dict

    @property
    def blurred_profile_image_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.blurred_profile_image_1d)

    @property
    def profile_subtracted_image_2d(self):
        return self.scaled_array_2d_from_array_1d(array_1d=self.profile_subtracted_image_1d)

    @property
    def model_image_2d_of_planes(self):
        return [self.blurred_profile_image_2d, self.inversion.reconstructed_data_2d]

    @property
    def figure_of_merit(self):
        return self.evidence


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
        return list(map(lambda positions: self.max_separation_of_grid(positions), self.positions))

    @staticmethod
    def max_separation_of_grid(grid):
        rdist_max = np.zeros((grid.shape[0]))
        for i in range(grid.shape[0]):
            xdists = np.square(np.subtract(grid[i, 0], grid[:, 0]))
            ydists = np.square(np.subtract(grid[i, 1], grid[:, 1]))
            rdist_max[i] = np.max(np.add(xdists, ydists))
        return np.max(np.sqrt(rdist_max))

# TODO : The [plane_index][galaxy_index] datas structure is going to be key to tracking galaxies / hyper galaxies in
# TODO : Multi-plane ray tracing. I never felt it was easy to follow using list comprehensions from ray_tracing.
# TODO : Can we make this neater?
