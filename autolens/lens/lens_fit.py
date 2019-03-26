import numpy as np

from autofit.tools import fit
from autolens import exc
from autolens.lens.util import lens_fit_util as util
from autolens.model.inversion import inversions


class LensDataFit(fit.DataFit):
    def __init__(self, image, noise_map, mask, model_image):
        super().__init__(data=image, noise_map=noise_map, mask=mask, model_data=model_image)

    @property
    def model_image(self):
        return self.model_data

    @property
    def image(self):
        return self.data

    @property
    def figure_of_merit(self):
        return self.likelihood

    @classmethod
    def for_data_and_tracer(cls, lens_data, tracer, padded_tracer=None):
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
            return LensProfileFit(lens_data=lens_data, tracer=tracer, padded_tracer=padded_tracer)
        elif not tracer.has_light_profile and tracer.has_pixelization:
            return LensInversionFit(lens_data=lens_data, tracer=tracer, padded_tracer=None)
        elif tracer.has_light_profile and tracer.has_pixelization:
            return LensProfileInversionFit(lens_data=lens_data, tracer=tracer, padded_tracer=None)
        else:
            raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                       'properties of the tracer')


class LensTracerFit(LensDataFit):

    def __init__(self, image, noise_map, mask, model_image, tracer, psf, map_to_scaled_array, padded_tracer=None):
        """ An  lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens data's hyper.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.TracerNonStack or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens data's \
            padded grid_stack such that unmasked model-images can be computed.
        map_to_scaled_array : func
            A function which maps the 1D lens hyper to its unmasked 2D array.
        """
        # Add hyper noise if these is a padded tracer...
        if padded_tracer is not None:
            original_noise_map = noise_map.copy()
            unmasked_model_image = padded_tracer.image_plane.grid_stack. \
                unmasked_blurred_image_from_psf_and_unmasked_image(psf=psf,
                                                                   unmasked_image_1d=padded_tracer.
                                                                   image_plane_image_1d)
            # ...to any galaxy with a corresponding HyperGalaxy
            for galaxy in tracer.galaxies:
                if galaxy.hyper_galaxy is not None:
                    plane = padded_tracer.plane_with_galaxy(galaxy)
                    unmasked_galaxy_image = plane.unmasked_blurred_image_of_galaxy_with_grid_stack_psf(
                        galaxy,
                        padded_tracer.image_plane.grid_stack,
                        psf)
                    hyper_galaxy = galaxy.hyper_galaxy
                    hyper_noise = hyper_galaxy.hyper_noise_from_model_image_galaxy_image_and_noise_map(
                        unmasked_model_image,
                        unmasked_galaxy_image,
                        original_noise_map)
                    noise_map += hyper_noise
        super().__init__(image=image, noise_map=noise_map, mask=mask, model_image=model_image)
        self.tracer = tracer
        self.padded_tracer = padded_tracer
        self.psf = psf
        self.map_to_scaled_array = map_to_scaled_array

    @property
    def total_inversions(self):
        return len(self.tracer.mappers_of_planes)

    @property
    def unmasked_model_image(self):
        if self.padded_tracer is not None and not self.padded_tracer.has_pixelization:
            return self.padded_tracer.image_plane.grid_stack.unmasked_blurred_image_from_psf_and_unmasked_image(
                psf=self.psf,
                unmasked_image_1d=self.padded_tracer.image_plane_image_1d)

    @property
    def unmasked_model_image_of_planes(self):
        if self.padded_tracer is not None:
            return util.unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(
                planes=self.padded_tracer.planes, padded_grid_stack=self.padded_tracer.image_plane.grid_stack,
                psf=self.psf)

    def unmasked_model_image_for_galaxy(self, galaxy):
        plane = self.padded_tracer.plane_with_galaxy(galaxy)
        return plane.unmasked_blurred_image_of_galaxy_with_grid_stack_psf(galaxy,
                                                                          self.padded_tracer.image_plane.grid_stack,
                                                                          self.psf)

    @property
    def unmasked_model_image_of_planes_and_galaxies(self):
        if self.padded_tracer is not None:
            return util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(
                planes=self.padded_tracer.planes, padded_grid_stack=self.padded_tracer.image_plane.grid_stack,
                psf=self.psf)


class LensProfileFit(LensTracerFit):

    def __init__(self, lens_data, tracer, padded_tracer=None):
        """ An  lens profile fitter, which generates the image-plane image of all galaxies (with light \
        profiles) in the tracer and blurs it with the lens data's PSF.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lens_data : lens_data.LensData
            The lens-image that is fitted.
        tracer : ray_tracing.TracerNonStack
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens data's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lens_data.convolver_image)
        super(LensProfileFit, self).__init__(
            image=lens_data.image, noise_map=lens_data.noise_map,
            mask=lens_data.mask, model_image=lens_data.map_to_scaled_array(
                array_1d=blurred_profile_image_1d),
            tracer=tracer, padded_tracer=padded_tracer, psf=lens_data.psf,
            map_to_scaled_array=lens_data.map_to_scaled_array)

        self.convolver_image = lens_data.convolver_image

    @property
    def model_image_of_planes(self):
        return util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=self.tracer.total_planes,
            image_plane_image_1d_of_planes=self.tracer.image_plane_image_1d_of_planes,
            image_plane_blurring_image_1d_of_planes=self.tracer.image_plane_blurring_image_1d_of_planes,
            convolver=self.convolver_image,
            map_to_scaled_array=self.map_to_scaled_array)

    @property
    def figure_of_merit(self):
        return self.likelihood


class InversionFit(LensTracerFit):
    def __init__(self, lens_data, model_image, tracer, inversion, padded_tracer=None):
        super().__init__(image=lens_data.image, noise_map=lens_data.noise_map,
                         mask=lens_data.mask,
                         model_image=model_image,
                         psf=lens_data.psf,
                         tracer=tracer,
                         padded_tracer=padded_tracer,
                         map_to_scaled_array=lens_data.map_to_scaled_array)
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

    def __init__(self, lens_data, tracer, padded_tracer=None):
        """ An  lens inversion fitter, which fits the lens data an inversion using the mapper(s) and \
        regularization(s) in the galaxies of the tracer.

        This inversion use's the lens-image, its PSF and an input noise-map.

        Parameters
        -----------
        lens_data : lens_data.LensData
            The lens-image that is fitted.
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lens configuration.
        """

        inversion = inversions.inversion_from_image_mapper_and_regularization(
            image_1d=lens_data.image_1d, noise_map_1d=lens_data.noise_map_1d,
            convolver=lens_data.convolver_mapping_matrix,
            mapper=tracer.mappers_of_planes[-1], regularization=tracer.regularizations_of_planes[-1])
        super().__init__(lens_data=lens_data,
                         model_image=inversion.reconstructed_data,
                         tracer=tracer,
                         inversion=inversion,
                         padded_tracer=padded_tracer)

    @property
    def figure_of_merit(self):
        return self.evidence

    @property
    def model_image_of_planes(self):
        return [None, self.inversion.reconstructed_data]


class LensProfileInversionFit(InversionFit):

    def __init__(self, lens_data, tracer, padded_tracer=None):
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
        blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lens_data.convolver_image)

        blurred_profile_image = lens_data.map_to_scaled_array(array_1d=blurred_profile_image_1d)
        profile_subtracted_image_1d = lens_data.image_1d - blurred_profile_image_1d
        inversion = inversions.inversion_from_image_mapper_and_regularization(
            image_1d=profile_subtracted_image_1d, noise_map_1d=lens_data.noise_map_1d,
            convolver=lens_data.convolver_mapping_matrix, mapper=tracer.mappers_of_planes[-1],
            regularization=tracer.regularizations_of_planes[-1])

        model_image = blurred_profile_image + inversion.reconstructed_data
        super(LensProfileInversionFit, self).__init__(tracer=tracer,
                                                      padded_tracer=padded_tracer,
                                                      lens_data=lens_data,
                                                      model_image=model_image,
                                                      inversion=inversion)

        self.convolver_image = lens_data.convolver_image

        self.blurred_profile_image = blurred_profile_image
        self.profile_subtracted_image = lens_data.image - self.blurred_profile_image

    @property
    def model_image_of_planes(self):
        return [self.blurred_profile_image, self.inversion.reconstructed_data]

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
