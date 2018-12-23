import numpy as np

from autofit.core import fitter
from autolens import exc
from autolens.model.inversion import inversions
from autolens.lensing.util import lensing_fitters_util as util
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing


def fit_lensing_image_with_tracer(lensing_image, tracer, padded_tracer=None):
    """Fit a lensing image with a model tracer, automatically determining the type of fit based on the \
    properties of the galaxies in the tracer.

    Parameters
    -----------
    lensing_image : li.LensingImage or li.LensingHyperImage
        The lensing-images that is fitted.
    tracer : ray_tracing.AbstractTracerNonStack
        The tracer, which describes the ray-tracing and strong lensing configuration.
    padded_tracer : ray_tracing.Tracer or None
        A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
        padded grid_stack such that unmasked model-images can be computed.
    """

    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFitter(lensing_image=lensing_image, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileHyperFitter(lensing_hyper_image=lensing_image, tracer=tracer,
                                             padded_tracer=padded_tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFitter(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return LensingInversionHyperFitter(lensing_hyper_image=lensing_image, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFitter(lensing_image=lensing_image, tracer=tracer,
                                                 padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileInversionHyperFitter(lensing_hyper_image=lensing_image, tracer=tracer,
                                                      padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


class AbstractLensingFitter(object):

    def __init__(self, tracer, padded_tracer, map_to_scaled_array):
        """ An abstract lensing fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lensing image's data.

        Parameters
        -----------
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lensing configuration.
        padded_tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stack such that unmasked model-images can be computed.
        map_to_scaled_array : func
            A function which maps the 1D lensing data to its unmasked 2D array.
        """
        self.tracer = tracer
        self.padded_tracer = padded_tracer
        self.map_to_scaled_array = map_to_scaled_array

    @property
    def total_inversions(self):
        return len(self.tracer.mappers_of_planes)


class AbstractLensingProfileFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, tracer, padded_tracer):
        """ An abstract lensing profile fitter, which generates the image-plane image of all galaxies (with light \
        profiles) in the tracer and blurs it with the lensing image's PSF.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing and strong lensing configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        super(AbstractLensingProfileFitter, self).__init__(tracer=tracer, padded_tracer=padded_tracer,
                                                           map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.psf = lensing_image.psf
        self.convolver_image = lensing_image.convolver_image

        blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=self.convolver_image)

        self.blurred_profile_image = self.map_to_scaled_array(array_1d=blurred_profile_image_1d)

    @property
    def model_image_of_planes(self):
        return util.blurred_image_of_planes_from_1d_images_and_convolver(total_planes=self.tracer.total_planes,
                image_plane_image_1d_of_planes=self.tracer.image_plane_image_1d_of_planes,
                image_plane_blurring_image_1d_of_planes=self.tracer.image_plane_blurring_image_1d_of_planes,
                convolver=self.convolver_image, map_to_scaled_array=self.map_to_scaled_array)

    @property
    def unmasked_model_image(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=self.padded_tracer.image_plane.grid_stack, psf=self.psf,
                unmasked_image_1d=self.padded_tracer.image_plane_image_1d)

    @property
    def unmasked_model_image_of_planes_and_galaxies(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return util.unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(
                planes=self.padded_tracer.planes, padded_grid_stack=self.padded_tracer.image_plane.grid_stack, psf=self.psf)


class AbstractLensingInversionFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, noise_map_1d, tracer):
        """ An abstract lensing inversion fitter, which fits the lensing image an inversion using the mapper(s) and \
        regularization(s) in the galaxies of the tracer.

        This inversion use's the lensing-image, its PSF and an input noise_map-map.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        noise_map_1d : ndarray
            The 1D noise_map map that is fitted, which is an input variable so a hyper-noise_map map can be used (see \
            *AbstractHyperFitter*).
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lensing configuration.
        """
        super(AbstractLensingInversionFitter, self).__init__(tracer=tracer, padded_tracer=None,
                                                             map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image_1d=lensing_image.image_1d, noise_map_1d=noise_map_1d, convolver=lensing_image.convolver_mapping_matrix,
            mapper=tracer.mappers_of_planes[-1], regularization=tracer.regularizations_of_planes[-1])

    @property
    def unmasked_model_image(self):
        return None

    @property
    def model_images_of_planes(self):
        return [None, self.inversion.reconstructed_data]


class AbstractLensingProfileInversionFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, noise_map_1d, tracer, padded_tracer):
        """ An abstract lensing profile and inversion fitter, which first generates and subtracts the image-plane \
        image of all galaxies (with light profiles) in the tracer, blurs it with the PSF and fits the residual image \
        with an inversion using the mapper(s) and regularization(s) in the galaxy's of the tracer.

        This inversion use's the lensing-image, its PSF and an input noise_map-map.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        noise_map_1d : ndarray
            The 1D noise_map map that is fitted, which is an input variable so a hyper-noise_map map can be used (see \
            *AbstractHyperFitter*).
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lensing configuration.
        padded_tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        super(AbstractLensingProfileInversionFitter, self).__init__(tracer=tracer, padded_tracer=padded_tracer,
                                                           map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.psf = lensing_image.psf
        self.convolver_image = lensing_image.convolver_image

        blurred_profile_image_1d = util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lensing_image.convolver_image)

        self.blurred_profile_image = self.map_to_scaled_array(array_1d=blurred_profile_image_1d)

        profile_subtracted_image_1d = lensing_image.image_1d - blurred_profile_image_1d
        self.profile_subtracted_image = lensing_image.image - self.blurred_profile_image

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image_1d=profile_subtracted_image_1d, noise_map_1d=noise_map_1d,
            convolver=lensing_image.convolver_mapping_matrix, mapper=tracer.mappers_of_planes[-1],
            regularization=tracer.regularizations_of_planes[-1])

    @property
    def unmasked_model_image(self):
        return None

    @property
    def model_image_of_planes(self):
        return [self.blurred_profile_image, self.inversion.reconstructed_data]


class LensingDataFitter(fitter.DataFitter):

    def __init__(self, image, noise_map, mask, model_image):
        """Class to fit a lensing image with a model image.

        Parameters
        -----------
        image : ndarray
            The observed image that is fitted.
        noise_map : ndarray
            The noise_map-map of the observed image.
        mask: msk.Mask
            The mask that is applied to the image.
        model_data : ndarray
            The model image the oberved image is fitted with.
        """
        super(LensingDataFitter, self).__init__(data=image, noise_map=noise_map, mask=mask, model_data=model_image)

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood


class LensingDataInversionFitter(LensingDataFitter):

    def __init__(self, image, noise_map, mask, model_image, inversion):
        """Class to fit a lensing image with a inversion model image.

        Parameters
        -----------
        image : ndarray
            The observed image that is fitted.
        noise_map : ndarray
            The noise_map-map of the observed image.
        mask: msk.Mask
            The mask that is applied to the image.
        model_data : ndarray
            The model image the oberved image is fitted with.
        inversion : inversions.Inversion
            The inversion used to ofit the image.
        """
        super(LensingDataInversionFitter, self).__init__(image=image, noise_map=noise_map,
                                                         mask=mask, model_image=model_image)

        self.likelihood_with_regularization = \
            util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
            chi_squared=self.chi_squared, regularization_term=inversion.regularization_term,
            noise_normalization=self.noise_normalization)

        self.evidence = util.evidence_from_inversion_terms(chi_squared=self.chi_squared,
                                                                   regularization_term=inversion.regularization_term,
                                                                   log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
                                                                   log_regularization_term=inversion.log_det_regularization_matrix_term,
                                                                   noise_normalization=self.noise_normalization)

    @property
    def figure_of_merit(self):
        return self.evidence


class LensingProfileFitter(LensingDataFitter, AbstractLensingProfileFitter):

    def __init__(self, lensing_image, tracer, padded_tracer=None):
        """ Fit a lensing image with galaxy light-profiles, as follows:

        1) Generates the image-plane image of all galaxies with light profiles in the tracer.
        2) Blur this image-plane image with the lensing image's PSF to generate the model-image.
        3) Fit the observed with this model-image.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing and strong lensing configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        AbstractLensingProfileFitter.__init__(self=self, lensing_image=lensing_image, tracer=tracer,
                                              padded_tracer=padded_tracer)

        super(LensingProfileFitter, self).__init__(image=lensing_image.image, noise_map=lensing_image.noise_map,
                                                   mask=lensing_image.mask, model_image=self.blurred_profile_image)


class LensingInversionFitter(LensingDataInversionFitter, AbstractLensingInversionFitter):

    def __init__(self, lensing_image, tracer):
        """ Fit a lensing image with an inversion, as follows:

        1) Extract the mapper(s) and regularization(s) of galaxies in the tracer.
        2) Use these to perform an inversion on the lensing image (including PSF blurring) and generate the model-image.
        3) Fit the observed with this model-image.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lensing configuration.
        """

        AbstractLensingInversionFitter.__init__(self=self, lensing_image=lensing_image,
                                                noise_map_1d=lensing_image.noise_map_1d, tracer=tracer)

        super(LensingInversionFitter, self).__init__(image=lensing_image.image, noise_map=lensing_image.noise_map,
                                                     mask=lensing_image.mask,
                                                     model_image=self.inversion.reconstructed_data,
                                                     inversion=self.inversion)


class LensingProfileInversionFitter(LensingDataInversionFitter, AbstractLensingProfileInversionFitter):

    def __init__(self, lensing_image, tracer, padded_tracer=None):
        """ Fit a lensing image with galaxy light-profiles and an inversion, as follows:

        1) Generates the image-plane image of all galaxies with light profiles in the tracer.
        2) Blur this image-plane image with the lensing image's PSF to generate the model-image.
        3) Subtract this image from the observed image to generate a profile subtracted image.
        4) Extract the mapper(s) and regularization(s) of galaxies in the tracer.
        5) Use these to perform an inversion on the profile subtracted image (including PSF blurring).
        6) Add the blurred profile image and reconstructed inversion image together to generate the model-image.
        7) Fit the observed with this model-image.

        Typically, this fit is used to subtract the foreground lens's light using light profiles \
        and then fit the source galaxy with an inversion, however the fitting routine in general

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing and strong lensing configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stack such that unmasked model-images can be computed.
        """

        AbstractLensingProfileInversionFitter.__init__(self=self, lensing_image=lensing_image,
                                                       noise_map_1d=lensing_image.noise_map_1d, tracer=tracer,
                                                       padded_tracer=padded_tracer)

        model_image = self.blurred_profile_image + self.inversion.reconstructed_data

        super(LensingProfileInversionFitter, self).__init__(image=lensing_image.image, noise_map=lensing_image.noise_map,
                                                            mask=lensing_image.mask, model_image=model_image,
                                                            inversion=self.inversion)


class AbstractLensingHyperFitter(object):

    def __init__(self, lensing_hyper_image, hyper_galaxies):
        """Abstract base class of a hyper-fit.

        A hyper-fit is a fit which performs the fit  described in the *AbstractFitter*, but also includes a set of \
        parameters which allow the noise_map-map of the image to be scaled. This prevents over-fitting of small regions of \
        an image with high chi-squared values, ensuring the model gives global fit to the data.

        This is performed using existing model-images of the observed image, from which a 'contribution-map' is \
        computed and used to increase the noise_map in localized regions of the noise_map-map.

        Parameters
        -----------
        lensing_hyper_image : [fit_data.FitDataHyper]
            The fitting image that are fitted, which include the hyper-image used for scaling the noise_map-map.
        hyper_galaxies : [galaxy.Galaxy]
            The hyper-galaxies which represent the model components used to scale the noise_map, which correspond to
            individual galaxies in the image.

        Attributes
        -----------
        contribution_maps : [scaled_array.ScaledSquarePixelArray]
            The contribution map of every image, where there is an individual contribution map for each hyper-galaxy in
            the model.
        hyper_noise_map : scaled_array.ScaledSquarePixelArray
            The hyper noise_map map of the image, computed after using the hyper-galaxies.
        """

        self.is_hyper_fit = True

        contribution_maps_1d = \
            util.contribution_maps_1d_from_hyper_images_and_galaxies(
                hyper_model_image_1d=lensing_hyper_image.hyper_model_image_1d,
                hyper_galaxy_images_1d=lensing_hyper_image.hyper_galaxy_images_1d,
                hyper_galaxies=hyper_galaxies, hyper_minimum_values=lensing_hyper_image.hyper_minimum_values)

        self.contribution_maps = list(map(lambda contribution_map_1d :
                                          lensing_hyper_image.map_to_scaled_array(array_1d=contribution_map_1d),
                                           contribution_maps_1d))

        self.hyper_noise_map_1d =\
            util.scaled_noise_map_from_hyper_galaxies_and_contribution_maps(
                contribution_maps=contribution_maps_1d, hyper_galaxies=hyper_galaxies,
                noise_map=lensing_hyper_image.noise_map_1d)

        self.hyper_noise_map = lensing_hyper_image.map_to_scaled_array(array_1d=self.hyper_noise_map_1d)


class LensingProfileHyperFitter(LensingDataFitter, AbstractLensingProfileFitter, AbstractLensingHyperFitter):

    def __init__(self, lensing_hyper_image, tracer, padded_tracer=None):
        """ Fit a lensing hyper-image with galaxy light-profiles, as follows:

        1) Use the hyper-image and tracer's hyper-galaxies to generate a hyper noise_map-map.
        2) Generates the image-plane image of all galaxies with light profiles in the tracer.
        3) Blur this image-plane image with the lensing image's PSF to generate the model-image.
        4) Fit the observed with this model-image, using the hyper-noise_map map to compute the chi-squared values..

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        ----------
        lensing_hyper_image : li.LensingHyperImage
            List of the lensing hyper image that are to be fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stacks such that unmasked model-image can be computed.
        """

        AbstractLensingHyperFitter.__init__(self=self, lensing_hyper_image=lensing_hyper_image,
                                            hyper_galaxies=tracer.hyper_galaxies)

        AbstractLensingProfileFitter.__init__(self=self, lensing_image=lensing_hyper_image, tracer=tracer,
                                              padded_tracer=padded_tracer)

        super(LensingProfileHyperFitter, self).__init__(image=lensing_hyper_image.image, noise_map=self.hyper_noise_map,
                                                  mask=lensing_hyper_image.mask, model_image=self.blurred_profile_image)


class LensingInversionHyperFitter(LensingDataInversionFitter, AbstractLensingInversionFitter,
                                  AbstractLensingHyperFitter):

    def __init__(self, lensing_hyper_image, tracer):
        """ Fit a lensing hyper-image with an inversion, as follows:

        1) Use the hyper-image and tracer's hyper-galaxies to generate a hyper noise_map-map.
        2) Extract the mapper(s) and regularization(s) of galaxies in the tracer.
        3) Use these to perform an inversion on the lensing image (including PSF blurring), using the hyper noise_map-map, \
           and generate the model-image.
        4) Fit the observed with this model-image, using the hyper-noise_map map to compute the chi-squared values.

        Parameters
        ----------
        lensing_image : li.LensingHyperImage
            List of the lensing-image that are to be fitted.
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        """

        AbstractLensingHyperFitter.__init__(self=self, lensing_hyper_image=lensing_hyper_image,
                                            hyper_galaxies=tracer.hyper_galaxies)

        AbstractLensingInversionFitter.__init__(self=self, lensing_image=lensing_hyper_image,
                                                noise_map_1d=self.hyper_noise_map_1d, tracer=tracer)

        super(LensingInversionHyperFitter, self).__init__(image=lensing_hyper_image.image,
                                                          noise_map=self.hyper_noise_map, mask=lensing_hyper_image.mask,
                                                          model_image=self.inversion.reconstructed_data,
                                                          inversion=self.inversion)


class LensingProfileInversionHyperFitter(LensingDataInversionFitter, AbstractLensingProfileInversionFitter,
                                         AbstractLensingHyperFitter):

    def __init__(self, lensing_hyper_image, tracer, padded_tracer=None):
        """Fit a lensing hyper-image with galaxy light-profiles and an inversion, as follows:

        1) Use the hyper-image and tracer's hyper-galaxies to generate a hyper noise_map-map.
        2) Generates the image-plane image of all galaxies with light profiles in the tracer.
        3) Blur this image-plane image with the lensing image's PSF to generate the model-image.
        4) Subtract this image from the observed image to generate a profile subtracted image.
        5) Extract the mapper(s) and regularization(s) of galaxies in the tracer.
        6) Use these to perform an inversion on the profile subtracted image (including PSF blurring), using the \
           hyper noise_map-map in the inversion.
        7) Add the blurred profile image and reconstructed inversion image together to generate the model-image.
        8) Fit the observed with this model-image, using the hyper-noise_map map to compute the chi-squared values.

        Typically, this fit is used to subtract the foreground lens's light using light profiles \
        and then fit the source galaxy with an inversion, however the fitting routine in general

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        ----------
        lensing_images : li.LensingImage
            List of the lensing-image that are to be fitted.
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        padded_tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stacks such that unmasked model-image can be computed.
        """

        AbstractLensingHyperFitter.__init__(self=self, lensing_hyper_image=lensing_hyper_image,
                                            hyper_galaxies=tracer.hyper_galaxies)

        AbstractLensingProfileInversionFitter.__init__(self=self, lensing_image=lensing_hyper_image,
                                                       noise_map_1d=self.hyper_noise_map_1d, tracer=tracer,
                                                       padded_tracer=padded_tracer)

        model_image = self.blurred_profile_image + self.inversion.reconstructed_data

        super(LensingProfileInversionHyperFitter, self).__init__(image=lensing_hyper_image.image,
                                                                 noise_map=self.hyper_noise_map,
                                                                 mask=lensing_hyper_image.mask, model_image=model_image,
                                                                 inversion=self.inversion)


class LensingPositionFitter(object):

    def __init__(self, positions, noise_map):
        """A lensing position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher likelihood.

        Parameters
        -----------
        positions : [[]]
            The (y,x) arc-second coordinates of pisitions which the maximum distance and likelihood is computed using.
        noise_map : ndarray
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
        if max(self.maximum_separations) > threshold:
            return False
        else:
            return True

    @property
    def maximum_separations(self):
        return list(map(lambda positions: self.max_separation_of_grid(positions), self.positions))

    def max_separation_of_grid(self, grid):
        rdist_max = np.zeros((grid.shape[0]))
        for i in range(grid.shape[0]):
            xdists = np.square(np.subtract(grid[i, 0], grid[:, 0]))
            ydists = np.square(np.subtract(grid[i, 1], grid[:, 1]))
            rdist_max[i] = np.max(np.add(xdists, ydists))
        return np.max(np.sqrt(rdist_max))





# TODO : The [plane_index][galaxy_index] datas structure is going to be key to tracking galaxies / hyper galaxies in
# TODO : Multi-plane ray tracing. I never felt it was easy to follow using list comprehensions from ray_tracing.
# TODO : Can we make this neater?