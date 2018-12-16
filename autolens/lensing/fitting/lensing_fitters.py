import numpy as np

from autolens import exc
from autolens.lensing.fitting import lensing_fitting_util
from autolens.data.fitting import fitter
from autolens.model.inversion import inversions
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing

minimum_value_profile = 0.1


def fit_lensing_image_with_tracer(lensing_image, tracer, padded_tracer=None):
    """Fit a lensing regular with a model tracer, automatically determining the type of fit based on the \
    properties of the galaxies in the tracer.

    Parameters
    -----------
    lensing_image : li.LensingImage or li.LensingHyperImage
        List of the lensing-unblurred_image_1d that are to be fitted.
    tracer : ray_tracing.AbstractTracerNonStack
        The tracer, which describes the ray-tracing of the strong lensing configuration.
    padded_tracer : ray_tracing.AbstractTracerNonStack or None
        A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
        padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
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
            return HyperLensingInversionFit(lensing_hyper_image=lensing_image, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFitter(lensing_image=lensing_image, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit(lensing_hyper_image=lensing_image, tracer=tracer,
                                                   padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit_normal routine did not call a Fit class - check the '
                                   'properties of the tracer_normal')


def fast_fit_from_lensing_image_and_tracer(lensing_image, tracer):
    """Fit a lensing regular with a model tracer, automatically determining the type of fit based on the properties of \
     the galaxies in the tracer.

    The likelihood is computed in the fastest, least memory-intensive, way possible, for efficient non-linear sampling.

    Parameters
    -----------
    lensing_image : [li.LensingImage] or [li.LensingHyperImage]
        List of the lensing-unblurred_image_1d that are to be fitted.
    tracer : ray_tracing.AbstractTracer
        The tracer, which describes the ray-tracing of the strong lensing configuration.
    padded_tracer : ray_tracing.AbstractTracer
        A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
        padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
    """
    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFitter.fast_fit(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileHyperFitter.fast_fit(lensing_hyper_image=lensing_image, tracer=tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFitter.fast_fit(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit.fast_scaled_evidence(lensing_hyper_image=lensing_image, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFitter.fast_fit(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit.fast_scaled_evidence(lensing_hyper_image=lensing_image,
                                                                        tracer=tracer)

    else:

        raise exc.FittingException('The fast likelihood routine did not call a likelihood functions - check the '
                                   'properties of the tracer_normal')


class AbstractLensingFitter(object):

    def __init__(self, tracer, padded_tracer, map_to_scaled_array):

        self.tracer = tracer
        self.padded_tracer = padded_tracer
        self.map_to_scaled_array = map_to_scaled_array

    @property
    def total_inversions(self):
        return len(self.tracer.mappers_of_planes)


class AbstractLensingProfileFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, tracer, padded_tracer):

        super(AbstractLensingProfileFitter, self).__init__(tracer=tracer, padded_tracer=padded_tracer,
                                                           map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.psf = lensing_image.psf
        self.convolver_image = lensing_image.convolver_image
        self.blurred_profile_image = lensing_fitting_util.blurred_image_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=self.convolver_image, map_to_scaled_array=self.map_to_scaled_array)

    @property
    def model_image_of_planes(self):
        return lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=self.tracer,
               convolver_image=self.convolver_image, map_to_scaled_array=self.map_to_scaled_array)

    @property
    def unmasked_model_image(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return lensing_fitting_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=self.padded_tracer.image_plane.grid_stack, psf=self.psf,
                unmasked_image_1d=self.padded_tracer.image_plane_image_1d)

    @property
    def unmasked_model_image_of_galaxies(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return lensing_fitting_util.unmasked_blurred_image_of_galaxies_from_padded_grid_stack_psf_and_tracer(
                padded_grid_stack=self.padded_tracer.image_plane.grid_stack, psf=self.psf, tracer=self.padded_tracer)


class AbstractLensingInversionFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, noise_map_1d, tracer):
        """Abstract base class for a fit to an regular which uses a linear inversion.

        This includes passing the regular / noise_maps / PSF and inversion objects to the inversons module to perform \
        the inversion.

        Parameters
        -----------
        lensing_image : li.LensingImage
            List of the lensing-unblurred_image_1d that are to be fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing of the strong lensing configuration.

        Attributes
        -----------
        likelihoods_with_regularization : [float]
            The likelihood of each fit to the regular, including jsut 3 terms, the chi-squared term, regularization
            penalty factor and noise normalization term.
        likelihood_with_regularization : float
            The sum of all likelihoods_with_regularization for all unblurred_image_1d.
        evidences : [float]
            The Bayesian evidence of each fit to the regular. The Bayesian evidence is described in Suyu et al. 2006 and
            the howtolens inversion tutorial chapter_4_inversions/tutorial_4_bayesian_regularization.
        evidence : float
            The sum of evidence values for all unblurred_image_1d.
        """

        super(AbstractLensingInversionFitter, self).__init__(tracer=tracer, padded_tracer=None,
                                                             map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image_1d=lensing_image.image_1d, noise_map_1d=noise_map_1d, convolver=lensing_image.convolver_mapping_matrix,
            mapper=tracer.mappers_of_planes[-1], regularization=tracer.regularizations_of_planes[-1])


class AbstractLensingProfileInversionFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, noise_map_1d, tracer, padded_tracer):

        super(AbstractLensingProfileInversionFitter, self).__init__(tracer=tracer, padded_tracer=padded_tracer,
                                                           map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.psf = lensing_image.psf
        self.convolver_image = lensing_image.convolver_image

        blurred_profile_image_1d = lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
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
    def model_image_of_planes(self):
        return [self.blurred_profile_image, self.inversion.reconstructed_data]


class LensingDataFitter(fitter.DataFitter):

    def __init__(self, image, noise_map, mask, model_image):

        super(LensingDataFitter, self).__init__(data=np.asarray(image), noise_map=np.asarray(noise_map),
                                                mask=np.asarray(mask), model_data=np.asarray(model_image))

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data


class LensingDataInversionFitter(LensingDataFitter):

    def __init__(self, image, noise_map, mask, model_image, inversion):

        super(LensingDataFitter, self).__init__(data=np.asarray(image), noise_map=np.asarray(noise_map),
                                                mask=np.asarray(mask), model_data=np.asarray(model_image))

        self.likelihood_with_regularization = \
            lensing_fitting_util.likelihood_with_regularization_from_chi_squared_term_regularization_and_noise_term(
            chi_squared_term=self.chi_squared_term, regularization_term=inversion.regularization_term,
            noise_term=self.noise_term)

        self.evidence = lensing_fitting_util.evidence_from_reconstruction_terms(chi_squared_term=self.chi_squared_term,
                        regularization_term=inversion.regularization_term,
                        log_covariance_regularization_term=inversion.log_det_curvature_reg_matrix_term,
                        log_regularization_term=inversion.log_det_regularization_matrix_term,
                        noise_term=self.noise_term)


class LensingProfileFitter(LensingDataFitter, AbstractLensingProfileFitter):

    def __init__(self, lensing_image, tracer, padded_tracer=None):
        """
        Class to evaluate the fit_normal between a model described by a tracer_normal and an actual lensing_image.

        Parameters
        ----------
        lensing_image : li.LensingImage
            List of the lensing-unblurred_image_1d that are to be fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        padded_tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
            padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
        """

        AbstractLensingProfileFitter.__init__(self=self, lensing_image=lensing_image, tracer=tracer,
                                              padded_tracer=padded_tracer)

        super(LensingProfileFitter, self).__init__(image=lensing_image.image, noise_map=lensing_image.noise_map,
                                                   mask=lensing_image.mask, model_image=self.blurred_profile_image)

    @classmethod
    def fast_fit(cls, lensing_image, tracer):
        """Perform the fit of this class as described above, but storing no results as class instances, thereby \
        minimizing memory use and maximizing run-speed."""

        blurred_profile_image_1d = lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lensing_image.convolver_image)

        fit = LensingDataFitter(image=lensing_image.image_1d, noise_map=lensing_image.noise_map_1d,
                                mask=lensing_image.mask_1d, model_image=blurred_profile_image_1d)
        return fit.likelihood


class LensingInversionFitter(LensingDataInversionFitter, AbstractLensingInversionFitter):

    def __init__(self, lensing_image, tracer):
        """
        Class to evaluate the fit_normal between a model described by a tracer_normal and an actual lensing_image.

        Parameters
        ----------
        lensing_image : li.LensingImage
            List of the lensing-unblurred_image_1d that are to be fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        """

        AbstractLensingInversionFitter.__init__(self=self, lensing_image=lensing_image,
                                                noise_map_1d=lensing_image.noise_map_1d, tracer=tracer)

        super(LensingInversionFitter, self).__init__(image=lensing_image.image, noise_map=lensing_image.noise_map,
                                                     mask=lensing_image.mask,
                                                     model_image=self.inversion.reconstructed_data,
                                                     inversion=self.inversion)

    @classmethod
    def fast_fit(cls, lensing_image, tracer):
        """Perform the fit of this class as described above, but storing no results as class instances, thereby \
        minimizing memory use and maximizing run-speed."""

        inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image_1d=lensing_image.image_1d, noise_map_1d=lensing_image.noise_map_1d, 
            convolver=lensing_image.convolver_mapping_matrix, mapper=tracer.mappers_of_planes[-1], 
            regularization=tracer.regularizations_of_planes[-1])

        fit = LensingDataInversionFitter(image=lensing_image.image_1d, noise_map=lensing_image.noise_map_1d,
                                         mask=lensing_image.mask_1d, model_image=inversion.reconstructed_data_vector,
                                         inversion=inversion)

        return fit.evidence

    @property
    def model_images_of_planes(self):
        return [None, self.model_image]


class LensingProfileInversionFitter(LensingDataInversionFitter, AbstractLensingProfileInversionFitter):

    def __init__(self, lensing_image, tracer, padded_tracer=None):
        """
        Class to evaluate the fit_normal between a model described by a tracer_normal and an actual lensing_image.

        Parameters
        ----------
        lensing_images : li.LensingImage
            List of the lensing-unblurred_image_1d that are to be fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        padded_tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
            padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
        """

        AbstractLensingProfileInversionFitter.__init__(self=self, lensing_image=lensing_image,
                                                       noise_map_1d=lensing_image.noise_map_1d, tracer=tracer,
                                                       padded_tracer=padded_tracer)

        model_image = self.blurred_profile_image + self.inversion.reconstructed_data

        super(LensingProfileInversionFitter, self).__init__(image=lensing_image.image, noise_map=lensing_image.noise_map,
                                                            mask=lensing_image.mask, model_image=model_image,
                                                            inversion=self.inversion)

    @classmethod
    def fast_fit(cls, lensing_image, tracer):
        """Perform the fit of this class as described above, but storing no results as class instances, thereby \
        minimizing memory use and maximizing run-speed."""

        blurred_profile_image_1d = lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lensing_image.convolver_image)

        profile_subtracted_image_1d = lensing_image.image_1d - blurred_profile_image_1d

        inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image_1d=profile_subtracted_image_1d, noise_map_1d=lensing_image.noise_map_1d,
            convolver=lensing_image.convolver_mapping_matrix, mapper=tracer.mappers_of_planes[-1],
            regularization=tracer.regularizations_of_planes[-1])

        fit = LensingDataInversionFitter(image=lensing_image.image_1d, noise_map=lensing_image.noise_map_1d,
                                         mask=lensing_image.mask_1d,
                                         model_image=blurred_profile_image_1d + inversion.reconstructed_data_vector,
                                         inversion=inversion)

        return fit.evidence


class AbstractLensingHyperFitter(object):

    def __init__(self, lensing_hyper_image, hyper_galaxies):
        """Abstract base class of a hyper-fit.

        A hyper-fit is a fit which performs a fit as described in the *AbstractFitter*, but also includes a set of
        parameters which allow the noise-map of the datas-set to be scaled. This is done to prevent over-fitting
        small regions of a datas-set with high chi-squared values and therefore provide a global fit to the overall
        datas-set.

        This is performed using an existing model of the datas-set to compute a contributions regular, which a set of
        hyper-parameters then use to increase the noise in localized regions of the datas-set.

        Parameters
        -----------
        lensing_hyper_image : [fit_data.FitDataHyper]
            The fitting unblurred_image_1d that are fitted, which include the hyper-unblurred_image_1d used for scaling the noise-map.
        hyper_galaxies : [galaxy.Galaxy]
            The hyper-galaxies which represent the model components used to scale the noise, which correspond to
            individual galaxies in the regular.

        Attributes
        -----------
        contributions : [scaled_array.ScaledSquarePixelArray]
            The contribution map of every regular, where there is an individual contribution map for each hyper-galaxy in
            the model.
        hyper_noise_map : scaled_array.ScaledSquarePixelArray
            The hyper noise map of the image, computed after using the hyper-galaxies.
        """

        self.is_hyper_fit = True

        contributions_1d = \
            lensing_fitting_util.contributions_from_hyper_images_and_galaxies(
                hyper_model_image_1d=lensing_hyper_image.hyper_model_image_1d,
                hyper_galaxy_images_1d=lensing_hyper_image.hyper_galaxy_images_1d,
                hyper_galaxies=hyper_galaxies, hyper_minimum_values=lensing_hyper_image.hyper_minimum_values)

        self.contributions = list(map(lambda contribution_1d :
                                      lensing_hyper_image.map_to_scaled_array(array_1d=contribution_1d),
                                      contributions_1d))

        hyper_noise_map_1d =\
            lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
                contributions_1d=contributions_1d, hyper_galaxies=hyper_galaxies,
                noise_map_1d=lensing_hyper_image.noise_map_1d)

        self.hyper_noise_map = lensing_hyper_image.map_to_scaled_array(array_1d=hyper_noise_map_1d)


class LensingProfileHyperFitter(LensingDataFitter, AbstractLensingProfileFitter, AbstractLensingHyperFitter):

    def __init__(self, lensing_hyper_image, tracer, padded_tracer=None):
        """
        Class to evaluate the fit_normal between a model described by a tracer_normal and an actual lensing_image.

        Parameters
        ----------
        lensing_hyper_image : li.LensingHyperImage
            List of the lensing hyper unblurred_image_1d that are to be fitted.
        tracer : ray_tracing.AbstractTracerNonStack
            The tracer, which describes the ray-tracing of the strong lensing configuration.
        tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
            padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
        """

        AbstractLensingHyperFitter.__init__(self=self, lensing_hyper_image=lensing_hyper_image,
                                            hyper_galaxies=tracer.hyper_galaxies)

        AbstractLensingProfileFitter.__init__(self=self, lensing_image=lensing_hyper_image, tracer=tracer,
                                              padded_tracer=padded_tracer)

        super(LensingProfileHyperFitter, self).__init__(image=lensing_hyper_image.image, noise_map=self.hyper_noise_map,
                                                  mask=lensing_hyper_image.mask, model_image=self.blurred_profile_image)

    @classmethod
    def fast_fit(cls, lensing_hyper_image, tracer):
        """Perform the fit of this class as described above, but storing no results as class instances, thereby \
        minimizing memory use and maximizing run-speed."""

        contributions_1d = \
            lensing_fitting_util.contributions_from_hyper_images_and_galaxies(
                hyper_model_image_1d=lensing_hyper_image.hyper_model_image_1d,
                hyper_galaxy_images_1d=lensing_hyper_image.hyper_galaxy_images_1d,
                hyper_galaxies=tracer.hyper_galaxies, hyper_minimum_values=lensing_hyper_image.hyper_minimum_values)

        hyper_noise_map_1d = lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
                contributions_1d=contributions_1d, hyper_galaxies=tracer.hyper_galaxies,
                noise_map_1d=lensing_hyper_image.noise_map_1d)

        model_image_1d = lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=lensing_hyper_image.convolver_image)

        fit = fitter.DataFitter(data=lensing_hyper_image.image_1d, noise_map=hyper_noise_map_1d,
                                mask=lensing_hyper_image.mask_1d, model_data=model_image_1d)

        return fit.likelihood


#
#
# class HyperLensingInversion(fitter.AbstractLensingHyperFitter):
#
#     @property
#     def scaled_model_images(self):
#         return fitting_util.map_arrays_to_scaled_arrays(arrays_=self.scaled_model_images_,
#                                                    map_to_scaled_arrays=self.map_to_scaled_arrays)
#
#     @property
#     def scaled_residuals(self):
#         return fitting_util.map_arrays_to_scaled_arrays(arrays_=self.scaled_residuals_,
#                                                    map_to_scaled_arrays=self.map_to_scaled_arrays)
#
#     @property
#     def scaled_evidences(self):
#         return fitting_util.evidence_from_reconstruction_terms(self.scaled_chi_squared_terms,
#                                                                [self.scaled_inversion.regularization_term],
#                                                                [self.scaled_inversion.log_det_curvature_reg_matrix_term],
#                                                                [self.scaled_inversion.log_det_regularization_matrix_term],
#                                                                self.scaled_noise_terms)
#
#     @property
#     def scaled_evidence(self):
#         return sum(self.scaled_evidences)
#
#     @property
#     def scaled_model_image(self):
#         return self.scaled_model_images[0]
#
#     @property
#     def scaled_residual(self):
#         return self.scaled_residuals[0]
#
#
# class HyperLensingInversionFit(LensingInversionFitter, HyperLensingInversion):
#
#     def __init__(self, lensing_hyper_images, tracer):
#         """
#         Class to evaluate the fit_normal between a model described by a tracer_normal and an actual lensing_image.
#
#         Parameters
#         ----------
#         lensing_hyper_images : [li.LensingHyperImage]
#             List of the lensing hyper unblurred_image_1d that are to be fitted.
#         tracer : ray_tracing.AbstractTracer
#             The tracer, which describes the ray-tracing of the strong lensing configuration.
#         """
#
#         LensingDataFitter.__init__(self=self, lensing_images=lensing_hyper_images, tracer=tracer)
#         fitter.AbstractLensingHyperFitter.__init__(self=self, fitting_hyper_images=lensing_hyper_images,
#                                          hyper_galaxies=tracer.hyper_galaxies)
#         super(HyperLensingInversionFit, self).__init__(lensing_hyper_images, tracer)
#
#         self.scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
#             lensing_hyper_images[0][:], self.scaled_noise_map_1d[0], lensing_hyper_images[0].convolver_mapping_matrix,
#             self.inversion.mapper, self.inversion.regularization)
#
#         self.scaled_model_images_ = [self.scaled_inversion.reconstructed_data_vector]
#         self.scaled_residuals_ = fitting_util.residuals_from_data_mask_and_model_data(self.datas_, self.scaled_model_images_)
#         self.scaled_chi_squareds_ = fitting_util.chi_squareds_from_residuals_and_noise_map(self.scaled_residuals_,
#                                                                                            self.scaled_noise_map_1d)
#
#     @classmethod
#     def fast_scaled_evidence(cls, lensing_hyper_images, tracer):
#         """Perform the fit of this class as described above, but storing no results as class instances, thereby \
#         minimizing memory use and maximizing run-speed."""
#
#         contributions_1d = fitting_util.contributions_from_fitting_hyper_images_and_hyper_galaxies(
#             fitting_hyper_images=lensing_hyper_images, hyper_galaxies=tracer.hyper_galaxies)
#
#         scaled_noise_map_1d = fitting_util.scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(
#             fitting_hyper_images=lensing_hyper_images, contributions_1d=contributions_1d,
#             hyper_galaxies=tracer.hyper_galaxies)
#
#         convolvers = list(map(lambda lensing_image : lensing_image.convolver_mapping_matrix, lensing_hyper_images))
#
#         mapper = tracer.mappers_of_planes[0]
#         regularization = tracer.regularizations_of_planes[0]
#
#         scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
#             lensing_hyper_images[0][:], scaled_noise_map_1d[0], convolvers[0], mapper, regularization)
#
#         scaled_residuals_ = fitting_util.residuals_from_data_mask_and_model_data(lensing_hyper_images,
#                                                                                  [scaled_inversion.reconstructed_data_vector])
#         scaled_chi_squareds_ = fitting_util.chi_squareds_from_residuals_and_noise_map(scaled_residuals_, scaled_noise_map_1d)
#         scaled_chi_squared_terms = fitting_util.chi_squared_term_from_chi_squareds(scaled_chi_squareds_)
#         scaled_noise_terms = fitting_util.noise_term_from_mask_and_noise_map(scaled_noise_map_1d)
#         return sum(fitting_util.evidence_from_reconstruction_terms(scaled_chi_squared_terms,
#                                                                    [scaled_inversion.regularization_term],
#                                                                    [scaled_inversion.log_det_curvature_reg_matrix_term],
#                                                                    [scaled_inversion.log_det_regularization_matrix_term],
#                                                                    scaled_noise_terms))
#
#

#
#
# class HyperLensingProfileInversionFit(LensingProfileInversionFitter, HyperLensingInversion):
#
#     def __init__(self, lensing_hyper_images, tracer, padded_tracer=None):
#         """
#         Class to evaluate the fit_normal between a model described by a tracer_normal and an actual lensing_image.
#
#         Parameters
#         ----------
#         lensing_hyper_images : [li.LensingHyperImage]
#             List of the lensing hyper unblurred_image_1d that are to be fitted.
#         tracer : ray_tracing.AbstractTracer
#             The tracer, which describes the ray-tracing of the strong lensing configuration.
#         padded_tracer : ray_tracing.AbstractTracer
#             A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
#             padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
#         """
#
#         LensingDataFitter.__init__(self=self, lensing_images=lensing_hyper_images, tracer=tracer,
#                                     padded_tracer=padded_tracer)
#         fitter.AbstractLensingHyperFitter.__init__(self=self, fitting_hyper_images=lensing_hyper_images,
#                                          hyper_galaxies=tracer.hyper_galaxies)
#         super(HyperLensingProfileInversionFit, self).__init__(lensing_hyper_images, tracer, padded_tracer)
#
#         self.scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
#             self.profile_subtracted_images_[0], self.scaled_noise_map_1d[0],
#             lensing_hyper_images[0].convolver_mapping_matrix, self.inversion.mapper, self.inversion.regularization)
#
#         self.scaled_model_images_ = list(map(lambda _profile_model_image :
#                                              _profile_model_image + self.scaled_inversion.reconstructed_data_vector,
#                                              self.profile_model_images_))
#
#         self.scaled_residuals_ = fitting_util.residuals_from_data_mask_and_model_data(self.datas_, self.scaled_model_images_)
#         self.scaled_chi_squareds_ = fitting_util.chi_squareds_from_residuals_and_noise_map(self.scaled_residuals_,
#                                                                                            self.scaled_noise_map_1d)
#
#     @classmethod
#     def fast_scaled_evidence(cls, lensing_hyper_images, tracer):
#         """Perform the fit of this class as described above, but storing no results as class instances, thereby \
#         minimizing memory use and maximizing run-speed."""
#         contributions_1d = fitting_util.contributions_from_fitting_hyper_images_and_hyper_galaxies(
#             fitting_hyper_images=lensing_hyper_images, hyper_galaxies=tracer.hyper_galaxies)
#
#         scaled_noise_map_1d = fitting_util.scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(
#             fitting_hyper_images=lensing_hyper_images, contributions_1d=contributions_1d,
#             hyper_galaxies=tracer.hyper_galaxies)
#
#         convolvers_image = list(map(lambda lensing_image : lensing_image.convolver_image, lensing_hyper_images))
#         convolvers_mapping_matrix = list(map(lambda lensing_image : lensing_image.convolver_mapping_matrix,
#                                              lensing_hyper_images))
#
#         profile_model_images_ = fitting_util.blur_image_including_blurring_region(image_=tracer.image_plane_images_,
#                                                                                   blurring_image_=tracer.image_plane_blurring_images_, convolver=convolvers_image)
#
#         profile_subtracted_images_ = list(map(lambda lensing_image, profile_model_image_ :
#                                               lensing_image[:] - profile_model_image_,
#                                               lensing_hyper_images, profile_model_images_))
#
#         mapper = tracer.mappers_of_planes[0]
#         regularization = tracer.regularizations_of_planes[0]
#
#         scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
#             profile_subtracted_images_[0], scaled_noise_map_1d[0], convolvers_mapping_matrix[0], mapper,
#             regularization)
#
#         scaled_model_images_ = list(map(lambda profile_model_image_ :
#                                         profile_model_image_ + scaled_inversion.reconstructed_data_vector,
#                                         profile_model_images_))
#
#         scaled_residuals_ = fitting_util.residuals_from_data_mask_and_model_data(lensing_hyper_images, scaled_model_images_)
#         scaled_chi_squareds_ = fitting_util.chi_squareds_from_residuals_and_noise_map(scaled_residuals_, scaled_noise_map_1d)
#         scaled_chi_squared_terms = fitting_util.chi_squared_term_from_chi_squareds(scaled_chi_squareds_)
#         scaled_noise_terms = fitting_util.noise_term_from_mask_and_noise_map(scaled_noise_map_1d)
#         return sum(fitting_util.evidence_from_reconstruction_terms(scaled_chi_squared_terms,
#                                                                    [scaled_inversion.regularization_term],
#                                                                    [scaled_inversion.log_det_curvature_reg_matrix_term],
#                                                                    [scaled_inversion.log_det_regularization_matrix_term],
#                                                                    scaled_noise_terms))
#
#
# class PositionFit:
#
#     def __init__(self, positions, noise):
#
#         self.positions = positions
#         self.noise = noise
#
#     @property
#     def chi_squared_map(self):
#         return np.square(np.divide(self.maximum_separations, self.noise))
#
#     @property
#     def likelihood(self):
#         return -0.5 * sum(self.chi_squared_map)
#
#     def maximum_separation_within_threshold(self, threshold):
#         if max(self.maximum_separations) > threshold:
#             return False
#         else:
#             return True
#
#     @property
#     def maximum_separations(self):
#         return list(map(lambda positions: self.max_separation_of_grid(positions), self.positions))
#
#     def max_separation_of_grid(self, grid):
#         rdist_max = np.zeros((grid.shape[0]))
#         for i in range(grid.shape[0]):
#             xdists = np.square(np.subtract(grid[i, 0], grid[:, 0]))
#             ydists = np.square(np.subtract(grid[i, 1], grid[:, 1]))
#             rdist_max[i] = np.max(np.add(xdists, ydists))
#         return np.max(np.sqrt(rdist_max))
#
#
#


# TODO : The [plane_index][galaxy_index] datas structure is going to be key to tracking galaxies / hyper galaxies in
# TODO : Multi-plane ray tracing. I never felt it was easy to follow using list comprehensions from ray_tracing.
# TODO : Can we make this neater?