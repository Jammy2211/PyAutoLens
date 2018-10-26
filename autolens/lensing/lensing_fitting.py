import numpy as np

from autolens import exc
from autolens.inversion import inversions
from autolens.fitting import fitting
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing

minimum_value_profile = 0.1

def fit_multiple_lensing_images_with_tracer(lensing_images, tracer, padded_tracer=None):
    """Fit.

    Parameters
    -----------
    lensing_image : li.LensingImage or li.LensingHyperImage
        Lensing _datas
    tracer : ray_tracing.AbstractTracer
        tracer
    """

    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFit(lensing_images=lensing_images, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileFit(lensing_hyper_images=lensing_images, tracer=tracer, padded_tracer=padded_tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFit(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit(lensing_hyper_images=lensing_images, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFit(lensing_images=lensing_images, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit(lensing_hyper_images=lensing_images, tracer=tracer,
                                                   padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


def fast_likelihood_from_multiple_lensing_images_and_tracer(lensing_images, tracer):
    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFit.fast_likelihood(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileFit.fast_scaled_likelihood(lensing_hyper_images=lensing_images, tracer=tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFit.fast_evidence(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit.fast_scaled_evidence(lensing_images=lensing_images, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFit.fast_evidence(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit.fast_scaled_evidence(lensing_images=lensing_images, tracer=tracer)

    else:

        raise exc.FittingException('The fast likelihood routine did not call a likelihood functions - check the '
                                   'properties of the tracer')


def fit_lensing_image_with_tracer(lensing_image, tracer, padded_tracer=None):
    """Fit.

    Parameters
    -----------
    lensing_image : li.LensingImage or li.LensingHyperImage
        Lensing _datas
    tracer : ray_tracing.AbstractTracer
        tracer
    """

    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFit(lensing_images=[lensing_image], tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileFit(lensing_hyper_images=[lensing_image], tracer=tracer, padded_tracer=padded_tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFit(lensing_images=[lensing_image], tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit(lensing_hyper_images=[lensing_image], tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFit(lensing_images=[lensing_image], tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit(lensing_hyper_images=[lensing_image], tracer=tracer,
                                                   padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


def fast_likelihood_from_lensing_image_and_tracer(lensing_image, tracer):
    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFit.fast_likelihood(lensing_images=[lensing_image], tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileFit.fast_scaled_likelihood(lensing_hyper_images=[lensing_image], tracer=tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFit.fast_evidence(lensing_images=[lensing_image], tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit.fast_scaled_evidence(lensing_images=[lensing_image], tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFit.fast_evidence(lensing_images=[lensing_image], tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit.fast_scaled_evidence(lensing_images=[lensing_image], tracer=tracer)

    else:

        raise exc.FittingException('The fast likelihood routine did not call a likelihood functions - check the '
                                   'properties of the tracer')


class AbstractLensingFit(object):

    def __init__(self, tracer, padded_tracer=None):

        self.tracer = tracer
        self.padded_tracer = padded_tracer

    @property
    def total_inversions(self):
        return len(self.tracer.mappers_of_planes)


class LensingProfileFit(fitting.AbstractProfileFit, AbstractLensingFit):

    def __init__(self, lensing_images, tracer, padded_tracer=None):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        AbstractLensingFit.__init__(self=self, tracer=tracer, padded_tracer=padded_tracer)
        super(LensingProfileFit, self).__init__(fitting_images=lensing_images, _images=tracer._image_plane_images,
                                                _blurring_images=tracer._image_plane_blurring_images)

    @classmethod
    def fast_likelihood(cls, lensing_images, tracer):
        """
        Fast calculation of likelihood

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        convolvers = list(map(lambda lensing_image : lensing_image.convolver_image, lensing_images))
        _noise_maps = list(map(lambda lensing_image : lensing_image.noise_map, lensing_images))
        _model_images = fitting.blur_images_including_blurring_regions(images=tracer._image_plane_images,
                        blurring_images=tracer._image_plane_blurring_images, convolvers=convolvers)
        _residuals = fitting.residuals_from_datas_and_model_datas(lensing_images, _model_images),
        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, _noise_maps),
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps(_noise_maps)
        return sum(fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms))

    @property
    def model_images_of_planes(self):

        _image_plane_images_of_planes = self.tracer._image_plane_images_of_planes
        _image_plane_blurring_images_of_planes = self.tracer._image_plane_blurring_images_of_planes

        model_images_of_planes = [[] for _ in range(self.tracer.total_images)]

        for image_index in range(self.tracer.total_images):
            convolver = self.convolvers_image[image_index]
            map_to_scaled_array = self.map_to_scaled_arrays[image_index]
            for plane_index in range(self.tracer.total_planes):
                _image = _image_plane_images_of_planes[image_index][plane_index]
                if np.count_nonzero(_image): # If all entries are zero, there was no light profile
                    _blurring_image = _image_plane_blurring_images_of_planes[image_index][plane_index]
                    _blurred_image = (convolver.convolve_image(image_array=_image, blurring_array=_blurring_image))
                    _blurred_image = map_to_scaled_array(_blurred_image)
                    model_images_of_planes[image_index].append(_blurred_image)
                else:
                    model_images_of_planes[image_index].append(None)

        return model_images_of_planes

    @property
    def unmasked_model_profile_images(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return fitting.unmasked_model_images_from_fitting_images(fitting_images=self._datas,
                                        _unmasked_images=self.padded_tracer._image_plane_images)

    @property
    def unmasked_model_profile_images_of_galaxies(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return unmasked_model_images_of_galaxies_from_lensing_images_and_tracer(lensing_images=self._datas,
                                                                                    tracer=self.padded_tracer)

    @property
    def unmasked_model_profile_image(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return self.unmasked_model_profile_images[0]

class HyperLensingProfileFit(LensingProfileFit, fitting.AbstractHyperFit):

    def __init__(self, lensing_hyper_images, tracer, padded_tracer=None):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_hyper_images: [li.LensingHyperImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """

        fitting.AbstractHyperFit.__init__(self=self, fitting_hyper_image=lensing_hyper_images,
                                          hyper_galaxies=tracer.hyper_galaxies)
        super(HyperLensingProfileFit, self).__init__(lensing_hyper_images, tracer, padded_tracer)

        self._scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(self._residuals,
                                                                                       self._scaled_noise_maps)

    @classmethod
    def fast_scaled_likelihood(cls, lensing_hyper_images, tracer):
        """
        Fast calculation of scaled likelihood

        Parameters
        ----------
        lensing_images: [li.LensingHyperImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        _contributions = fitting.contributions_from_hyper_images_and_galaxies(lensing_hyper_images.hyper_model_image,
                                                                              lensing_hyper_images.hyper_galaxy_images, tracer.hyper_galaxies,
                                                                              lensing_hyper_images.hyper_minimum_values)
        _scaled_noise_map = fitting.scaled_noise_from_hyper_galaxies_and_contributions(_contributions, tracer.hyper_galaxies,
                                                                                       lensing_hyper_images.noise_maps)

        convolve_image = lensing_hyper_images.convolver_image.convolve_image
        _model_image = convolve_image(tracer._image_plane_images, tracer._image_plane_blurring_images)
        _residuals = fitting.residuals_from_datas_and_model_datas(lensing_hyper_images[:], _model_image)
        _scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, _scaled_noise_map)

        scaled_chi_squared_term = fitting.chi_squared_terms_from_chi_squareds(_scaled_chi_squareds)
        scaled_noise_term = fitting.noise_terms_from_noise_maps(_scaled_noise_map)
        return fitting.likelihoods_from_chi_squareds_and_noise_terms(scaled_chi_squared_term, scaled_noise_term)

    @property
    def scaled_likelihood(self):
        return fitting.likelihoods_from_chi_squareds_and_noise_terms(self.scaled_chi_squared_terms, self.scaled_noise_terms)


class LensingInversionFit(fitting.AbstractInversionFit, AbstractLensingFit):

    def __init__(self, lensing_images, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """

        AbstractLensingFit.__init__(self=self, tracer=tracer)

        mapper = tracer.mappers_of_planes[0]
        regularization = tracer.regularizations_of_planes[0]

        super(LensingInversionFit, self).__init__(fitting_images=lensing_images, mapper=mapper,
                                                  regularization=regularization)

    @classmethod
    def fast_evidence(cls, lensing_images, tracer):
        """
        Fast calculation of evidence.

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        mapper = tracer.mappers_of_planes[0]
        regularization = tracer.regularizations_of_planes[0]
        inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(lensing_images[:], lensing_images.noise_maps,
                                                                                      lensing_images.convolver_mapping_matrix,
                                                                                      mapper, regularization)
        _model_image = inversion.reconstructed_data_vector
        _residuals = fitting.residuals_from_datas_and_model_datas(lensing_images[:], _model_image)
        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, lensing_images.noise_maps)
        chi_squared_term = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds)
        noise_term = fitting.noise_terms_from_noise_maps(lensing_images.noise_maps)
        return fitting.evidences_from_reconstruction_terms(chi_squared_term, inversion.regularization_term,
                                                           inversion.log_det_curvature_reg_matrix_term,
                                                           inversion.log_det_regularization_matrix_term, noise_term)

    @property
    def model_images_of_planes(self):
        return [None, self.map_to_scaled_arrays(self._model_datas)]


class HyperLensingInversion(fitting.AbstractHyperFit):

    @property
    def scaled_model_image(self):
        return list(map(lambda map_to_scaled_array, _scaled_model_image : map_to_scaled_array(_scaled_model_image),
                        self.map_to_scaled_array, self._scaled_model_images))


    @property
    def scaled_residuals(self):
        return list(map(lambda map_to_scaled_array, _scaled_residuals : map_to_scaled_array(_scaled_residuals),
                        self.map_to_scaled_array, self._scaled_residuals))


    @property
    def scaled_evidence(self):
        return fitting.evidences_from_reconstruction_terms(self.scaled_chi_squared_terms,
                                                           self.scaled_inversion.regularization_term,
                                                           self.scaled_inversion.log_det_curvature_reg_matrix_term,
                                                           self.scaled_inversion.log_det_regularization_matrix_term,
                                                           self.scaled_noise_terms)


class HyperLensingInversionFit(LensingInversionFit, HyperLensingInversion):

    def __init__(self, lensing_hyper_images, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_hyper_images: [li.LensingHyperImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """

        fitting.AbstractHyperFit.__init__(self=self, fitting_hyper_image=lensing_hyper_images,
                                          hyper_galaxies=tracer.hyper_galaxies)
        super(HyperLensingInversionFit, self).__init__(lensing_hyper_images, tracer)

        self.scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            lensing_hyper_images[:], self._scaled_noise_maps, lensing_hyper_images.convolver_mapping_matrix,
            self.inversion.mapper, self.inversion.regularization)

        self._scaled_model_images = self.scaled_inversion.reconstructed_data_vector
        self._scaled_residuals = fitting.residuals_from_datas_and_model_datas(self._datas, self._scaled_model_images)
        self._scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(self._scaled_residuals,
                                                                                       self._scaled_noise_maps)

    @classmethod
    def fast_scaled_evidence(cls, lensing_hyper_images, tracer):
        """
        Fast calculation of scaled evidence.

        Parameters
        ----------
        lensing_images: [li.LensingHyperImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        _contributions = fitting.contributions_from_hyper_images_and_galaxies(lensing_hyper_images.hyper_model_image,
                                                                              lensing_hyper_images.hyper_galaxy_images,
                                                                              tracer.hyper_galaxies,
                                                                              lensing_hyper_images.hyper_minimum_values)
        _scaled_noise_map = fitting.scaled_noise_from_hyper_galaxies_and_contributions(_contributions, tracer.hyper_galaxies,
                                                                                       lensing_hyper_images.noise_maps)

        mapper = tracer.mappers_of_planes[0]
        regularization = tracer.regularizations_of_planes[0]

        scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            lensing_hyper_images[:], _scaled_noise_map, lensing_hyper_images.convolver_mapping_matrix, mapper, regularization)

        _scaled_residuals = fitting.residuals_from_datas_and_model_datas(lensing_hyper_images[:], scaled_inversion.reconstructed_data_vector)
        _scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_scaled_residuals, _scaled_noise_map)
        scaled_chi_squared_term = fitting.chi_squared_terms_from_chi_squareds(_scaled_chi_squareds)
        scaled_noise_term = fitting.noise_terms_from_noise_maps(_scaled_noise_map)
        return fitting.evidences_from_reconstruction_terms(scaled_chi_squared_term, scaled_inversion.regularization_term,
                                                           scaled_inversion.log_det_curvature_reg_matrix_term,
                                                           scaled_inversion.log_det_regularization_matrix_term,
                                                           scaled_noise_term)


class LensingProfileInversionFit(fitting.AbstractProfileInversionFit, AbstractLensingFit):

    def __init__(self, lensing_images, tracer, padded_tracer=None):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """

        mapper = tracer.mappers_of_planes[0]
        regularization = tracer.regularizations_of_planes[0]

        AbstractLensingFit.__init__(self=self, tracer=tracer, padded_tracer=padded_tracer)
        super(LensingProfileInversionFit, self).__init__(fitting_images=lensing_images, _images=tracer._image_plane_images,
                                                         _blurring_images=tracer._image_plane_blurring_images,
                                                         mapper=mapper, regularization=regularization)

    @classmethod
    def fast_evidence(cls, lensing_images, tracer):
        """
        Fast calculation of evidence

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        convolve_image = lensing_images.convolver_image.convolve_image
        _profile_model_image = convolve_image(tracer._image_plane_images, tracer._image_plane_blurring_images)
        _profile_subtracted_image = lensing_images[:] - _profile_model_image
        mapper = tracer.mappers_of_planes[0]
        regularization = tracer.regularizations_of_planes[0]
        inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(_profile_subtracted_image,
                                                                                      lensing_images.noise_maps,
                                                                                      lensing_images.convolver_mapping_matrix,
                                                                                      mapper, regularization)
        _model_image = _profile_model_image + inversion.reconstructed_data_vector
        _residuals = fitting.residuals_from_datas_and_model_datas(lensing_images[:], _model_image)
        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, lensing_images.noise_maps)
        chi_squared_term = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds)
        noise_term = fitting.noise_terms_from_noise_maps(lensing_images.noise_maps)
        return fitting.evidences_from_reconstruction_terms(chi_squared_term, inversion.regularization_term,
                                                           inversion.log_det_curvature_reg_matrix_term,
                                                           inversion.log_det_regularization_matrix_term, noise_term)

    @property
    def model_images_of_planes(self):
        return [self.profile_model_images, self.inversion_model_images]


class HyperLensingProfileInversionFit(LensingProfileInversionFit, HyperLensingInversion):

    def __init__(self, lensing_hyper_images, tracer, padded_tracer=None):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_hyper_images: [li.LensingHyperImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """

        fitting.AbstractHyperFit.__init__(self=self, fitting_hyper_image=lensing_hyper_images,
                                          hyper_galaxies=tracer.hyper_galaxies)
        super(HyperLensingProfileInversionFit, self).__init__(lensing_hyper_images, tracer, padded_tracer)

        self.scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            self._profile_subtracted_images, self._scaled_noise_maps, lensing_hyper_images.convolver_mapping_matrix,
            self.inversion.mapper, self.inversion.regularization)

        self._scaled_model_image = self._profile_model_images + self.scaled_inversion.reconstructed_data_vector
        self._scaled_residuals = fitting.residuals_from_datas_and_model_datas(self._datas, self._scaled_model_image)
        self._scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(self._scaled_residuals,
                                                                                       self._scaled_noise_maps)

    @classmethod
    def fast_scaled_evidence(cls, lensing_hyper_images, tracer):
        """
        Fast calculation of likelihood

        Parameters
        ----------
        lensing_images: [li.LensingHyperImage]
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.AbstractTracer
            An object describing the model
        """
        _contributions = fitting.contributions_from_hyper_images_and_galaxies(lensing_hyper_images.hyper_model_image,
                                                                              lensing_hyper_images.hyper_galaxy_images,
                                                                              tracer.hyper_galaxies,
                                                                              lensing_hyper_images.hyper_minimum_values)
        _scaled_noise_map = fitting.scaled_noise_from_hyper_galaxies_and_contributions(_contributions, tracer.hyper_galaxies,
                                                                                       lensing_hyper_images.noise_maps)

        convolve_image = lensing_hyper_images.convolver_image.convolve_image
        _profile_model_image = convolve_image(tracer._image_plane_images, tracer._image_plane_blurring_images)
        _profile_subtracted_image = lensing_hyper_images[:] - _profile_model_image

        mapper = tracer.mappers_of_planes[0]
        regularization = tracer.regularizations_of_planes[0]

        scaled_inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            _profile_subtracted_image, _scaled_noise_map, lensing_hyper_images.convolver_mapping_matrix, mapper,
            regularization)

        _scaled_model_image = _profile_model_image + scaled_inversion.reconstructed_data_vector
        _scaled_residuals = fitting.residuals_from_datas_and_model_datas(lensing_hyper_images[:], _scaled_model_image)
        _scaled_chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_scaled_residuals, _scaled_noise_map)
        scaled_chi_squared_term = fitting.chi_squared_terms_from_chi_squareds(_scaled_chi_squareds)
        scaled_noise_term = fitting.noise_terms_from_noise_maps(_scaled_noise_map)
        return fitting.evidences_from_reconstruction_terms(scaled_chi_squared_term, scaled_inversion.regularization_term,
                                                           scaled_inversion.log_det_curvature_reg_matrix_term,
                                                           scaled_inversion.log_det_regularization_matrix_term,
                                                           scaled_noise_term)


class PositionFit:

    def __init__(self, positions, noise):

        self.positions = positions
        self.noise = noise

    @property
    def chi_squareds(self):
        return np.square(np.divide(self.maximum_separations, self.noise))

    @property
    def likelihood(self):
        return -0.5 * sum(self.chi_squareds)

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


def unmasked_model_images_of_galaxies_from_lensing_images_and_tracer(lensing_images, tracer):
    return list(map(lambda lensing_image, image_index :
                    unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image, tracer, image_index),
                    lensing_images, list(range(tracer.total_images))))

def unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(lensing_image, tracer, image_index):


    padded_model_images_of_galaxies = [[] for _ in range(len(tracer.all_planes))]

    for plane_index, plane in enumerate(tracer.all_planes):
        for galaxy_index in range(len(plane.galaxies)):

            _galaxy_image_plane_image = plane._image_plane_images_of_galaxies[image_index][galaxy_index]

            galaxy_model_image = fitting.unmasked_model_image_from_fitting_image(fitting_image=lensing_image,
                                                                                  _unmasked_image=_galaxy_image_plane_image)

            padded_model_images_of_galaxies[plane_index].append(galaxy_model_image)

    return padded_model_images_of_galaxies