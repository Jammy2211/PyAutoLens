import numpy as np

from autolens import exc
from autolens.inversion import inversions
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing


class AbstractFit(object):

    def __init__(self, fitting_datas, _model_datas):

        self.masks = list(map(lambda fit_data : fit_data.mask, fitting_datas))
        self._datas = list(map(lambda fit_data : fit_data[:], fitting_datas))
        self._noise_maps = list(map(lambda fit_data : fit_data.noise_map, fitting_datas))
        self.map_to_scaled_arrays = list(map(lambda fit_data: fit_data.grids.image.scaled_array_from_array_1d,
                                             fitting_datas))

        self._model_datas = _model_datas
        self._residuals = residuals_from_datas_and_model_datas(datas=self._datas, model_datas=self._model_datas)
        self._chi_squareds = chi_squareds_from_residuals_and_noise_maps(self._residuals, self._noise_maps)

    @property
    def chi_squared_terms(self):
        return chi_squared_terms_from_chi_squareds(self._chi_squareds)

    @property
    def chi_squared_term(self):
        return sum(self.chi_squared_terms)

    @property
    def reduced_chi_squareds(self):
        return list(map(lambda chi_squared_term, mask : chi_squared_term / mask.pixels_in_mask,
                        self.chi_squared_terms, self.masks))

    @property
    def reduced_chi_squared(self):
        return sum(self.reduced_chi_squareds)

    @property
    def noise_terms(self):
        return noise_terms_from_noise_maps(self._noise_maps)

    @property
    def noise_term(self):
        return sum(self.noise_terms)

    @property
    def likelihoods(self):
        return likelihoods_from_chi_squareds_and_noise_terms(self.chi_squared_terms, self.noise_terms)

    @property
    def likelihood(self):
        return sum(self.likelihoods)

    @property
    def noise_maps(self):
        return map_arrays_to_scaled_arrays(_arrays=self._noise_maps, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def model_datas(self):
        return map_arrays_to_scaled_arrays(_arrays=self._model_datas, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def residuals(self):
        return map_arrays_to_scaled_arrays(_arrays=self._residuals, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def chi_squareds(self):
        return map_arrays_to_scaled_arrays(_arrays=self._chi_squareds, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def noise_map(self):
        return self.noise_maps[0]

    @property
    def model_data(self):
        return self.model_datas[0]

    @property
    def residual(self):
        return self.residuals[0]

    @property
    def chi_squared(self):
        return self.chi_squareds[0]


class AbstractDataFit(AbstractFit):

    def __init__(self, fitting_datas, _model_datas):

        self.datas = fitting_datas.array
        super(AbstractDataFit, self).__init__(fitting_datas=fitting_datas, _model_datas=_model_datas)


class AbstractImageFit(AbstractFit):

    def __init__(self, fitting_images, _model_images):

        self.images = list(map(lambda fit_image : fit_image.image, fitting_images))
        super(AbstractImageFit, self).__init__(fitting_datas=fitting_images, _model_datas=_model_images)

    @property
    def model_images(self):
        return self.model_datas

    @property
    def model_image(self):
        return self.model_datas[0]


class AbstractHyperFit(object):

    def __init__(self, fitting_hyper_image, hyper_galaxies):

        self.is_hyper_fit = True
        self._contributions = contributions_from_hyper_images_and_galaxies(fitting_hyper_image.hyper_model_image,
                              fitting_hyper_image.hyper_galaxy_images, hyper_galaxies,
                              fitting_hyper_image.hyper_minimum_values)

        self._scaled_noise_maps = scaled_noise_from_hyper_galaxies_and_contributions(self._contributions,
                                                                                     hyper_galaxies,
                                                                                     fitting_hyper_image.noise_maps)

    @property
    def scaled_chi_squared_terms(self):
        return chi_squared_terms_from_chi_squareds(self._scaled_chi_squareds)

    @property
    def scaled_noise_terms(self):
        return noise_terms_from_noise_maps(self._scaled_noise_maps)

    @property
    def scaled_noise_maps(self):
        return map_arrays_to_scaled_arrays(_arrays=self._scaled_noise_maps,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def scaled_chi_squareds(self):
        return map_arrays_to_scaled_arrays(_arrays=self._scaled_chi_squareds,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def contributions(self):
        return map_arrays_to_scaled_arrays(_arrays=self._contributions, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def scaled_noise_map(self):
        return self.scaled_noise_maps[0]

    @property
    def scaled_chi_squared(self):
        return self.scaled_chi_squareds[0]

class AbstractHyperImageFit(AbstractImageFit, AbstractHyperFit):

    def __init__(self, fitting_hyper_images, _model_images, hyper_galaxies):

       AbstractHyperFit.__init__(self=self, fitting_hyper_image=fitting_hyper_images, hyper_galaxies=hyper_galaxies)
       super(AbstractHyperImageFit, self).__init__(fitting_images=fitting_hyper_images, _model_images=_model_images)
       self._scaled_chi_squareds = chi_squareds_from_residuals_and_noise_maps(self._residuals, self._scaled_noise_maps)


class AbstractProfileFit(AbstractImageFit):

    def __init__(self, fitting_images, _images, _blurring_images):

        self.convolvers_image = list(map(lambda fit_image : fit_image.convolver_image, fitting_images))

        _model_images = blur_images_including_blurring_regions(images=_images, blurring_images=_blurring_images,
                                                               convolvers=self.convolvers_image)

        super(AbstractProfileFit, self).__init__(fitting_images=fitting_images, _model_images=_model_images)


class AbstractInversionFit(AbstractImageFit):

    def __init__(self, fitting_images, mapper, regularization):

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(image=fitting_images[0][:],
        noise_map=fitting_images[0].noise_map, convolver=fitting_images[0].convolver_mapping_matrix,
        mapper=mapper, regularization=regularization)

        super(AbstractInversionFit, self).__init__(fitting_images=fitting_images,
                                                   _model_images=self.inversion.reconstructed_data_vector)

    @property
    def likelihoods_with_regularization(self):
        return likelihoods_with_regularization_from_chi_squared_regularization_and_noise_terms(self.chi_squared_terms,
               [self.inversion.regularization_term], self.noise_terms)

    @property
    def likelihood_with_regularization(self):
        return sum(self.likelihoods_with_regularization)

    @property
    def evidences(self):
        return evidences_from_reconstruction_terms(self.chi_squared_terms, [self.inversion.regularization_term],
                                                   [self.inversion.log_det_curvature_reg_matrix_term],
                                                   [self.inversion.log_det_regularization_matrix_term],
                                                   self.noise_terms)

    @property
    def evidence(self):
        return sum(self.evidences)


class AbstractProfileInversionFit(AbstractImageFit):
    
    def __init__(self, fitting_images, _images, _blurring_images, mapper, regularization):
        
        self.convolvers_image = list(map(lambda fit_image : fit_image.convolver_image, fitting_images))

        self._profile_model_images = blur_images_including_blurring_regions(images=_images, blurring_images=_blurring_images,
                                                                            convolvers=self.convolvers_image)



        self._profile_subtracted_images = list(map(lambda fitting_image, _profile_model_image :
                                                   fitting_image[:] - _profile_model_image,
                                                   fitting_images, self._profile_model_images))

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image=self._profile_subtracted_images[0], noise_map=fitting_images[0].noise_map,
            convolver=fitting_images[0].convolver_mapping_matrix, mapper=mapper, regularization=regularization)
        
        self._inversion_model_images = [self.inversion.reconstructed_data_vector]

        _model_images = list(map(lambda _profile_model_image, _inversion_model_image :
                                 _profile_model_image + _inversion_model_image,
                                 self._profile_model_images, self._inversion_model_images))

        super(AbstractProfileInversionFit, self).__init__(fitting_images=fitting_images, _model_images=_model_images)

    @property
    def profile_subtracted_images(self):
        return map_arrays_to_scaled_arrays(_arrays=self._profile_subtracted_images,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def profile_model_images(self):
        return map_arrays_to_scaled_arrays(_arrays=self._profile_model_images,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def inversion_model_images(self):
        return map_arrays_to_scaled_arrays(_arrays=self._inversion_model_images,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def profile_subtracted_image(self):
        return self.profile_subtracted_images[0]

    @property
    def profile_model_image(self):
        return self.profile_model_images[0]

    @property
    def inversion_model_image(self):
        return self.inversion_model_images[0]

    @property
    def evidences(self):
        return evidences_from_reconstruction_terms(self.chi_squared_terms, [self.inversion.regularization_term],
                                                   [self.inversion.log_det_curvature_reg_matrix_term],
                                                   [self.inversion.log_det_regularization_matrix_term],
                                                   self.noise_terms)

    @property
    def evidence(self):
        return sum(self.evidences)


def map_arrays_to_scaled_arrays(_arrays, map_to_scaled_arrays):
    return list(map(lambda _array, map_to_scaled_array, : map_to_scaled_array(_array), _arrays, map_to_scaled_arrays))

def blur_images_including_blurring_regions(images, blurring_images, convolvers):
    """For a given lensing_image and blurring region, convert them to 2D and blur with the PSF, then return as
    the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The lensing_image data_vector using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region data_vector, using the GridData 1D representation.
    convolver : auto_lens.inversion.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return list(map(lambda _image, _blurring_image, convolver :
                    convolver.convolve_image(image_array=_image, blurring_array=_blurring_image),
                    images, blurring_images, convolvers))

def residuals_from_datas_and_model_datas(datas, model_datas):
    """Compute the residuals between an observed charge injection lensing_image and post-cti model_datas lensing_image.

    Residuals = (Data - Model).

    Parameters
    -----------
    datas : [np.ndarray]
        The observed charge injection lensing_image datas.
    model_datas : [np.ndarray]
        The model_datas lensing_image.
    """
    return list(map(lambda data, model_data : np.subtract(data, model_data), datas, model_datas))

def chi_squareds_from_residuals_and_noise_maps(residuals, noise_maps):
    """Computes a chi-squared lensing_image, by calculating the squared residuals between an observed charge injection \
    images and post-cti hyper_model_image lensing_image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    residuals : [np.ndarray]
        The residuals of the model fit to the data
    noise_maps : [np.ndarray]
        The noises in the lensing_image.
    """
    return list(map(lambda residual, noise_map : np.square((np.divide(residual, noise_map))), residuals, noise_maps))

def chi_squared_terms_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model lensing_image's fit to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image, dividing by the noise_map in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    chi_squareds
    """
    return list(map(lambda chi_squared : np.sum(chi_squared), chi_squareds))

def noise_terms_from_noise_maps(noise_maps):
    """Compute the noise_map normalization term of an lensing_image, which is computed by summing the noise_map in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_maps : [np.ndarray]
        The noise_map in each pixel.
    """
    return list(map(lambda noise_map : np.sum(np.log(2 * np.pi * noise_map ** 2.0)), noise_maps))

def likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms):
    """Compute the likelihood of a model lensing_image's fit to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - data_vector) of every pixel divided by the noise_map in each pixel, all
    squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise_map is also included, by summing the log noise_map value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    """
    return list(map(lambda chi_squared_term, noise_term : -0.5 * (chi_squared_term + noise_term),
                    chi_squared_terms, noise_terms))

def likelihoods_with_regularization_from_chi_squared_regularization_and_noise_terms(chi_squared_terms,
                                                                                    regularization_terms, noise_terms):
    return list(map(lambda chi_squared_term, regularization_term, noise_term :
                    -0.5 * (chi_squared_term + regularization_term + noise_term),
                    chi_squared_terms, regularization_terms, noise_terms))

def evidences_from_reconstruction_terms(chi_squared_terms, regularization_terms, log_covariance_regularization_terms,
                                        log_regularization_terms, noise_terms):
    return list(map(lambda chi_squared_term, regularization_term, log_covariance_regularization_term,
                           log_regularization_term, noise_term :
                    -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -
                            log_regularization_term + noise_term),
                    chi_squared_terms, regularization_terms, log_covariance_regularization_terms,
                    log_regularization_terms, noise_terms))

def unmasked_model_images_from_fitting_images(fitting_images, _unmasked_images):

    return list(map(lambda fitting_image, _unmasked_image :
                    unmasked_model_image_from_fitting_image(fitting_image, _unmasked_image),
                    fitting_images, _unmasked_images))

def unmasked_model_image_from_fitting_image(fitting_image, _unmasked_image):

    _model_image = fitting_image.padded_grids.image.convolve_array_1d_with_psf(_unmasked_image,
                                                                               fitting_image.psf)

    return fitting_image.padded_grids.image.scaled_array_from_array_1d(_model_image)

def contributions_from_fitting_hyper_images_and_hyper_galaxies(fitting_hyper_images, hyper_galaxies):

    return list(map(lambda hyp :
                    contributions_from_hyper_images_and_galaxies(hyper_model_image=hyp.hyper_model_image,
                                                                 hyper_galaxy_images=hyp.hyper_galaxy_images,
                                                                 hyper_galaxies=hyper_galaxies,
                                                                 minimum_values=hyp.hyper_minimum_values),
                fitting_hyper_images))

def contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies, minimum_values):
    """Use the model lensing_image and galaxy lensing_image (computed in the previous phase of the pipeline) to determine the
    contributions of each hyper galaxy.

    Parameters
    -----------
    minimum_values
    hyper_model_image : ndarray
        The best-fit model lensing_image to the data_vector, from a previous phase of the pipeline
    hyper_galaxy_images : [ndarray]
        The best-fit model lensing_image of each hyper-galaxy, which can tell us how much flux each pixel contributes to.
    hyper_galaxies : [galaxy.HyperGalaxy]
        Each hyper-galaxy which is used to determine its contributions.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_hyper_images(hyper_model_image, galaxy_image, minimum_value),
                    hyper_galaxies, hyper_galaxy_images, minimum_values))

def scaled_noises_from_fitting_hyper_images_hyper_galaxies_and_contributions(fitting_hyper_images, contributions,
                                                                             hyper_galaxies):
    return list(map(lambda hyp, contribution :
                    scaled_noise_from_hyper_galaxies_and_contributions(contributions=contribution,
                                                                       hyper_galaxies=hyper_galaxies,
                                                                       noise_map=hyp.noise_map),
                    fitting_hyper_images, contributions))

def scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise_map):
    """Use the contributions of each hyper galaxy to compute the scaled noise_map.
    Parameters
    -----------
    noise_map
    hyper_galaxies
    contributions : [ndarray]
        The contribution of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
    """
    scaled_noises = list(map(lambda hyper, contribution: hyper.scaled_noise_from_contributions(noise_map, contribution),
                             hyper_galaxies, contributions))
    return noise_map + sum(scaled_noises)