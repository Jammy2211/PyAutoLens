import numpy as np

from autolens import exc
from autolens.inversion import inversions
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing


class AbstractFit(object):

    def __init__(self, fitting_data, _model_data):

        self.mask = fitting_data.mask
        self._data = fitting_data[:]
        self._noise_map = fitting_data.noise_map
        self._model_data = _model_data
        self._residuals = residuals_from_data_and_model(self._data, self._model_data)
        self._chi_squareds = chi_squareds_from_residuals_and_noise(self._residuals, self._noise_map)
        self.map_to_scaled_array = fitting_data.grids.image.scaled_array_from_array_1d

    @property
    def chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self._chi_squareds)

    @property
    def reduced_chi_squared(self):
        return self.chi_squared_term / self.mask.pixels_in_mask

    @property
    def noise_term(self):
        return noise_term_from_noise_map(self._noise_map)

    @property
    def likelihood(self):
        return likelihood_from_chi_squared_and_noise_terms(self.chi_squared_term, self.noise_term)

    @property
    def noise_map(self):
        return self.map_to_scaled_array(self._noise_map)

    @property
    def residuals(self):
        return self.map_to_scaled_array(self._residuals)

    @property
    def chi_squareds(self):
        return self.map_to_scaled_array(self._chi_squareds)


class AbstractDataFit(AbstractFit):

    def __init__(self, fitting_data, _model_data):

        self.data = fitting_data.data
        super(AbstractDataFit, self).__init__(fitting_data=fitting_data, _model_data=_model_data)

    @property
    def model_data(self):
        return self.map_to_scaled_array(self._model_data)


class AbstractImageFit(AbstractFit):

    def __init__(self, fitting_image, _model_image):

        self.image = fitting_image.image
        super(AbstractImageFit, self).__init__(fitting_data=fitting_image, _model_data=_model_image)

    @property
    def model_image(self):
        return self.map_to_scaled_array(self._model_data)


class AbstractImageProfileFit(AbstractImageFit):

    def __init__(self, fitting_image, _image, _blurring_image):

        self.convolver_image = fitting_image.convolver_image
        _model_image = self.convolver_image.convolve_image(image_array=_image, blurring_array=_blurring_image)

        super(AbstractImageProfileFit, self).__init__(fitting_image=fitting_image, _model_image=_model_image)


class AbstractImageInversionFit(AbstractImageFit):

    def __init__(self, fitting_image, mapper, regularization):

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(image=fitting_image[:],
                        noise_map=fitting_image.noise_map, convolver=fitting_image.convolver_mapping_matrix,
                        mapper=mapper, regularization=regularization)

        super(AbstractImageInversionFit, self).__init__(fitting_image=fitting_image,
                                                        _model_image=self.inversion.reconstructed_data_vector)

    @property
    def likelihood_with_regularization(self):
        return likelihood_with_regularization_from_chi_squared_regularization_and_noise_terms(self.chi_squared_term,
                                                 self.inversion.regularization_term, self.noise_term)

    @property
    def evidence(self):
        return evidence_from_reconstruction_terms(self.chi_squared_term, self.inversion.regularization_term,
                                                  self.inversion.log_det_curvature_reg_matrix_term,
                                                  self.inversion.log_det_regularization_matrix_term,
                                                  self.noise_term)


class AbstractImageProfileInversionFit(AbstractImageFit):
    
    def __init__(self, fitting_image, _image, _blurring_image, mapper, regularization):
        
        self.convolver_image = fitting_image.convolver_image

        self._profile_model_image = fitting_image.convolver_image.convolve_image(image_array=_image,
                                                                                 blurring_array=_blurring_image)
        
        self._profile_subtracted_image = fitting_image[:] - self._profile_model_image

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image=self._profile_subtracted_image, noise_map=fitting_image.noise_map, 
            convolver=fitting_image.convolver_mapping_matrix, mapper=mapper, regularization=regularization)
        
        self._inversion_model_image = self.inversion.reconstructed_data_vector

        super(AbstractImageProfileInversionFit, self).__init__(fitting_image=fitting_image,
                                                               _model_image=self._profile_model_image + self._inversion_model_image)

    
    @property
    def profile_subtracted_image(self):
        return self.map_to_scaled_array(self._profile_subtracted_image)

    @property
    def profile_model_image(self):
        return self.map_to_scaled_array(self._profile_model_image)

    @property
    def inversion_model_image(self):
        return self.map_to_scaled_array(self._inversion_model_image)

    @property
    def evidence(self):
        return evidence_from_reconstruction_terms(self.chi_squared_term, self.inversion.regularization_term,
                                                  self.inversion.log_det_curvature_reg_matrix_term,
                                                  self.inversion.log_det_regularization_matrix_term,
                                                  self.noise_term)
    

class AbstractHyperFit(object):

    def __init__(self, fitting_hyper_image, hyper_galaxies):

        self.is_hyper_fit = True
        self._contributions = contributions_from_hyper_images_and_galaxies(fitting_hyper_image.hyper_model_image,
                              fitting_hyper_image.hyper_galaxy_images, hyper_galaxies,
                              fitting_hyper_image.hyper_minimum_values)

        self._scaled_noise_map = scaled_noise_from_hyper_galaxies_and_contributions(self._contributions,
                                                                                    hyper_galaxies,
                                                                                    fitting_hyper_image.noise_map)

    @property
    def scaled_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self._scaled_chi_squareds)

    @property
    def scaled_noise_term(self):
        return noise_term_from_noise_map(self._scaled_noise_map)

    @property
    def scaled_noise_map(self):
        return self.map_to_scaled_array(self._scaled_noise_map)

    @property
    def scaled_chi_squareds(self):
        return self.map_to_scaled_array(self._scaled_chi_squareds)

    @property
    def contributions(self):
        return list(map(lambda contributions: self.map_to_scaled_array(contributions), self._contributions))


def residuals_from_data_and_model(data, model):
    """Compute the residuals between an observed charge injection lensing_image and post-cti model lensing_image.

    Residuals = (Data - Model).

    Parameters
    -----------
    data : ChInj.CIImage
        The observed charge injection lensing_image data.
    model : np.ndarray
        The model lensing_image.
    """
    return np.subtract(data, model)

def chi_squareds_from_residuals_and_noise(residuals, noise):
    """Computes a chi-squared lensing_image, by calculating the squared residuals between an observed charge injection \
    images and post-cti hyper_model_image lensing_image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    residuals
    noise : np.ndarray
        The noises in the lensing_image.
    """
    return np.square((np.divide(residuals, noise)))

def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model lensing_image's fit to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image, dividing by the noise_map in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    chi_squareds
    """
    return np.sum(chi_squareds)

def noise_term_from_noise_map(noise_map):
    """Compute the noise_map normalization term of an lensing_image, which is computed by summing the noise_map in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : grids.GridData
        The noise_map in each pixel.
    """
    return np.sum(np.log(2 * np.pi * noise_map ** 2.0))

def likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term):
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
    return -0.5 * (chi_squared_term + noise_term)

def blur_image_including_blurring_region(image, blurring_image, convolver):
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
    return convolver.convolve_image(image, blurring_image)

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

def scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise):
    """Use the contributions of each hyper galaxy to compute the scaled noise_map.
    Parameters
    -----------
    noise
    hyper_galaxies
    contributions : [ndarray]
        The contribution of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
    """
    scaled_noises = list(map(lambda hyper, contribution: hyper.scaled_noise_from_contributions(noise, contribution),
                             hyper_galaxies, contributions))
    return noise + sum(scaled_noises)

def likelihood_with_regularization_from_chi_squared_regularization_and_noise_terms(chi_squared_term,
                                                                                   regularization_term, noise_term):
    return -0.5 * (chi_squared_term + regularization_term + noise_term)

def evidence_from_reconstruction_terms(chi_squared_term, regularization_term, log_covariance_regularization_term,
                                       log_regularization_term, noise_term):
    return -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -
                   log_regularization_term + noise_term)

def unmasked_model_image_from_fitting_image(fitting_image, _unmasked_image):

    _model_image = fitting_image.padded_grids.image.convolve_array_1d_with_psf(_unmasked_image, fitting_image.psf)
    return fitting_image.padded_grids.image.scaled_array_from_array_1d(_model_image)