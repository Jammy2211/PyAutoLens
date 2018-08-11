import numpy as np
from autolens.imaging import masked_image as mi
from autolens.imaging import mask
from autolens.analysis import ray_tracing

# TODO : Can we make model_image, galaxy_images, minimum_Values a part of hyper galaxies?

minimum_value_profile = 0.1


class AbstractHyperFitter(object):

    # TODO : Removing Pycharm inspectioon, can it be done better?

    masked_image = None
    tracer = None
    hyper_model_image = None
    hyper_galaxy_images = None
    hyper_minimum_values = None

    @property
    def contributions(self):
        return contributions_from_hyper_images_and_galaxies(self.hyper_model_image, self.hyper_galaxy_images,
               self.tracer.hyper_galaxies, self.hyper_minimum_values)

    @property
    def scaled_noise(self):
        return scaled_noise_from_hyper_galaxies_and_contributions(self.contributions, self.tracer.hyper_galaxies,
                                                                  self.masked_image.noise)

    @property
    def scaled_noise_term(self):
        return noise_term_from_data(self.scaled_noise)

class ProfileFitter(object):

    def __init__(self, masked_image, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        tracer: ray_tracing.Tracer
            An object describing the model
        """
        self.masked_image = masked_image
        self.tracer = tracer

    @property
    def noise_term(self):
        return noise_term_from_data(self.masked_image.noise)

    @property
    def image(self):
        return self.tracer.generate_image_of_galaxy_light_profiles()

    @property
    def blurring_region_image(self):
        return self.tracer.generate_blurring_image_of_galaxy_light_profiles()

    @property
    def blurred_image(self):
        return self.masked_image.convolver_image.convolve_image_jit(self.image,
                                                             self.blurring_region_image)

    @property
    def blurred_image_residuals(self):
        return residuals_from_image_and_model(self.masked_image, self.blurred_image)

    @property
    def blurred_image_chi_squareds(self):
        return chi_squareds_from_residuals_and_noise(self.blurred_image_residuals, self.masked_image.noise)

    @property
    def blurred_image_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.blurred_image_chi_squareds)

    @property
    def blurred_image_likelihood(self):
        """
        Fit the weighted_data using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.
        """
        return likelihood_from_chi_squared_and_noise_terms(self.blurred_image_chi_squared_term,
                                                           self.noise_term)


class HyperProfileFitter(ProfileFitter, AbstractHyperFitter):

    def __init__(self, masked_image, tracer, hyper_model_image, hyper_galaxy_images, hyper_minimum_values):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        tracer: ray_tracing.Tracer
            An object describing the model
        """
        super(HyperProfileFitter, self).__init__(masked_image, tracer)
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values

    @property
    def blurred_image_scaled_chi_squareds(self):
        return chi_squareds_from_residuals_and_noise(self.blurred_image_residuals, self.scaled_noise)

    @property
    def blurred_image_scaled_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.blurred_image_scaled_chi_squareds)

    @property
    def blurred_image_scaled_likelihood(self):
        """
        Fit the weighted_data using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.
        """
        return likelihood_from_chi_squared_and_noise_terms(self.blurred_image_scaled_chi_squared_term,
                                                           self.scaled_noise_term)


class PixelizedFitter(ProfileFitter):

    def __init__(self, masked_image, sparse_mask, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        sparse_mask: mask.SparseMask
            A mask describing which pixels should be used in clustering for pixelizations
        borders : mask.BorderCollection
            The pixels representing the border of each plane, used for relocation.
        tracer: ray_tracing.Tracer
            An object describing the model
        """
        super().__init__(masked_image, tracer)
        self.sparse_mask = sparse_mask

    def fit_data_with_pixelization(self):
        """Fit the weighted_data using the ray_tracing model, where only pixelizations are used to represent the galaxy
        images.
        """

        pix_pre_fit = self.tracer.reconstructors_from_source_plane(self.image.borders, self.sparse_mask)
        pix_fit = pix_pre_fit.reconstruct_image(self.image, self.image.noise,
                                                self.image.convolver_mapping_matrix)

        model_image = pix_fit.model_image_from_reconstruction_jit()

        return pixelization_evidence_from_data_model_and_pix(self.image, self.image.noise, model_image, pix_fit)

    def fit_data_with_pixelization_profiles_and_model_images(self, model_image, galaxy_images):
        raise NotImplementedError()

    def fit_data_with_pixelization_and_profiles(self):
        raise NotImplementedError()


def blur_image_including_blurring_region(image, blurring_image, convolver):
    """For a given masked_image and blurring region, convert them to 2D and blur with the PSF, then return as
    the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The masked_image weighted_data using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region weighted_data, using the GridData 1D representation.
    convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return convolver.convolve_image_jit(image, blurring_image)

def residuals_from_image_and_model(image, model):
    """Compute the residuals between an observed charge injection masked_image and post-cti model masked_image.

    Residuals = (Data - Model).

    Parameters
    -----------
    image : ChInj.CIImage
        The observed charge injection masked_image data.
    model : np.ndarray
        The model masked_image.
    """
    return np.subtract(image, model)

def chi_squareds_from_residuals_and_noise(residuals, noise):
    """Computes a chi-squared masked_image, by calculating the squared residuals between an observed charge injection \
    images and post-cti model_image masked_image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    masked_image : ChInj.CIImage
        The observed charge injection masked_image data (includes the mask).
    mask : ChInj.CIMask
        The mask of the charge injection masked_image data.
    noise : np.ndarray
        The noises in the masked_image.
    model : np.ndarray
        The model_image masked_image.
    """
    return np.square((np.divide(residuals, noise)))

def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model masked_image's fit to the weighted_data, by taking the difference between the
    observed masked_image and model ray-tracing masked_image, dividing by the noise in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    masked_image : grids.GridData
        The masked_image weighted_data.
    noise : grids.GridData
        The noise in each pixel.
    model : grids.GridData
        The model masked_image of the weighted_data.
    """
    return np.sum(chi_squareds)

def noise_term_from_data(noise):
    """Compute the noise normalization term of an masked_image, which is computed by summing the noise in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise : grids.GridData
        The noise in each pixel.
    """
    return np.sum(np.log(2 * np.pi * noise ** 2.0))

def likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term):
    """Compute the likelihood of a model masked_image's fit to the weighted_data, by taking the difference between the
    observed masked_image and model ray-tracing masked_image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - weighted_data) of every pixel divided by the noise in each pixel, all
    squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise is also included, by summing the log noise value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    masked_image : grids.GridData
        The masked_image weighted_data.
    noise : grids.GridData
        The noise in each pixel.
    model : grids.GridData
        The model masked_image of the weighted_data.
    """
    return -0.5 * (chi_squared_term + noise_term)

def contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies,
                                                 minimum_values):
    """Use the model masked_image and galaxy masked_image (computed in the previous phase of the pipeline) to determine the
    contributions of each hyper galaxy.

    Parameters
    -----------
    minimum_values
    hyper_model_image : ndarray
        The best-fit model masked_image to the weighted_data, from a previous phase of the pipeline
    hyper_galaxy_images : [ndarray]
        The best-fit model masked_image of each hyper-galaxy, which can tell us how much flux each pixel contributes to.
    hyper_galaxies : [galaxy.HyperGalaxy]
        Each hyper-galaxy which is used to determine its contributions.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_preload_images(hyper_model_image, galaxy_image, minimum_value),
                    hyper_galaxies, hyper_galaxy_images, minimum_values))

def scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise):
    """Use the contributions of each hyper galaxy to compute the scaled noise.
    Parameters
    -----------
    contributions : [ndarray]
        The contribution of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
    """
    scaled_noises = list(map(lambda hyper, contribution: hyper.scaled_noise_from_contributions(noise, contribution),
                             hyper_galaxies, contributions))
    return noise + sum(scaled_noises)

def pixelization_evidence_from_data_model_and_pix(image, noise, model, pix_fit):
    return -0.5 * (chi_squared_term_from_chi_squareds(image, noise, model)
                   + pix_fit.regularization_term_from_reconstruction()
                   + pix_fit.log_determinant_of_matrix_cholesky(pix_fit.covariance_regularization)
                   - pix_fit.log_determinant_of_matrix_cholesky(pix_fit.regularization)
                   + noise_term_from_data(noise))

# noinspection PyUnusedLocal
def fit_data_with_pixelization_and_profiles(grid_data_collection, pixelization, convolver, tracer,
                                            mapper_cluster, image=None):
    # TODO: Implement me
    raise NotImplementedError("fit_data_with_pixelization_and_profiles has not been implemented")
