import numpy as np
from autolens.imaging import masked_image as mi
from autolens.imaging import mask
from autolens.analysis import ray_tracing

# TODO : Can we make model_image, galaxy_images, minimum_Values a part of hyper galaxies?

minimum_value_profile = 0.1


class Fitter(object):

    def __init__(self, masked_image, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An image that has been masked for efficiency
        tracer: ray_tracing.Tracer
            An object describing the model
        """
        self.image = masked_image
        self.tracer = tracer

    def fit_data_with_profiles_and_model_images(self, model_image, galaxy_images):
        """Fit the weighted_data using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.

        Parameters
        ----------
        model_image : ndarray
            The best-fit model image to the weighted_data, from a previous phase of the pipeline
        galaxy_images : [ndarray]
            The best-fit model image of each hyper-galaxy, which can tell us how much flux each pixel contributes
            to.
        """
        contributions = generate_contributions(model_image, galaxy_images, self.tracer.hyper_galaxies,
                                               [minimum_value_profile for _ in range(len(galaxy_images))])
        scaled_noise = self.scaled_noise_for_contributions(contributions)
        blurred_model_image = self.blurred_light_profile_image()
        fitness = compute_likelihood(self.image, scaled_noise, blurred_model_image)
        return fitness

    def scaled_noise_for_contributions(self, contributions):
        """Use the contributions of each hyper galaxy to compute the scaled noise.
        Parameters
        -----------
        contributions : [ndarray]
            The contribution of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
        """
        scaled_noises = list(
            map(lambda hyper, contribution: hyper.scaled_noise_for_contributions(self.image.noise, contribution),
                self.tracer.hyper_galaxies, contributions))
        return self.image.noise + sum(scaled_noises)

    def blurred_light_profile_image(self):
        """
        For a given ray-tracing model, compute the light profile image(s) of its galaxies and blur them with the
        PSF.
        """
        image_light_profile = self.tracer.generate_image_of_galaxy_light_profiles()
        blurring_image_light_profile = self.tracer.generate_blurring_image_of_galaxy_light_profiles()
        return blur_image_including_blurring_region(image_light_profile, blurring_image_light_profile,
                                                    self.image.convolver_image)

    def fit_data_with_profiles(self):
        """
        Fit the weighted_data using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.
        """
        blurred_model_image = self.blurred_light_profile_image()
        fitness = compute_likelihood(self.image, self.image.noise, blurred_model_image)
        return fitness


class PixelizedFitter(Fitter):
    def __init__(self, masked_image, sparse_mask, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An image that has been masked for efficiency
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

        return compute_pixelization_evidence(self.image, self.image.noise, model_image, pix_fit)

    def fit_data_with_pixelization_profiles_and_model_images(self, model_image, galaxy_images):
        raise NotImplementedError()

    def fit_data_with_pixelization_and_profiles(self):
        raise NotImplementedError()


def generate_contributions(model_image, galaxy_images, hyper_galaxies, minimum_values):
    """Use the model image and galaxy image (computed in the previous phase of the pipeline) to determine the
    contributions of each hyper galaxy.

    Parameters
    -----------
    minimum_values
    model_image : ndarray
        The best-fit model image to the weighted_data, from a previous phase of the pipeline
    galaxy_images : [ndarray]
        The best-fit model image of each hyper-galaxy, which can tell us how much flux each pixel contributes to.
    hyper_galaxies : [galaxy.HyperGalaxy]
        Each hyper-galaxy which is used to determine its contributions.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_model_images(model_image, galaxy_image, minimum_value),
                    hyper_galaxies, galaxy_images, minimum_values))


def blur_image_including_blurring_region(image, blurring_image, convolver):
    """For a given image and blurring region, convert them to 2D and blur with the PSF, then return as
    the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The image weighted_data using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region weighted_data, using the GridData 1D representation.
    convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return convolver.convolve_image_jit(image, blurring_image)


def compute_likelihood(image, noise, model_image):
    """Compute the likelihood of a model image's fit to the weighted_data, by taking the difference between the
    observed image and model ray-tracing image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - weighted_data) of every pixel divided by the noise in each pixel, all
    squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise is also included, by summing the log noise value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    image : grids.GridData
        The image weighted_data.
    noise : grids.GridData
        The noise in each pixel.
    model_image : grids.GridData
        The model image of the weighted_data.
    """
    return -0.5 * (compute_chi_sq_term(image, noise, model_image) + compute_noise_term(noise))


def compute_pixelization_evidence(image, noise, model_image, pix_fit):
    return -0.5 * (compute_chi_sq_term(image, noise, model_image)
                   + pix_fit.regularization_term_from_reconstruction()
                   + pix_fit.log_determinant_of_matrix_cholesky(pix_fit.covariance_regularization)
                   - pix_fit.log_determinant_of_matrix_cholesky(pix_fit.regularization)
                   + compute_noise_term(noise))


def compute_chi_sq_term(image, noise, model_image):
    """Compute the chi-squared of a model image's fit to the weighted_data, by taking the difference between the
    observed image and model ray-tracing image, dividing by the noise in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    image : grids.GridData
        The image weighted_data.
    noise : grids.GridData
        The noise in each pixel.
    model_image : grids.GridData
        The model image of the weighted_data.
    """
    return np.sum((np.add(image.view(float), - model_image) / noise) ** 2.0)


def compute_noise_term(noise):
    """Compute the noise normalization term of an image, which is computed by summing the noise in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise : grids.GridData
        The noise in each pixel.
    """
    return np.sum(np.log(2 * np.pi * noise ** 2.0))


# noinspection PyUnusedLocal
def fit_data_with_pixelization_and_profiles(grid_data_collection, pixelization, convolver, tracer,
                                            mapper_cluster, image=None):
    # TODO: Implement me
    raise NotImplementedError("fit_data_with_pixelization_and_profiles has not been implemented")
