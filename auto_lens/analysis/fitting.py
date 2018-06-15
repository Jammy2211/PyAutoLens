import numpy as np

from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing


def likelihood_for_image_tracer_pixelization_and_instrumentation(image, tracer, pixelization, instrumentation):
    # TODO: This function should take a tracer and return a likelihood. The ModelAnalysis class in the pipeline module
    # TODO: will construct the tracer using a non linear optimiser and priors.
    return 1.0


def fit_data_with_model(grid_datas, grid_mappers, kernel_convolver, tracer):
    """Fit the data using the ray_tracing model

    Parameters
    ----------
    grid_datas : grids.GridDataCollection
        The collection of grid data-sets (image, noise, psf, etc.)
    grid_mappers : grids.GridMapperCollection
        The collection of grid mappings, used to map images from 2d and 1d.
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    """
    blurred_model_image = generate_blurred_light_profile_image(tracer, kernel_convolver)
    return compute_likelihood(grid_datas.image, grid_datas.noise, blurred_model_image)

def generate_blurred_light_profile_image(tracer, kernel_convolver):
    """For a given ray-tracing model, compute the light profile image(s) of its galaxies and blur them with the
    PSF.

    Parameters
    ----------
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    image_light_profile = tracer.generate_image_of_galaxy_light_profiles()
    blurring_image_light_profile = tracer.generate_blurring_image_of_galaxy_light_profiles()
    return blur_image_including_blurring_region(image_light_profile, blurring_image_light_profile, kernel_convolver)

def blur_image_including_blurring_region(image, blurring_image, kernel_convolver):
    """For a given image and blurring region, convert them to 2D and blur with the PSF, then return as the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The image data using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region data, using the GridData 1D representation.
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return grids.GridData(kernel_convolver.convolve_array(image, blurring_image))

def compute_likelihood(image, noise, model_image):
    """Compute the likelihood of a model image's fit to the data, by taking the difference between the observed \
    image and model ray-tracing image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - data) of every pixel divided by the noise in each pixel, all squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise is also included, by summing the log noise value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    image : grids.GridData
        The image data.
    noise : grids.GridData
        The noise in each pixel.
    model_image : grids.GridData
        The model image of the data.
    """
    return -0.5 * (np.sum(((image - model_image) / noise) ** 2.0 + np.log(2 * np.pi * noise ** 2.0)))