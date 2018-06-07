import numpy as np

from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing


def fit_data_with_model(grid_datas, grid_mappers, tracer):
    """Fit the data using the ray_tracing model

    Parameters
    ----------
    grid_datas : grids.GridDataCollection
        The collection of grid data-sets (image, noise, psf, etc.)
    grid_mappers : grids.GridMapperCollection
        The collection of grid mappings, used to map images from 2d and 1d.
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    """
    blurred_model_image = generate_blurred_light_profile_image(tracer, grid_datas.psf, grid_mappers)
    return compute_likelihood(grid_datas.image, grid_datas.noise, blurred_model_image)


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


def generate_blurred_light_profile_image(tracer, psf, grid_mappers):
    """For a given ray-tracing model, compute the light profile image(s) of its galaxies and blur them with the
    PSF.

    Parameters
    ----------
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    psf : imaging.PSF
        The 2D Point Spread Function (PSF).
    grid_mappers : grids.GridMapperCollection
        The collection of grid mappings, used to map images from 2d and 1d.
    """

    image_light_profile = tracer.generate_image_of_galaxy_light_profiles()
    blurring_image_light_profile = tracer.generate_blurring_image_of_galaxy_light_profiles()
    return blur_image_including_blurring_region(image_light_profile, grid_mappers.image_to_pixel, psf,
                                                blurring_image_light_profile, grid_mappers.blurring_to_pixel)


# TODO : Do this convolution in 1D eventually..

def blur_image_including_blurring_region(image, image_to_pixel, psf, blurring_image=None, blurring_to_pixel=None):
    """For a given image and blurring region, convert them to 2D and blur with the PSF, then return as the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The image data using the GridData 1D representation.
    image_to_pixel : grids.GridMapperDataToPixel
        The mapping between a 1D image pixel (GridData) and 2D image location.
    psf : imaging.PSF
        The 2D Point Spread Function (PSF).
    blurring_image : ndarray
        The blurring region data, using the GridData 1D representation.
    blurring_to_pixel : grid.GridMapperBlurringToPixel
        The mapping between a 1D blurring image pixel (GridData) and 2D image location.
    """

    image_2d = image_to_pixel.map_to_2d(image)

    if blurring_image is not None:
        image_2d += blurring_to_pixel.map_to_2d(blurring_image)

    image_2d_blurred = psf.convolve(image_2d)

    return grids.GridData(image_to_pixel.map_to_1d(image_2d_blurred))
