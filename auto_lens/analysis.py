from auto_lens.imaging import grids
from auto_lens import ray_tracing


def compute_blurred_light_profile_image(ray_tracing, psf, data_to_pixel, blurring_to_pixel):
    """For a given ray-tracing model, compute the model image of its light profile components and blur them with the
    PSF.

    Parameters
    ----------
    ray_tracing : ray_tracing.TraceImageAndSource
        The ray-tracing configuration of the model galaxies and their profiles.
    psf : imaging.PSF
        The 2D Point Spread Function (PSF).
    data_to_pixels : grids.GridMapperDataToPixel
        The mapping between 1D GridData points to its 2D image.
    blurring_to_pixel : grid.GridMapperBlurringToPixel
        The mapping between 1D GridBlurring
    """

    # TODO : Do this convolution in 1D eventually..

    galaxy_image_2d = data_to_pixels.map_to_2d(image)
    galaxy_image_blurred_2d = psf.convolve_with_image(galaxy_image_2d)
    galaxy_image_blurred_1d = data_to_pixels.map_to_1d(galaxy_image_blurred_2d)
    return grids.GridData(galaxy_image_blurred_1d)
