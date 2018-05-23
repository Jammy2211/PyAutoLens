from auto_lens.imaging import grids
from auto_lens import ray_tracing

# TODO : Do this convolution in 1D eventually..

def generate_blurred_light_profie_image(ray_tracing, psf, grid_mappers):
    """For a given ray-tracing model, compute the light profile image(s) of its galaxies and blur them with the
    PSF.

    Parameters
    ----------
    ray_tracing : ray_tracing.TraceImageAndSource
        The ray-tracing configuration of the model galaxies and their profiles.
    psf : imaging.PSF
        The 2D Point Spread Function (PSF).
    grid_mappers : grids.GridMapperCollection
        The collection of grid mappings, used to map images from 2d and 1d.
    """

    image_light_profile = ray_tracing.generate_image_of_galaxy_light_profiles()
    blurring_image_light_profile = ray_tracing.generate_blurring_image_of_galaxy_light_profiles()
    return blur_image_including_blurring_region(image_light_profile, grid_mappers.image_to_pixel, psf,
                                                blurring_image_light_profile, grid_mappers.blurring_to_pixel)

def blur_image_including_blurring_region(image, image_to_pixel, psf, blurring_image=None, blurring_to_pixel=None):
    """For a given image and blurring region, convert them to 2D and blur with the PSF, then return as the 1D DataGrid.

    Parameters
    ----------
    image : grids.GridData
        The image data using the GridData 1D representation.
    image_to_pixel : grids.GridMapperDataToPixel
        The mapping between a 1D image pixel (GridData) and 2D image location.
    psf : imaging.PSF
        The 2D Point Spread Function (PSF).
    blurring_image : grids.GridData
        The blurring regioon data, using the GridData 1D representation.
    blurring_to_pixel : grid.GridMapperBlurringToPixel
        The mapping between a 1D blurring image pixel (GridData) and 2D image location.
    """

    image_2d = image_to_pixel.map_to_2d(image)

    if blurring_image is not None:
        image_2d += blurring_to_pixel.map_to_2d(blurring_image)

    image_2d_blurred = psf.convolve_with_image(image_2d)

    return grids.GridData(image_to_pixel.map_to_1d(image_2d_blurred))