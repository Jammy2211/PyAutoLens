from auto_lens.imaging import grids
from auto_lens import ray_tracing

# TODO : Do this convolution in 1D eventually..

def compute_blurred_light_profile_image(image, image_to_pixel, psf, blurring_image=None, blurring_to_pixel=None):
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