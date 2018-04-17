import numpy as np
from auto_lens.imaging import imaging

def setup_data(mask, data):
    """ Given an image.Mask, convert a 2d data of data values to a 1D vector, structured for efficient lens modeling \
    calculations.

    Parameters
    ----------
    data : ndarray
        A 2D data of data, e.g. the image, noise-map, etc.
    mask : imaging.Mask
        The image mask containing the pixels we are computing the coordinates of and the image pixel_dimensions / pixel scale.

    Returns
    -------
    One-dimensional data containing data of data.
    """
    image_pixels = mask.pixels_in_mask
    data_1d = np.zeros(shape=(image_pixels))
    data_count = 0
    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                data_1d[data_count] = data[y, x]
                data_count += 1

    return data_1d

def setup_mapper_2d(mask):
    """ Given an image.Mask, setup an array that can be used to map the input masks coordinates back to their 2D image \
    pixels.

    Parameters
    ----------
    mask : imaging.Mask
        The image mask containing the pixels we are computing the coordinates of and the image pixel_dimensions / pixel scale.
    """
    image_pixels = mask.pixels_in_mask
    mapper_2d = np.zeros(shape=(image_pixels, 2), dtype=int)
    data_count = 0
    for y in range(mask.pixel_dimensions[0]):
        for x in range(mask.pixel_dimensions[1]):
            if mask[y, x] == False:
                mapper_2d[data_count, :] = np.array([y, x])
                data_count += 1

    return mapper_2d

class AnalysisData(object):

    def __init__(self, mask, image, noise, psf, sub_grid_size=2):
        """The core grouping of lens modeling data, including the image data, noise-map and psf. Optional \
        inputs (e.g. effective exposure time map / positional image pixels) have their functionality automatically \
        switched on or off depending on if they are included.

        A mask must be supplied, which converts all 2D image quantities to data vectors. These vectors are designed to \
        provide optimal lens modeling efficiency. Image region vectors are also set-up, which describe specific \
        regions of the image. These are used for specific calculations, like the image sub-grid, and to provide \

        Parameters
        ----------
        mask : imaging.Mask
            The image mask, where False indicates a pixel is included in the analysis.
        image : image.Image
            The image data, in electrons per second.
        noise : image.Noise
            The image noise-map, in variances in electrons per second.
        psf : image.PSF
            The image PSF
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        self.image = AnalysisImage(mask, image)
        self.noise = setup_data(mask, noise)
        self.psf = psf
        self.coordinates = setup_image_coordinates(mask)
        self.sub_coordinates = setup_sub_coordinates(mask, sub_grid_size)
        self.blurring_coordinates = setup_blurring_coordinates(mask, self.psf.shape)
        self.border_pixels = setup_border_pixels(mask)


class AnalysisArray(np.ndarray):

    def __new__(cls, mask, data):
        """

        Parameters
        ----------
        mask_array : ndarray
            The boolean array of masked pixels (False = pixel is not masked and included in analysis)

        Returns
        -------
            An empty array
        """
        data = setup_data(mask, data).view(cls)
        data.mapper_2d = setup_mapper_2d(mask)
        data.shape_2d = mask.shape
        return data

    def map_to_2d(self):
        array = np.zeros((self.shape_2d))
        for data_count, [y, x] in enumerate(self.mapper_2d):
            array[y,x] = self[data_count]

        return array


# TODO : This may be where we put our hyper-parameter functions for the image and noise maps. E.g. for noise map \
# TODO : scaling, we could have an AnalysisNoise class with functions def scale_lens_noise, etc. I'll decide on this \
# TODO : As we continue to develop the analysis code.

# TODO : Can we handle these classes using inheritance of Analysis Array? I can't figue out how but this works...

class AnalysisImage(np.ndarray):

    def __new__(cls, mask, image):
        return AnalysisArray(mask, image)