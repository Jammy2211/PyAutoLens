from autolens import exc
import numpy as np
from astropy.io import fits
import numba
from functools import wraps
import inspect

class Memoizer(object):

    def __init__(self):
        """
        Class to store the results of a function given a set of inputs.
        """
        self.results = {}
        self.calls = 0
        self.arg_names = None

    def __call__(self, func):
        """
        Memoize decorator. Any time a function is called that a memoizer has been attached to its results are stored in
        the results dictionary or retrieved from the dictionary if the function has already been called with those
        arguments.

        Note that the same memoizer persists over all instances of a class. Any state for a given instance that is not
        given in the representation of that instance will be ignored. That is, it is possible that the memoizer will
        give incorrect results if instance state does not affect __str__ but does affect the value returned by the
        memoized method.

        Parameters
        ----------
        func: function
            A function for which results should be memoized

        Returns
        -------
        decorated: function
            A function that memoizes results
        """
        if self.arg_names is not None:
            raise AssertionError("Instantiate a new Memoizer for each function")
        self.arg_names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = ", ".join(
                ["('{}', {})".format(arg_name, arg) for arg_name, arg in
                 list(zip(self.arg_names, args)) + [(k, v) for k, v in kwargs.items()]])
            if key not in self.results:
                self.calls += 1
            self.results[key] = func(*args, **kwargs)
            return self.results[key]

        return wrapper


@numba.jit(nopython=True, cache=True)
def total_image_pixels_from_mask(mask):

    total_image_pixels = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                total_image_pixels += 1

    return total_image_pixels

@numba.jit(nopython=True, cache=True)
def total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size):
    return total_image_pixels_from_mask(mask) * sub_grid_size ** 2

@numba.jit(nopython=True, cache=True)
def total_border_pixels_from_mask(mask):

    border_pixel_total = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                if mask[x + 1, y]     or mask[x - 1, y]     or mask[x, y + 1]     or mask[x, y - 1] or \
                   mask[x + 1, y + 1] or mask[x + 1, y - 1] or mask[x - 1, y + 1] or mask[x - 1, y - 1]:
                    border_pixel_total += 1

    return border_pixel_total

@numba.jit(nopython=True, cache=True)
def image_grid_2d_from_shape_and_pixel_scale(shape, pixel_scale):
    """
    Computes the arc second grids of every pixel on the data_vector-grid_coords.

    This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
    value and positive y value in arc seconds.
    """

    grid_2d = np.zeros((shape[0], shape[1], 2))

    x_cen = float(shape[0] - 1) / 2
    y_cen = float(shape[1] - 1) / 2

    for x in range(shape[0]):
        for y in range(shape[1]):

            grid_2d[x, y, 0] = (x - x_cen) * pixel_scale
            grid_2d[x, y, 1] = (y - y_cen) * pixel_scale

    return grid_2d

@numba.jit(nopython=True, cache=True)
def image_grid_masked_from_mask_and_pixel_scale(mask, pixel_scale):
    """
    Compute the masked_image grid_coords grids from a mask, using the center of every unmasked pixel.
    """

    grid_2d = image_grid_2d_from_shape_and_pixel_scale(mask.shape, pixel_scale)

    total_image_pixels = total_image_pixels_from_mask(mask)
    image_grid = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                image_grid[pixel_count, :] = grid_2d[x, y]
                pixel_count += 1

    return image_grid

@numba.jit(nopython=True, cache=True)
def sub_grid_masked_from_mask_pixel_scale_and_sub_grid_size(mask, pixel_scale, sub_grid_size):

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_grid = np.zeros(shape=(total_sub_pixels, 2))

    x_cen = float(mask.shape[0] - 1) / 2
    y_cen = float(mask.shape[1] - 1) / 2

    sub_index = 0

    sub_half = pixel_scale / 2
    sub_step = pixel_scale / (sub_grid_size+1)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            if not mask[x, y]:

                x_arcsec = (x - x_cen) * pixel_scale
                y_arcsec = (y - y_cen) * pixel_scale

                for x1 in range(sub_grid_size):
                    for y1 in range(sub_grid_size):

                        sub_grid[sub_index, 0] = x_arcsec - sub_half + (x1+1) * sub_step
                        sub_grid[sub_index, 1] = y_arcsec - sub_half + (y1+1) * sub_step
                        sub_index += 1

    return sub_grid

@numba.jit(nopython=True, cache=True)
def grid_to_pixel_from_mask(mask):

    total_image_pixels = total_image_pixels_from_mask(mask)
    grid_to_pixel = np.zeros(shape=(total_image_pixels, 2))
    pixel_count = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                grid_to_pixel[pixel_count, :] = x, y
                pixel_count += 1

    return grid_to_pixel

@numba.jit(nopython=True, cache=True)
def sub_to_image_from_mask(mask, sub_grid_size):
    """ Compute the pairing of every sub-pixel to its original masked_image pixel from a mask. """

    total_sub_pixels = total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size)

    sub_to_image = np.zeros(shape=total_sub_pixels)
    image_index = 0
    sub_index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                for x1 in range(sub_grid_size):
                    for y1 in range(sub_grid_size):
                        sub_to_image[sub_index] = image_index
                        sub_index += 1

                image_index += 1

    return sub_to_image

@numba.jit(nopython=True, cache=True)
def border_pixels_from_mask(mask):

    border_pixel_total = total_border_pixels_from_mask(mask)

    border_pixels = np.zeros(border_pixel_total)
    border_index = 0
    image_index = 0
    
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                if mask[x + 1, y] or mask[x - 1, y] or mask[x, y + 1] or mask[x, y - 1] or \
                        mask[x + 1, y + 1] or mask[x + 1, y - 1] or mask[x - 1, y + 1] or mask[x - 1, y - 1]:
                    border_pixels[border_index] = image_index
                    border_index += 1
    
                image_index += 1
    
    return border_pixels

@numba.jit(nopython=True, cache=True)
def border_sub_pixels_from_mask_pixel_scale_and_sub_grid_size(mask, pixel_scale, sub_grid_size):
    """Compute the border masked_image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on
    its edge, therefore neighboring a pixel with a *True* value.
    """
    border_pixel_total = total_border_pixels_from_mask(mask)
    border_sub_pixels = np.zeros(border_pixel_total)

    image_index = 0

    x_cen = float(mask.shape[0] - 1) / 2
    y_cen = float(mask.shape[1] - 1) / 2

    sub_half = pixel_scale / 2
    sub_step = pixel_scale / (sub_grid_size+1)

    border_index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                if mask[x + 1, y] or mask[x - 1, y] or mask[x, y + 1] or mask[x, y - 1] or \
                        mask[x + 1, y + 1] or mask[x + 1, y - 1] or mask[x - 1, y + 1] or mask[x - 1, y - 1]:

                    x_arcsec = (x - x_cen) * pixel_scale
                    y_arcsec = (y - y_cen) * pixel_scale

                    sub_grid = np.zeros((sub_grid_size ** 2, 2))
                    sub_index = 0
    
                    for x1 in range(sub_grid_size):
                        for y1 in range(sub_grid_size):

                            sub_grid[sub_index, 0] = x_arcsec - sub_half + (x1 + 1) * sub_step
                            sub_grid[sub_index, 1] = y_arcsec - sub_half + (y1 + 1) * sub_step
                            sub_index += 1
    
                    sub_grid_radii = np.add(np.square(sub_grid[:, 0]), np.square(sub_grid[:, 1]))
                    border_sub_index = image_index * (sub_grid_size ** 2) + np.argmax(sub_grid_radii)
                    border_sub_pixels[border_index] = border_sub_index
                    border_index += 1

                image_index += 1
    
    return border_sub_pixels

@numba.jit(nopython=True, cache=True)
def mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_arcsec, centre=(0.0, 0.0)):

    mask = np.full(shape, True)

    x_cen = (float(mask.shape[0] - 1) / 2) + centre[0]
    y_cen = (float(mask.shape[1] - 1) / 2) + centre[1]

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            x_arcsec = (x - x_cen) * pixel_scale
            y_arcsec = (y - y_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if r_arcsec <= radius_arcsec:
                mask[x,y] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec,
                                                  centre=(0.0, 0.0)):

    mask = np.full(shape, True)

    x_cen = (float(mask.shape[0] - 1) / 2) + centre[0]
    y_cen = (float(mask.shape[1] - 1) / 2) + centre[1]

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            x_arcsec = (x - x_cen) * pixel_scale
            y_arcsec = (y - y_cen) * pixel_scale

            r_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            if r_arcsec <= outer_radius_arcsec and r_arcsec >= inner_radius_arcsec:
                mask[x,y] = False

    return mask

@numba.jit(nopython=True, cache=True)
def mask_blurring_from_mask_and_psf_shape(mask, psf_shape):

    blurring_mask = np.full(mask.shape, True)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                for y1 in range((-psf_shape[1] + 1) // 2, (psf_shape[1] + 1) // 2):
                    for x1 in range((-psf_shape[0] + 1) // 2, (psf_shape[0] + 1) // 2):
                        if 0 <= x + x1 <= mask.shape[0] - 1 and 0 <= y + y1 <= mask.shape[1] - 1:
                            if mask[x + x1, y + y1]:
                                blurring_mask[x + x1, y + y1] = False
                        else:
                            raise exc.MaskException(
                                "setup_blurring_mask extends beyond the sub_grid_size of the mask - pad the masked_image"
                                "before masking")

    return blurring_mask

@numba.jit(nopython=True, cache=True)
def map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d):

    total_image_pixels = total_image_pixels_from_mask(mask)

    array_1d = np.zeros(shape=total_image_pixels)
    index = 0

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not mask[x, y]:
                array_1d[index] = array_2d[x, y]
                index += 1

    return array_1d

@numba.jit(nopython=True, cache=True)
def map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two):

    array_2d = np.zeros(shape)

    for index in range(len(one_to_two)):
        array_2d[one_to_two[index,0], one_to_two[index,1]] = array_1d[index]

    return array_2d

@numba.jit(nopython=True, cache=True)
def map_masked_deflections_to_2d_deflections_from_deflections_shape_and_one_to_two(deflections, shape, one_to_two):

    deflections_2d = np.zeros((shape[0], shape[1], 2))

    for index in range(len(one_to_two)):
        deflections_2d[one_to_two[index,0], one_to_two[index,1 ], 0] = deflections[index, 0]
        deflections_2d[one_to_two[index,0], one_to_two[index, 1], 1] = deflections[index, 1]

    return deflections_2d

@numba.jit(nopython=True, cache=True)
def map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape):

    array_2d = np.zeros(shape)

    index = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            array_2d[x, y] = array_1d[index]
            index += 1

    return array_2d

@numba.jit(nopython=True, cache=True)
def map_unmasked_deflections_to_2d_deflections_from_deflections_and_shape(deflections, shape):

    deflections_2d = np.zeros((shape[0], shape[1], 2))

    index = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            deflections_2d[x, y, 0] = deflections[index, 0]
            deflections_2d[x, y, 1] = deflections[index, 1]
            index += 1

    return deflections_2d

def numpy_array_to_fits(array, file_path):

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array, new_hdr)
    hdu.writeto(file_path + '.fits')

def numpy_array_from_fits(file_path, hdu):
    hdu_list = fits.open(file_path + '.fits')
    return np.array(hdu_list[hdu].data)

def compute_variances_from_noise(noise):
    """The variances are the signal_to_noise_ratio (standard deviations) squared."""
    return np.square(noise)