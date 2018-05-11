from scipy.stats import norm
from astropy.io import fits
import numpy as np
from matplotlib import pyplot
import scipy.signal
import os
import logging

from auto_lens.imaging import grids

logging.basicConfig()
logger = logging.getLogger(__name__)

data_path = "{}/../data/".format(os.path.dirname(os.path.realpath(__file__)))

# TODO : These will ultimately be performed in the ExposureTime class, once the module takes shape :)

def convert_array_to_counts(array, exposure_time_array):
    """For an array (in electrons per second) and exposure time array, return an array in units counts.

    Parameters
    ----------
    array : ndarray
        The image from which the Poisson signal_to_noise_ratio map is estimated.
    exposure_time_array : ndarray
        The exposure time in each image pixel."""
    return np.multiply(array, exposure_time_array)

def convert_array_to_electrons_per_second(array, exposure_time_array):
    """For an array (in counts) and exposure time array, convert the array to units electrons per second
    Parameters
    ----------
    array : ndarray
        The image from which the Poisson signal_to_noise_ratio map is estimated.
    exposure_time_array : ndarray
        The exposure time in each image pixel.
    """
    return np.divide(array, exposure_time_array)

# TODO : and these two in the noise class(es)

def estimate_noise_in_quadrature(image_counts, sigma_counts):
    return np.sqrt(image_counts + np.square(sigma_counts))

def estimate_noise_from_image(image, exposure_time, background_noise):
    """Estimate the two-dimensional signal_to_noise_ratio of an input image, including signal_to_noise_ratio due to Poisson counting statistics and \
    a background component.

    Parameters
    ----------
    image : ndarray
        The image in electrons per second, used to estimate the Poisson signal_to_noise_ratio map.
    exposure_time : float or ndarray
        The exposure time in each image pixel, used to convert the image from electrons per second to counts.
    background_noise : float or ndarray
        The standard deviation estimate of the 1D Gaussian level of signal_to_noise_ratio in each pixxel due to background signal_to_noise_ratio \
        sources, in electrns per second
    exposure_time_mean : float
        The mean exposure time of the image and therefore background.
    """
    image_counts = convert_array_to_counts(image, exposure_time)
    background_noise_counts = convert_array_to_counts(background_noise, exposure_time)
    noise_counts = estimate_noise_in_quadrature(image_counts, background_noise_counts)
    return convert_array_to_electrons_per_second(noise_counts, exposure_time)

def numpy_array_from_fits(file_path, hdu):
    hdu_list = fits.open(file_path)  # Open the fits file
    return np.array(hdu_list[hdu].data)

def output_for_fortran(path, array, image_name):
    """ Outputs the data-array for the Fortran AutoLens code. This will ultimately be removed so you can ignore
    and I've not bothered with unit-tests.
    Parameters
    ----------
    array: ndarray (or Noise or PSF)
        The image array
    path : str
        The directory the files are output too
    image_name : str
        The name of the image for this file
    """
    if isinstance(array, PSF):
        file_path = path + image_name + "PSF.dat"
    elif isinstance(array, Noise):
        file_path = path + image_name + "Noise.dat"
    else:
        file_path = path + image_name + ".dat"

    shape = array.data.shape

    with open(file_path, "w+") as f:
        for ix, x in enumerate(range(shape[0])):
            for iy, y in enumerate(range(shape[1])):
                line = str(round(float(ix + 1), 2))
                line += ' ' * (8 - len(line))
                line += str(round(float(iy + 1), 2))
                line += ' ' * (16 - len(line))
                line += str(float(array.data[ix,iy])) + '\n'
                f.write(line)


class DataGrid(object):

    def __init__(self, pixel_dimensions, pixel_scale):
        """
        Class storing the grids for 2D pixel grids (e.g. image, PSF, signal_to_noise_ratio).

        Parameters
        ----------
        pixel_dimensions : (int, int)
            The (x,y) dimensions of the pixel grid_coords.
        pixel_scale : float
            The arc-second to pixel conversion factor of each pixel.
        """

        self.pixel_scale = pixel_scale
        self.pixel_dimensions = pixel_dimensions
        self.arc_second_dimensions = self.pixel_dimensions_to_arc_seconds(self.pixel_dimensions)

    @classmethod
    def from_arcsecond_dimensions(cls, arc_second_dimensions, pixel_scale):
        cls.pixel_scale = pixel_scale
        pixel_dimensions = cls.arc_second_dimensions_to_pixel(cls, arc_second_dimensions)
        return DataGrid(pixel_dimensions, pixel_scale)

    @property
    def central_pixels(self):
        return tuple(map(lambda d: (float(d + 1) / 2) - 1, self.pixel_dimensions))

    def arc_second_dimensions_to_pixel(self, arc_second_dimensions):
        return tuple(map(lambda d: int(d / self.pixel_scale), arc_second_dimensions))

    def pixel_dimensions_to_arc_seconds(self, pixel_dimensions):
        return tuple(map(lambda d: d * self.pixel_scale, pixel_dimensions))

    def x_pixel_to_arc_seconds(self, x_pixel):
        return (x_pixel - self.central_pixels[1]) * self.pixel_scale

    def x_arc_seconds_to_pixel(self, x_arcsec):
        return (x_arcsec) / self.pixel_scale + self.central_pixels[1]

    def y_pixel_to_arc_seconds(self, y_pixel):
        return -(y_pixel - self.central_pixels[0]) * self.pixel_scale

    def y_arc_seconds_to_pixel(self, y_arcsec):
        return -(y_arcsec) / self.pixel_scale + self.central_pixels[0]

    def x_sub_pixel_to_coordinate(self, x_sub_pixel, x_arcsec, sub_grid_size):
        """Convert a coordinate on the regular image-pixel grid_coords to a sub-coordinate, using the pixel scale and sub-grid_coords \
        size"""

        half = self.pixel_scale / 2
        step = self.pixel_scale / (sub_grid_size + 1)

        return x_arcsec - half + (x_sub_pixel + 1) * step

    def y_sub_pixel_to_coordinate(self, y_sub_pixel, y_arcsec, sub_grid_size):
        """Convert a coordinate on the regular image-pixel grid_coords to a sub-coordinate, using the pixel scale and sub-grid_coords \
        size"""

        half = self.pixel_scale / 2
        step = self.pixel_scale / (sub_grid_size + 1)

        return y_arcsec + half - (y_sub_pixel + 1) * step

    def grid_coordinates(self):
        """
        Computes the arc second grids of every pixel on the data-grid_coords.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value and positive y value in arc seconds.
        """

        coordinates_array = np.zeros((self.pixel_dimensions[0], self.pixel_dimensions[1], 2))

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                coordinates_array[y, x, 0] = self.x_pixel_to_arc_seconds(x)
                coordinates_array[y, x, 1] = self.y_pixel_to_arc_seconds(y)

        return coordinates_array


class Data(DataGrid):

    def __init__(self, data, pixel_scale):
        """
        Class storing the data of a 2D pixel grid_coords (e.g. image, PSF, signal_to_noise_ratio)

        Parameters
        ----------
        data : ndarray
            The array of data of the grid_coords.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """

        self.data = data

        super(Data, self).__init__(data.shape, pixel_scale)

    def pad(self, new_dimensions):
        """ Pad the data array with zeros around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be padded to even arrays \
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        new_dimensions : (int, int)
            The (x,y) new pixel dimension of the padded data-array.
        """
        if new_dimensions[0] < self.pixel_dimensions[0]:
            raise ValueError('grids.Grid2d.pad - You have specified a new x_size smaller than the data array')
        elif new_dimensions[1] < self.pixel_dimensions[1]:
            raise ValueError('grids.Grid2d.pad - You have specified a new y_size smaller than the data array')

        x_pad = int((new_dimensions[0] - self.pixel_dimensions[0] + 1) / 2)
        y_pad = int((new_dimensions[1] - self.pixel_dimensions[1] + 1) / 2)

        self.data = np.pad(self.data, ((x_pad, y_pad), (x_pad, y_pad)), 'constant')
        self.pixel_dimensions = self.data.shape
        self.arc_second_dimensions = self.pixel_dimensions_to_arc_seconds(self.pixel_dimensions)

    def trim(self, new_dimensions):
        """ Trim the data array to a new size around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays must be trimmed to even arrays \
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        new_dimensions : (int, int)
            The (x,y) new pixel dimension of the trimmed data-array.
        """
        if new_dimensions[0] > self.pixel_dimensions[0]:
            raise ValueError('grids.Grid2d.trim_data - You have specified a new x_size bigger than the data array')
        elif new_dimensions[1] > self.pixel_dimensions[1]:
            raise ValueError('grids.Grid2d.trim_data - You have specified a new y_size bigger than the data array')

        x_trim = int((self.pixel_dimensions[0] - new_dimensions[0]) / 2)
        y_trim = int((self.pixel_dimensions[1] - new_dimensions[1]) / 2)

        self.data = self.data[x_trim:self.pixel_dimensions[0] - x_trim, y_trim:self.pixel_dimensions[1] - y_trim]
        self.pixel_dimensions = self.data.shape
        self.arc_second_dimensions = self.pixel_dimensions_to_arc_seconds(self.pixel_dimensions)

        if self.pixel_dimensions[0] != new_dimensions[0]:
            logger.debug(
                'image.data.trim_data - Your specified x_size was odd (even) when the image x dimension is even (odd)')
            logger.debug(
                'The method has automatically used x_size+1 to ensure the image is not miscentred by a half-pixel.')
        elif self.pixel_dimensions[1] != new_dimensions[1]:
            logger.debug(
                'image.data.trim_data - Your specified y_size was odd (even) when the image y dimension is even (odd)')
            logger.debug(
                'The method has automatically used y_size+1 to ensure the image is not miscentred by a half-pixel.')


class Image(Data):

    def __init__(self, data, pixel_scale):
        """
        Class storing a 2D image, including its data and coordinate grid_coords.

        Parameters
        ----------
        data : ndarray
            The array of data of the image.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        sky_background_level : float
            The level of sky background in the image.
        sky_background_noise : float
            An estimate of the signal_to_noise_ratio in the sky background.
        """

        super(Image, self).__init__(data, pixel_scale)

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale):
        """
        Loads the image data from a .fits file.

        Parameters
        ----------
        path : str
            The directory path to the fits file.
        filename : str
            The file name of the fits file.
        hdu : int
            The HDU number in the fits file containing the image data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        sky_background_level : float
            An estimate of the level of background sky in the image (electrons per second).
        sky_background_noise : float
            An estimate of the signal_to_noise_ratio level in the background sky (electrons per second).
        """
        data = numpy_array_from_fits(path + filename, hdu)
        return Image(data, pixel_scale)

    def estimate_background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning pixels located at the edge(s) of an image into a histogram and \
        fitting a Gaussian profiles to this histogram. The standard deviation (sigma) of this Gaussian gives a signal_to_noise_ratio \
        estimate.

        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the background signal_to_noise_ratio.

        """

        edges = []

        for edge_no in range(no_edges):
            top_edge = self.data[edge_no, edge_no:self.pixel_dimensions[1] - edge_no]
            bottom_edge = self.data[self.pixel_dimensions[0] - 1 - edge_no,
                          edge_no:self.pixel_dimensions[1] - edge_no]
            left_edge = self.data[edge_no + 1:self.pixel_dimensions[0] - 1 - edge_no, edge_no]
            right_edge = self.data[edge_no + 1:self.pixel_dimensions[0] - 1 - edge_no,
                         self.pixel_dimensions[1] - 1 - edge_no]

            edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

        return norm.fit(edges)[1]

    def circle_mask(self, radius_mask):
        """
        Create a new circular mask for this image

        Parameters
        ----------
        radius_mask : float
            The radius of the mask

        Returns
        -------
        A circular mask for this image
        """
        return Mask.circular(self.arc_second_dimensions, self.pixel_scale, radius_mask=radius_mask)

    def annulus_mask(self, inner_radius_mask, outer_radius_mask):
        """
        Create a new annular mask for this image

        Parameters
        ----------
        inner_radius_mask : float
            The inner radius of the annular mask
        outer_radius_mask : float
            The outer radius of the annular mask

        Returns
        -------
        An annulus mask for this image
        """
        return Mask.annular(self.arc_second_dimensions, self.pixel_scale, inner_radius_mask, outer_radius_mask)

    def unmasked(self):
        """Create a new mask for this image, which is all False and thus completely unmasked"""
        return Mask.unmasked(self.arc_second_dimensions, self.pixel_scale)

    def plot(self):
        pyplot.imshow(self.data)
        pyplot.show()


class Noise(Data):

    def __init__(self, data, pixel_scale):
        """
        Class storing a 2D signal_to_noise_ratio image, including its data and coordinate grid_coords.

        Parameters
        ----------
        data : ndarray
            The array of signal_to_noise_ratio data
        pixel_scale : float
            The arc-second to pixel conversion factor of each pixel.
        """

        super(Noise, self).__init__(data, pixel_scale)

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale):
        """
        Loads the signal_to_noise_ratio data from a .fits file.

        Parameters
        ----------
        path : str
            The directory path to the fits file
        filename : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        data = numpy_array_from_fits(path + filename, hdu)
        return Noise(data, pixel_scale)


class NoiseBackground(Data):

    def __init__(self, data, pixel_scale):
        """
        Class storing the standard deivation of the background signal_to_noise_ratio, or 2D array of the signal_to_noise_ratio estimate in every pixel.

        Parameters
        ----------
        data : float or ndarray
            The background sky map data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """

        super(NoiseBackground, self).__init__(data, pixel_scale)

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale):
        """
        Load the exposure time map data from a .fits file.

        Parameters
        ----------
        path : str
            The directory path to the fits file
        filename : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        renormalize : bool
            Renormalize the PSF such that its value added up to 1.0?
        """
        data = numpy_array_from_fits(path + filename, hdu)
        return NoiseBackground(data, pixel_scale)

    @classmethod
    def from_image_via_edges(cls, image, no_edges):
        background_noise = image.estimate_background_noise_from_edges(no_edges)
        return NoiseBackground(background_noise)

    @classmethod
    def from_one_value(cls, background_noise, pixel_dimensions, pixel_scale):
        data = np.ones(pixel_dimensions) * background_noise
        return NoiseBackground(data, pixel_scale)


class ExposureTime(Data):

    def __init__(self, data, pixel_scale):
        """
        Class storing a 2D exposure time map, including its data and coordinate grid_coords.

        Parameters
        ----------
        data : float or ndarray
            The exposure time map data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """

        super(ExposureTime, self).__init__(data, pixel_scale)

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale):
        """
        Load the exposure time map data from a .fits file.

        Parameters
        ----------
        path : str
            The directory path to the fits file
        filename : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        renormalize : bool
            Renormalize the PSF such that its value added up to 1.0?
        """
        data = numpy_array_from_fits(path + filename, hdu)
        return ExposureTime(data, pixel_scale)

    @classmethod
    def from_one_value(cls, exposure_time, pixel_dimensions, pixel_scale):
        data = np.ones(pixel_dimensions)*exposure_time
        return ExposureTime(data, pixel_scale)


class PSF(Data):

    def __init__(self, data, pixel_scale, renormalize=True):
        """
        Class storing a 2D Point Spread Function (PSF), including its data and coordinate grid_coords.

        Parameters
        ----------
        data : ndarray
            The psf data.
        pixel_scale : float
            The arc-second to pixel conversion factor of each pixel.
        renormalize : bool
            Renormalize the PSF such that its value added up to 1.0?
        """

        super(PSF, self).__init__(data, pixel_scale)

        if renormalize:
            self.renormalize()

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale, renormalize=True):
        """
        Load the PSF data from a .fits file.

        Parameters
        ----------
        path : str
            The directory path to the fits file
        filename : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        renormalize : bool
            Renormalize the PSF such that its value added up to 1.0?
        """
        data = numpy_array_from_fits(path + filename, hdu)
        return PSF(data, pixel_scale, renormalize)

    def convolve_with_image(self, image):
        """
        Convolve a two-dimensional array with a two-dimensional kernel (e.g. a PSF)

        NOTE1 : The PSF kernel must be size odd x odd to avoid ambiguities with convolution offsets.

        NOTE2 : SciPy has multiple 'mode' options for the size of the output array (e.g. does it include zero padding). We \
        require the output array to be the same size as the input image.

        Parameters
        ----------
        image : ndarray
            The image the PSF is convolved with.
        """

        if self.pixel_dimensions[0] % 2 == 0 or self.pixel_dimensions[1] % 2 == 0:
            raise KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(image, self.data, mode='same')

    def renormalize(self):
        """Renormalize the PSF such that its data values sum to unity."""
        return np.divide(self.data, np.sum(self.data))


class Mask(DataGrid):

    # TODO : The mask class methods are a bit messy with how we use DataGrid to make them. Can this be done cleaner?

    def __init__(self, mask, pixel_scale):
        """
        Class stroing a 2D boolean mask, including its coordinate grid_coords.

        Parameters
        ----------
        mask : ndarray
            The boolean array of masked pixels (False = pixel is not masked and included in analysis)
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        self.mask = np.ma.asarray(mask)
        super(Mask, self).__init__(mask.shape, pixel_scale)

    @classmethod
    def circular(cls, arc_second_dimensions, pixel_scale, radius_mask, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified arc second radius.

        Parameters
        ----------
        arc_second_dimensions : (float, float)
            The (x,y) dimensions of the mask in arc seconds.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask : float
            The radius of the circular mask in arc seconds.
        centre: (float, float)
            The centre of the mask in arc seconds.
        """

        grid = DataGrid.from_arcsecond_dimensions(arc_second_dimensions, pixel_scale)

        mask_array = np.zeros((int(grid.pixel_dimensions[0]), int(grid.pixel_dimensions[1])))

        for y in range(int(grid.pixel_dimensions[0])):
            for x in range(int(grid.pixel_dimensions[1])):
                x_arcsec = grid.x_pixel_to_arc_seconds(x) - centre[1]
                y_arcsec = grid.y_pixel_to_arc_seconds(y) - centre[0]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                mask_array[y, x] = radius_arcsec > radius_mask

        return Mask(mask_array, pixel_scale)

    @classmethod
    def annular(cls, arc_second_dimensions, pixel_scale, inner_radius_mask, outer_radius_mask, centre=(0., 0.)):
        """
                Setup the mask as a circle, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        arc_second_dimensions : (float, float)
            The (x,y) dimensions of the mask in arc seconds
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_mask : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius_mask : float
            The outer radius of the annulus mask in arc seconds.
        centre: (float, float)
            The centre of the mask in arc seconds.
        """

        grid = DataGrid.from_arcsecond_dimensions(arc_second_dimensions, pixel_scale)

        mask_array = np.zeros((int(grid.pixel_dimensions[0]), int(grid.pixel_dimensions[1])))

        for y in range(int(grid.pixel_dimensions[0])):
            for x in range(int(grid.pixel_dimensions[1])):
                x_arcsec = grid.x_pixel_to_arc_seconds(x) - centre[1]
                y_arcsec = grid.y_pixel_to_arc_seconds(y) - centre[0]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                mask_array[y, x] = radius_arcsec > outer_radius_mask or radius_arcsec < inner_radius_mask

        return Mask(mask_array, pixel_scale)

    @classmethod
    def unmasked(cls, arc_second_dimensions, pixel_scale):
        """
        Setup the mask such that all values are unmasked, thus corresponding to the entire image.

        Parameters
        ----------
        arc_second_dimensions : (float, float)
            The (x,y) dimensions of the mask in arc seconds
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        grid = DataGrid.from_arcsecond_dimensions(arc_second_dimensions, pixel_scale)
        return Mask(np.ma.make_mask_none(grid.pixel_dimensions), pixel_scale)

    @property
    def pixels_in_mask(self):
        return int(np.size(self.mask) - np.sum(self.mask))

    def compute_grid_coords_image(self):
        """
        Compute the image grid_coords grids from a mask, using the center of every unmasked pixel.
        """
        coordinates = self.grid_coordinates()

        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2))
        pixel_count = 0

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    grid[pixel_count, :] = coordinates[y,x]
                    pixel_count += 1

        return grid

    def compute_grid_coords_image_sub(self, grid_size_sub):
        """ Compute the image sub-grid_coords grids from a mask, using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image sub-grid_coords is computed for and the image's data grid_coords.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) of the sub-grid_coords of each image pixel.
        """

        image_pixels = self.pixels_in_mask
        image_pixel_count = 0

        grid = np.zeros(shape=(image_pixels, grid_size_sub ** 2, 2))

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:

                    x_arcsec = self.x_pixel_to_arc_seconds(x)
                    y_arcsec = self.y_pixel_to_arc_seconds(y)
                    sub_pixel_count = 0

                    for y1 in range(grid_size_sub):
                        for x1 in range(grid_size_sub):
                            grid[image_pixel_count, sub_pixel_count, 0] = \
                                self.x_sub_pixel_to_coordinate(x1, x_arcsec, grid_size_sub)

                            grid[image_pixel_count, sub_pixel_count, 1] = \
                                self.y_sub_pixel_to_coordinate(y1, y_arcsec, grid_size_sub)

                            sub_pixel_count += 1

                    image_pixel_count += 1

        return grid

    def compute_grid_coords_blurring(self, psf_size):
        """ Compute the blurring grid_coords grids from a mask, using the center of every unmasked pixel.

        The blurring grid_coords contains all pixels which are not in the mask, but close enough to it that a fraction of \
        their will be blurred into the mask region (and therefore they are needed for the analysis). They are located \
        by scanning for all pixels which are outside the mask but within the psf size.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid_coords is computed for and the image's data grid_coords.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """

        blurring_mask = self.compute_blurring_mask(psf_size)

        return blurring_mask.compute_grid_coords_image()

    def compute_grid_data(self, data):
        """Compute a data grid, which represents the data values of a data-set (e.g. an image, noise, in the mask.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid_coords is computed for and the image's data grid_coords.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels))
        pixel_count = 0

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    grid[pixel_count] = data[y, x]
                    pixel_count += 1

        return grid

    def compute_grid_mapper_data_to_2d(self):
        """
        Compute the mapping of every pixel in the mask to its 2D pixel coordinates.
        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2), dtype='int')
        pixel_count = 0

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    grid[pixel_count, :] = y,x
                    pixel_count += 1

        return grid

    def compute_grid_mapper_sparse(self, sparse_grid_size):
        """Given an image.Mask, compute the sparse cluster image pixels, defined as the sub-set of image-pixels used \
        to perform KMeans clustering (this is used purely for speeding up the KMeans clustering algorithim).

        This sparse grid_coords is a uniform subsample of the masked image and is computed by only including image pixels \
        which, when divided by the sparse_grid_size, do not give a remainder.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask we are finding the sparse clustering pixels of and the image pixel_dimensions / pixel scale.
        sparse_grid_size : int
            The spacing of the sparse image pixel grid_coords (e.g. a value of 2 will compute a sparse grid_coords of pixels which \
            are two pixels apart)

        Returns
        -------
        clustering_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the mask.
        image_to_clustering : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D \
            index of the sparse pixel in sparse_pixel arrays.
        """

        sparse_mask = self.compute_sparse_uniform_mask(sparse_grid_size)
        sparse_index_image = self.compute_sparse_index_image(sparse_mask)
        sparse_to_image = self.compute_sparse_to_image(sparse_mask)
        image_to_sparse = self.compute_image_to_sparse(sparse_mask, sparse_index_image)

        return sparse_to_image, image_to_sparse

    def compute_grid_border(self):
        """Compute the border image pixels from a mask, where a border pixel is a pixel inside the mask but on its \
        edge, therefore neighboring a pixel with a *True* value.
        """

        # TODO : Border pixels for a circular mask and annulus mask are different (the inner annulus pixels should be \
        # TODO : ignored. Should we turn this to classes for Masks?

        border_pixels = np.empty(0)
        image_pixel_index = 0

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    if self.mask[y + 1, x] == 1 or self.mask[y - 1, x] == 1 or self.mask[y, x + 1] == 1 or \
                            self.mask[y, x - 1] == 1 or self.mask[y + 1, x + 1] == 1 or self.mask[y + 1, x - 1] == 1 \
                            or self.mask[y - 1, x + 1] == 1 or self.mask[y - 1, x - 1] == 1:
                        border_pixels = np.append(border_pixels, image_pixel_index)

                    image_pixel_index += 1

        return border_pixels

    def compute_blurring_mask(self, psf_size):
        """Compute the blurring mask, which represents all pixels not in the mask but close enough to it that a \
        fraction of their light will be blurring in the image.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid_coords is computed for and the image's data grid_coords.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """

        blurring_mask = np.ones(self.pixel_dimensions)

        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    for y1 in range((-psf_size[1] + 1) // 2, (psf_size[1] + 1) // 2):
                        for x1 in range((-psf_size[0] + 1) // 2, (psf_size[0] + 1) // 2):
                            if 0 <= y + y1 <= self.pixel_dimensions[0] - 1 \
                                    and 0 <= x + x1 <= self.pixel_dimensions[1] - 1:
                                if self.mask[y + y1, x + x1] == True:
                                    blurring_mask[y + y1, x + x1] = False
                            else:
                                raise MaskException(
                                    "setup_blurring_mask extends beynod the size of the mask - pad the image"
                                    "before masking")

        return Mask(blurring_mask, self.pixel_scale)
    
    def compute_sparse_uniform_mask(self, sparse_grid_size):
        """Setup a two-dimensional sparse mask of image pixels, by keeping all image pixels which do not give a remainder \
        when divided by the sub-grid_coords size. """
        sparse_mask = np.ones(self.pixel_dimensions)
    
        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                        sparse_mask[y, x] = False
    
        return Mask(sparse_mask, self.pixel_scale)
    
    def compute_sparse_index_image(self, sparse_mask):
        """Setup an image which, for each *False* entry in the sparse mask, puts the sparse pixel index in that pixel.
    
         This is used for computing the image_to_clustering vector, whereby each image pixel is paired to the sparse pixel \
         in this image via a neighbor search."""
    
        sparse_index_2d = np.zeros(self.pixel_dimensions)
        sparse_pixel_index = 0
    
        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if sparse_mask.mask[y,x] == False:
                    sparse_pixel_index += 1
                    sparse_index_2d[y,x] = sparse_pixel_index
    
        return sparse_index_2d
    
    def compute_sparse_to_image(self, sparse_mask):
        """Compute the mapping of each sparse image pixel to its closest image pixel, defined using a mask of image \
        pixels.
    
        Parameters
        ----------
        mask : imaging.Mask
            The image mask we are finding the sparse clustering pixels of and the image pixel_dimensions / pixel scale.
        sparse_mask : ndarray
            The two-dimensional boolean image of the sparse grid_coords.
    
        Returns
        -------
        clustering_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the self.
        """
        sparse_to_image = np.empty(0)
        image_pixel_index = 0
    
        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
    
                if sparse_mask.mask[y, x] == False:
                    sparse_to_image = np.append(sparse_to_image, image_pixel_index)
    
                if self.mask[y, x] == False:
                    image_pixel_index += 1
    
        return sparse_to_image
    
    def compute_image_to_sparse(self, sparse_mask, sparse_index_image):
        """Compute the mapping between every image pixel in the mask and its closest sparse clustering pixel.
    
        This is performed by going to each image pixel in the *mask*, and pairing it with its nearest neighboring pixel \
        in the *sparse_mask*. The index of the *sparse_mask* pixel is drawn from the *sparse_index_image*. This \
        neighbor search continue grows larger and larger around a pixel, until a pixel contained in the *sparse_mask* is \
        successfully found.
    
        Parameters
        ----------
        mask : imaging.Mask
            The image mask we are finding the sparse clustering pixels of and the image pixel_dimensions / pixel scale.
        sparse_mask : ndarray
            The two-dimensional boolean image of the sparse grid_coords.
    
        Returns
        -------
        image_to_clustering : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D \
            index of the sparse pixel in sparse_pixel arrays.
    
        """
        image_to_sparse = np.empty(0)
    
        for y in range(self.pixel_dimensions[0]):
            for x in range(self.pixel_dimensions[1]):
                if self.mask[y, x] == False:
                    iboarder = 0
                    pixel_match = False
                    while pixel_match == False:
                        for y1 in range(y-iboarder, y+iboarder+1):
                            for x1 in range(x-iboarder, x+iboarder+1):
                                if y1 >= 0 and y1 < self.pixel_dimensions[0] and x1 >= 0 and x1 < self.pixel_dimensions[1]:
                                    if sparse_mask.mask[y1, x1] == False and pixel_match == False:
                                        image_to_sparse = np.append(image_to_sparse, sparse_index_image[y1,x1]-1)
                                        pixel_match = True
    
                        iboarder += 1
                        if iboarder == 100:
                            raise MaskException('compute_image_to_sparse - Stuck in infinite loop')
    
        return image_to_sparse


class MaskException(Exception):
    pass


class KernelException(Exception):
    pass