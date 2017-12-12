from scipy.stats import norm
from astropy.io import fits
import os
from functools import wraps

import numpy as np

data_path = "{}/../../data/prep_lens/".format(os.path.dirname(os.path.realpath(__file__)))


def numpy_array_from_fits(file_path, hdu):
    hdu_list = fits.open(file_path)  # Open the fits file
    return np.array(hdu_list[hdu].data)


class Data(object):
    """Abstract Base Class for all classes which store a two-dimensional data array, e.g. the image, PSF, Nosie etc."""

    def __init__(self, data, pixel_scale):
        """Setup an Image class, which holds the image of a strong lens to be modeled.

        Parameters
        ----------
        data : ndarray
            Two-dimensional array of the data (e.g. the image, PSF, noise).
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        """
        self.data = data
        self.pixel_scale = pixel_scale  # Set its pixel scale using the input value
        self.update_dimensions()

    def update_dimensions(self):
        """Update the image dimensions and central pixel"""
        self.dimensions = self.data.shape[:]  # x dimension (pixels)
        self.central_pixels = self.central_pixel(self.dimensions)
        self.dimensions_arc_seconds = self.dimensions_to_arc_seconds(self.dimensions, self.pixel_scale)

    @staticmethod
    def central_pixel(dimensions):
        return tuple(map(lambda l: (float(l + 1) / 2) - 1, dimensions))

    @staticmethod
    def dimensions_to_arc_seconds(dimensions, pixel_scale):
        return tuple(map(lambda l: l * pixel_scale, dimensions))

    @property
    def x_dimension(self):
        return self.dimensions[0]

    @property
    def y_dimension(self):
        return self.dimensions[1]

    @property
    def x_cen_pixel(self):
        return self.central_pixels[0]

    @property
    def y_cen_pixel(self):
        return self.central_pixels[1]

    def trim_data(self, x_size, y_size):
        """ Trim the data array to a new size around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays are trimmed to even arrays
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        x_size : int
            The new x dimension of the data-array
        y_size : int
            The new y dimension of the data-array
        """
        if x_size >  self.x_dimension:
            raise ValueError ('image.Data.trim_data - You have specified a new x_size bigger than the data array')
        elif y_size >  self.y_dimension:
            raise ValueError ('image.Data.trim_data - You have specified a new y_size bigger than the data array')

        x_trim = int((self.x_dimension - x_size)/2)
        y_trim = int((self.y_dimension - y_size)/2)

        self.data = self.data[x_trim:self.x_dimension-x_trim, y_trim:self.y_dimension-y_trim]

        self.update_dimensions()

        if self.x_dimension != x_size:
            print ('image.data.trim_data - Your specified x_size was odd (even) when the image x dimension is even (odd)')
            print('The method has automatically used x_size+1 to ensure the image is not miscentred by a half-pixel.')
        elif self.y_dimension != y_size:
            print ('image.data.trim_data - Your specified y_size was odd (even) when the image y dimension is even (odd)')
            print('The method has automatically used y_size+1 to ensure the image is not miscentred by a half-pixel.')

    def pad_data(self, x_size, y_size):
        """ Pad the data array with zeros around its central pixel.

        NOTE: The centre of the array cannot be shifted. Therefore, even arrays are padded to even arrays
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3).

        Parameters
        ----------
        x_size : int
            The new x dimension of the data-array
        y_size : int
            The new y dimension of the data-array
        """
        if x_size <  self.x_dimension:
            raise ValueError ('image.Data.pad_data - You have specified a new x_size smaller than the data array')
        elif y_size <  self.y_dimension:
            raise ValueError ('image.Data.pad_data - You have specified a new y_size smaller than the data array')

        x_pad = int((x_size- self.x_dimension + 1)/2)
        y_pad = int((y_size- self.y_dimension + 1)/2)

        self.data = np.pad(self.data, ((x_pad, y_pad), (x_pad, y_pad)), 'constant')

        self.update_dimensions()

class Image(Data):

    def __init__(self, image, pixel_scale, sky_background_level=None, sky_background_noise=None):
        """Setup an Image class, which holds the image of a strong lens to be modeled.

        Parameters
        ----------
        image : ndarray
            Two-dimensional array of the imaging data (electrons per second).
            This can be loaded from a fits file using the via_fits method.
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        sky_background_level : float
            An estimate of the level of background sky in the image (electrons per second).
        sky_background_noise : float
            An estimate of the noise level in the background sky (electrons per second).
        """
        super(Image, self).__init__(image, pixel_scale)

        self.sky_background_level = sky_background_level
        self.sky_background_noise = sky_background_noise

    @classmethod
    def from_fits(cls, file_name, hdu, pixel_scale, sky_background_level=None, sky_background_noise=None,
                  path=data_path):
        """Load the image from a fits file.

        Parameters
        ----------
        file_name : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        sky_background_level : float
            An estimate of the level of background sky in the image (electrons per second).
        sky_background_noise : float
            An estimate of the noise level in the background sky (electrons per second).
        path : str
            The directory path to the fits file
        """
        array = numpy_array_from_fits(path + file_name, hdu)
        return Image(array, pixel_scale, sky_background_level,
                     sky_background_noise)

    def set_sky_via_edges(self, no_edges):
        """Estimate the background sky level and noise by binning pixels located at the edge(s) of an image into a
        histogram and fitting a Gaussian profile to this histogram. The mean (mu) of this Gaussian gives the background
        sky level, whereas the FWHM (sigma) gives the noise estimate.

        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the backgroundd sky properties

        """

        edges = []

        for edge_no in range(no_edges):
            top_edge = self.data[edge_no, edge_no:self.y_dimension - edge_no]
            bottom_edge = self.data[self.x_dimension - 1 - edge_no, edge_no:self.y_dimension - edge_no]
            left_edge = self.data[edge_no + 1:self.x_dimension - 1 - edge_no, edge_no]
            right_edge = self.data[edge_no + 1:self.x_dimension - 1 - edge_no, self.y_dimension - 1 - edge_no]

            edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

        self.sky_background_level, self.sky_background_noise = norm.fit(edges)

    def load_psf(self, file_name, hdu, path):
        """Load the PSF for this image

        Parameters
        ----------
        file_name : str
            The PSF file_name to be loaded from
        hdu : int
            The PSF HDU in the fits file
        path : str
            The path to the PSF image file

        """
        return PSF.from_fits(file_name=file_name, hdu=hdu, pixel_scale=self.pixel_scale, path=path)

    def circle_mask(self, radius_arc):
        """
        Create a new circular mask for this image

        Parameters
        ----------
        radius_arc : float
            The radius of the mask

        Returns
        -------
        A circular mask for this image
        """
        return Mask.circular(dimensions=self.dimensions, pixel_scale=self.pixel_scale, radius=radius_arc)

    def annulus_mask(self, inner_radius_arc, outer_radius_arc):
        """
        Create a new annular mask for this image

        Parameters
        ----------
        inner_radius_arc : float
            The inner radius of the annular mask
        outer_radius_arc : float
            The outer radius of the annular mask

        Returns
        -------
        An annulus mask for this image
        """
        return Mask.annular(dimensions=self.dimensions, pixel_scale=self.pixel_scale,
                            outer_radius=outer_radius_arc,
                            inner_radius=inner_radius_arc)


class PSF(Data):

    def __init__(self, psf, pixel_scale):
        """Setup a PSF class, which holds the PSF of an image of a strong lens.

        Parameters
        ----------
        psf : ndarray
            Two-dimensional array of the PSF (Automatically normalized to unit normalization).
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        """
        super(PSF, self).__init__(psf, pixel_scale)

    @classmethod
    def from_fits(cls, file_name, hdu, pixel_scale, path=data_path):
        """Load the image from a fits file.

        Parameters
        ----------
        file_name : str
            The file name of the fits file
        hdu : int
            The HDU number in the fits file containing the data
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        path : str
            The directory path to the fits file
        """
        array = numpy_array_from_fits(path + file_name, hdu)
        return PSF(array, pixel_scale)


def as_mask(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return np.ma.make_mask(func(*args, **kwargs))

    return wrapper


# TODO: I haven't yet tested the central coordinates and if these objects are to match those in the profile package then
# TODO: it may also make sense to implement the same pixel scale paradigm.
class Mask(object):
    """Abstract Class for preparing and storing the image mask used for the AutoLens analysis"""

    @staticmethod
    def central_pixel(dimensions):
        return list(map(lambda l: (float(l + 1) / 2) - 1, dimensions))

    @classmethod
    def mask(cls, dimensions):
        """

        Parameters
        ----------
        dimensions: (float, float)
            The spatial dimensions of the mask

        Returns
        -------
            An empty array
        """
        # TODO: this line (with other modifications) would bring the mask dimensions convention into line with the
        # TODO: profile classes
        # return np.zeros((int(dimensions[0] / pixel_scale), int(dimensions[1] / pixel_scale)))
        return np.zeros((dimensions[0], dimensions[1]))

    @classmethod
    @as_mask
    def circular(cls, dimensions, pixel_scale, radius, centre=(0., 0.)):
        """

        Parameters
        ----------
        centre
        dimensions : (int, int)
            The dimensions of the image (x, y)
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        radius : float
            The radius of the circle (arc seconds)
        """
        array = Mask.mask(dimensions)
        central_pixel = Mask.central_pixel(dimensions)
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):

                x_pix = i - central_pixel[0]  # Shift x coordinate using central x pixel
                y_pix = j - central_pixel[1]  # Shift u coordinate using central y pixel

                radius_arc = pixel_scale * np.sqrt((x_pix - centre[0]) ** 2 + (y_pix - centre[1]) ** 2)

                if radius_arc <= radius:
                    array[i, j] = True
        return array

    @classmethod
    @as_mask
    def annular(cls, dimensions, pixel_scale, inner_radius, outer_radius, centre=(0., 0.)):
        """

        Parameters
        ----------
        centre
        dimensions : (int, int)
            The dimensions of the image (x, y)
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        inner_radius : float
            The inner radius of the circular annulus (arc seconds
        outer_radius : float
            The outer radius of the circular annulus (arc seconds)
        """
        array = Mask.mask(dimensions)
        central_pixel = Mask.central_pixel(dimensions)
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):

                x_pix = i - central_pixel[0]  # Shift x coordinate using central x pixel
                y_pix = j - central_pixel[1]  # Shift u coordinate using central y pixel

                radius_arc = pixel_scale * np.sqrt((x_pix - centre[0]) ** 2 + (y_pix - centre[1]) ** 2)

                if outer_radius >= radius_arc >= inner_radius:
                    array[i, j] = True
        return array
