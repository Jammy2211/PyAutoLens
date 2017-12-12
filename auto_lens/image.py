from scipy.stats import norm
from astropy.io import fits
import os

import numpy as np

data_path = "{}/../../data/prep_lens/".format(os.path.dirname(os.path.realpath(__file__)))


class Data(object):
    """Abstract Base Class for all classes which store a two-dimensional data array, e.g. the image, PSF, Nosie etc."""

    def __init__(self, data, pixel_scale):
        """Setup an Image class, which holds the image of a strong lens to be modeled.

        Parameters
        ----------
        image : ndarray
            Two-dimensional array of the data (e.g. the image, PSF, noise).
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds.
        """
        self.data = data
        self.pixel_scale = pixel_scale  # Set its pixel scale using the input value
        self.xy_dim = self.data.shape[:]  # x dimension (pixels)
        self.xy_cen_pixel = tuple(map(lambda l: (l / 2.0)-0.5, self.xy_dim))
        self.xy_arcsec = tuple(map(lambda l: l * pixel_scale, self.xy_dim))  # Convert image dimensions to arcseconds

    def trim_data(self, x_size, y_size):
        """ Trim the data array to a new size around its central pixel.

        NOTE: The centre of the array currently cannot be shifted. Therefore, even arrays are trimmed to even arrays
        (e.g. 8x8 -> 4x4) and odd to odd (e.g. 5x5 -> 3x3). Centre offsets may be considered at a later date.

        Parameters
        ----------
        x_size : int
            The new x dimension of the data-array
        y_size : int
            The new y dimension of the data-array
        """
        pass
      #  xy_central_pixel = self.xy_dim[:] / 2

     #   self.data = self.data[]

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
    def via_fits(cls, file_name, hdu, pixel_scale, sky_background_level=None, sky_background_noise=None, path=data_path):
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
        hdu_list = fits.open(path + file_name)  # Open the fits file
        data_2d = np.array(hdu_list[hdu].data)
        return Image(data_2d, pixel_scale, sky_background_level, sky_background_noise)

    def set_sky_via_edges(self, no_edges):
        """Estimate the background sky level and noise by binning pixels located at the edge(s) of an image into a
        histogram and fitting a Gaussian profile to this histogram. The mean (mu) of this Gaussian gives the background
        sky level, whereas the FWHM (sigma) gives the noise estimate.

        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the backgroundd sky properties

        """

        xdim = self.xy_dim[0]
        ydim = self.xy_dim[1]

        edges = []

        for edge_no in range(no_edges):
            top_edge = self.data[edge_no, edge_no:ydim - edge_no]
            bottom_edge = self.data[xdim - 1 - edge_no, edge_no:ydim - edge_no]
            left_edge = self.data[edge_no + 1:xdim - 1 - edge_no, edge_no]
            right_edge = self.data[edge_no + 1:xdim - 1 - edge_no, ydim - 1 - edge_no]

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
        return PSF.via_fits(file_name=file_name, hdu=hdu, pixel_scale=self.pixel_scale, path=path)

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
        return CircleMask(dimensions=self.xy_dim, pixel_scale=self.pixel_scale, radius=radius_arc)

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
        return AnnulusMask(dimensions=self.xy_dim, pixel_scale=self.pixel_scale, outer_radius=outer_radius_arc,
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
    def via_fits(cls, file_name, hdu, pixel_scale, path=data_path):
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
        hdu_list = fits.open(path + file_name)  # Open the fits file
        data_2d = np.array(hdu_list[hdu].data)
        return PSF(data_2d, pixel_scale)


# TODO : I've defined a mask so that True means we keep the pixel, False means we don't. This means we can use compress
# TODO : To remove everything outside the mask. opiinon?
# TODO : Just to confuse coordinates furhter, we need to decide how we choose the centre of an image and mask. Currently,
# TODO : masks are automatically centred on the central pixel.
class Mask(object):
    """Abstract Class for preparing and storing the image mask used for the AutoLens analysis"""

    def __init__(self, dimensions, pixel_scale):
        """
        Setup the boolean mask, where True means a pixel is included in the analysis and False means its excluded.

        Parameters
        ----------
        dimensions : (int, int)
            The dimensions of the image (x, y)
        pixel_scale :
            The scale size of a pixel (x, y) in arc seconds
        """

        self.pixel_scale = pixel_scale
        self.central_pixel = list(map(lambda l: (float(l + 1) / 2) - 1, dimensions))
        self.array = np.zeros((dimensions[0], dimensions[1]))

class CircleMask(Mask):
    """Class for preparing and storing a circular image mask used for the AutoLens analysis"""

    def __init__(self, dimensions, pixel_scale, radius):
        """

        Parameters
        ----------
        dimensions : (int, int)
            The dimensions of the image (x, y)
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        radius : float
            The radius of the circle (arc seconds)
        """
        super(CircleMask, self).__init__(dimensions, pixel_scale)
        self.radius = radius

        for i in range(dimensions[0]):
            for j in range(dimensions[1]):

                x_pix = i - self.central_pixel[0]  # Shift x coordinate using central x pixel
                y_pix = j - self.central_pixel[1]  # Shift u coordinate using central y pixel

                radius_arc = pixel_scale * np.sqrt(x_pix ** 2 + y_pix ** 2)

                if radius_arc <= radius:
                    self.array[i, j] = True

class AnnulusMask(Mask):
    """Class for preparing and storing an annulus image mask used for the AutoLens analysis"""

    def __init__(self, dimensions, pixel_scale, inner_radius, outer_radius):
        """

        Parameters
        ----------
        dimensions : (int, int)
            The dimensions of the image (x, y)
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        inner_radius : float
            The inner radius of the circular annulus (arc seconds)

        outer_radius : float
            The outer radius of the circular annulus (arc seconds)
        """
        super(AnnulusMask, self).__init__(dimensions, pixel_scale)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        for i in range(dimensions[0]):
            for j in range(dimensions[1]):

                x_pix = i - self.central_pixel[0]  # Shift x coordinate using central x pixel
                y_pix = j - self.central_pixel[1]  # Shift u coordinate using central y pixel

                radius_arc = pixel_scale * np.sqrt(x_pix ** 2 + y_pix ** 2)

                if outer_radius >= radius_arc >= inner_radius:
                    self.array[i, j] = int(1)
