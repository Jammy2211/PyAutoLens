from auto_lens.tools import image_tools
from scipy.stats import norm
import os

import numpy as np


# TODO: The default data path is determined here to save you always having to pass it in
data_path = "{}/../../data/prep_lens/".format(os.path.dirname(os.path.realpath(__file__)))


class Image(object):

    def __init__(self, filename, hdu, pixel_scale, sky_background_level=None, sky_background_noise=None, path=data_path):

        self.image2d, self.xy_dim = image_tools.load_fits(path, filename, hdu)  # Load image from .fits file
        self.pixel_scale = pixel_scale  # Set its pixel scale using the input value
        self.xy_arcsec = list(map(lambda l: l * pixel_scale, self.xy_dim))  # Convert image dimensions to arcseconds
        self.sky_background_level = sky_background_level
        self.sky_background_noise = sky_background_noise

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
            top_edge = self.image2d[edge_no, edge_no:ydim - edge_no]
            bottom_edge = self.image2d[xdim - 1 - edge_no, edge_no:ydim - edge_no]
            left_edge = self.image2d[edge_no + 1:xdim - 1 - edge_no, edge_no]
            right_edge = self.image2d[edge_no + 1:xdim - 1 - edge_no, ydim - 1 - edge_no]

            edges = np.concatenate(((edges, top_edge, bottom_edge, right_edge, left_edge)))

        self.sky_background_level, self.sky_background_noise = norm.fit(edges)

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


class PSF(object):

    def __init__(self):
        pass

    def load_fits(self, dir, file, hdu, pixel_scale):
        self.psf2d, self.xy_dim = image_tools.load_fits(dir, file, hdu)  # Load image from .fits file
        self.pixel_scale = pixel_scale  # Set its pixel scale using the input value
        self.xy_arcsec = list(map(lambda l: l * pixel_scale, self.xy_dim))  # Convert image dimensions to arcseconds


class Mask(object):
    """Abstract Class for preparing and storing the image mask used for the AutoLens analysis"""

    def __init__(self, dimensions, pixel_scale):
        """

        Parameters
        ----------
        dimensions : int
            The dimensions of the image (x, y)
        pixel_scale :
            The scale size of a pixel (x, y) in arc seconds
        """
        # Calculate the central pixel of the mask. This is a half pixel value for an even sized array.
        # Also minus one from value so that mask2d is shifted to python array (i.e. starts at 0)
        self.pixel_scale = pixel_scale
        self.central_pixel = list(map(lambda l: (float(l + 1) / 2) - 1, dimensions))
        self.array = np.zeros((dimensions[0], dimensions[1]))


class CircleMask(Mask):
    """Class for preparing and storing a circular image mask used for the AutoLens analysis"""

    def __init__(self, dimensions, pixel_scale, radius):
        """

        Parameters
        ----------
        dimensions : int
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

                x_pix = i - self.central_pixel[0] # Shift x coordinate using central x pixel
                y_pix = j - self.central_pixel[1] # Shift u coordinate using central y pixel

                radius_arc = pixel_scale * np.sqrt((x_pix)**2 + (y_pix)**2)

                if radius_arc <= radius:
                    self.array[i, j] = int(1)

class AnnulusMask(Mask):
    """Class for preparing and storing an annulus image mask used for the AutoLens analysis"""

    def __init__(self, dimensions, pixel_scale, inner_radius, outer_radius):
        """

        Parameters
        ----------
        dimensions : int
            The dimensions of the image (x, y)
        pixel_scale : float
            The scale size of a pixel (x, y) in arc seconds
        inner_radius : float
            The inner radius of the circlular annulus (arc seconds)

        outer_radius : float
            The outer radius of the circlular annulus (arc seconds)
        """
        super(AnnulusMask, self).__init__(dimensions, pixel_scale)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        for i in range(dimensions[0]):
            for j in range(dimensions[1]):

                x_pix = i - self.central_pixel[0] # Shift x coordinate using central x pixel
                y_pix = j - self.central_pixel[1] # Shift u coordinate using central y pixel

                radius_arc = pixel_scale * np.sqrt((x_pix)**2 + (y_pix)**2)

                if radius_arc <= outer_radius and radius_arc >= inner_radius:
                    self.array[i, j] = int(1)