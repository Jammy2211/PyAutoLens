from tools import image_tools
from scipy.stats import norm

import numpy as np


class Image(object):
    def __init__(self):
        pass

    # TODO: Some of these functions might be doing what the __init__ should be doing
    def load_fits(self, dir, file, hdu, pixel_scale):
        self.image2d, self.xy_dim = image_tools.load_fits(dir, file, hdu)  # Load image from .fits file
        self.pixel_scale = pixel_scale  # Set its pixel scale using the input value
        self.xy_arcsec = list(map(lambda l: l * pixel_scale, self.xy_dim))  # Convert image dimensions to arcseconds

    def set_sky(self, sky_background_level, sky_background_noise):
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


class PSF(object):
    def __init__(self):
        pass

    def load_fits(self, dir, file, hdu, pixel_scale):
        self.psf2d, self.xy_dim = image_tools.load_fits(dir, file, hdu)  # Load image from .fits file
        self.pixel_scale = pixel_scale  # Set its pixel scale using the input value
        self.xy_arcsec = list(map(lambda l: l * pixel_scale, self.xy_dim))  # Convert image dimensions to arcseconds


class Mask(object):
    "Class for preparing and storing the image mask used for the AutoLens analysis"

    def __init__(self):
        pass

    def set_circle(self, image, mask_radius_arcsec):
        """Setup the image mask, using a circular mask given the image dimensions (pixels), pixel to arcsecond scale
         and mask radius (arcseconds).

        Parameters
        ----------
        xy_dim : list(int)
            x and y pixel dimensions of image (xy_dim[0] = x dimension, xy_dim[1] = y dimension)
        pixel_scale : float
            Size of each pixel in arcseconds
        mask_radius_arcsec : float
            Circular radius of mask to be generated in arcseconds.
        """

        # Calculate the central pixel of the mask. This is a half pixel value for an even sized array.
        # Also minus one from value so that mask2d is shifted to python array (i.e. starts at 0)
        xy_cen_pix = list(map(lambda l: ((l + 1) / 2) - 1, image.xy_dim))

        self.mask2d = np.zeros((image.xy_dim[0], image.xy_dim[1]))

        for i in range(image.xy_dim[0]):
            for j in range(image.xy_dim[1]):

                r_arcsec = image.pixel_scale * np.sqrt((i - xy_cen_pix[0]) ** 2 + (j - xy_cen_pix[1]) ** 2)

                if r_arcsec <= mask_radius_arcsec:
                    self.mask2d[i, j] = int(1)
