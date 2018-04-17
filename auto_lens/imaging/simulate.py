import numpy as np


class SimulateImage(object):

    def __init__(self, data, pixel_scale, psf):
        """
        Creates a new simulated image.

        Parameters
        ----------
        ndarray : array
            The image of the lensed to be simulated.
        pixel_scale: float
            The scale of an image pixel.
        psf : imaging.PSF
            The image of the simulated image.
        """
        self.image = data
        self.image_original = data
        self.pixel_scale = pixel_scale
        self.psf = psf

        self.simulate_optics()

    def simulate_optics(self):
        """
        Setup the PSF of a simulated image and blur with the original exposure

        Parameters
        ----------
        ndarray : array
            The image of the lensed to be simulated.
        pixel_scale: float
            The scale of an image pixel.
        psf : imaging.PSF
            The image of the simulated image.
        """
        self.image =