from auto_lens.imaging import data
import numpy as np
from scipy.stats import norm
import scipy.signal


class Noise(data.DataGrid):
    pass


class ExposureTime(data.DataGrid):
    pass


class Image(data.DataGrid):
    def __init__(self, array, pixel_scale, psf=None, background_noise=None, poisson_noise=None,
                 effective_exposure_time=None):
        super(Image, self).__init__(array, pixel_scale)
        self.psf = psf
        self.background_noise = background_noise
        self.poisson_noise = poisson_noise
        self.effective_exposure_time = effective_exposure_time

    def background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning data_to_pixels located at the edge(s) of an image
        into a histogram and fitting a Gaussian profiles to this histogram. The standard deviation (sigma) of this
        Gaussian gives a signal_to_noise_ratio estimate.

        Parameters
        ----------
        no_edges : int
            Number of edges used to estimate the background signal_to_noise_ratio.

        """

        edges = []

        for edge_no in range(no_edges):
            top_edge = self[edge_no, edge_no:self.shape[1] - edge_no]
            bottom_edge = self[self.shape[0] - 1 - edge_no, edge_no:self.shape[1] - edge_no]
            left_edge = self[edge_no + 1:self.shape[0] - 1 - edge_no, edge_no]
            right_edge = self[edge_no + 1:self.shape[0] - 1 - edge_no, self.shape[1] - 1 - edge_no]

            edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

        return norm.fit(edges)[1]


class NoiseBackground(data.DataGrid):
    @classmethod
    def from_image_via_edges(cls, image, no_edges):
        background_noise = image.estimate_background_noise_from_edges(no_edges)
        return NoiseBackground(background_noise, image.pixel_scale)


class PSF(data.DataGrid):

    def __init__(self, array, pixel_scale, renormalize=True):
        """
        Class storing a 2D Point Spread Function (PSF), including its data and coordinate grid_coords.

        Parameters
        ----------
        array : ndarray
            The psf data.
        pixel_scale : float
            The arc-second to pixel conversion factor of each pixel.
        renormalize : bool
            Renormalize the PSF such that its value added up to 1.0?
        """

        super(PSF, self).__init__(array, pixel_scale)

        if renormalize:
            self.renormalize()

    @classmethod
    def from_fits_renormalized(cls, file_path, hdu, pixel_scale):
        psf = PSF.from_fits(file_path, hdu, pixel_scale)
        psf.renormalize()
        return psf

    def convolve_with_image(self, image):
        """
        Convolve a two-dimensional array with a two-dimensional kernel (e.g. a PSF)

        NOTE1 : The PSF kernel must be size odd x odd to avoid ambiguities with convolution offsets.

        NOTE2 : SciPy has multiple 'mode' options for the size of the output array (e.g. does it include zero padding).
        We require the output array to be the same size as the input image.

        Parameters
        ----------
        image : ndarray
            The image the PSF is convolved with.
        """

        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(image, self.data, mode='same')

    def renormalize(self):
        """Renormalize the PSF such that its data values sum to unity."""
        return np.divide(self.data, np.sum(self.data))


class KernelException(Exception):
    pass
