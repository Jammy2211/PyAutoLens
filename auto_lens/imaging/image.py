from auto_lens.imaging.data import Array
import numpy as np
from scipy.stats import norm
import scipy.signal


class Image(Array):
    def __init__(self, array, effective_exposure_time=1, pixel_scale=1, psf=None, background_noise=None,
                 poisson_noise=None):
        super(Image, self).__init__(array, pixel_scale)
        self.psf = psf
        self.background_noise = background_noise
        self.poisson_noise = poisson_noise
        self.effective_exposure_time = effective_exposure_time

    @classmethod
    def simulate(cls, array, effective_exposure_time=1, pixel_scale=1, background_sky_map=None,
                 psf=None, include_poisson_noise=False, seed=-1):

        if background_sky_map is not None:
            array += background_sky_map
            array_counts = np.multiply(background_sky_map, effective_exposure_time)
            background_noise = np.sqrt(array_counts)
            background_noise = np.divide(background_noise, effective_exposure_time)
        else:
            background_noise = None

        if psf is not None:
            array = psf.convolve(array)

        # TODO : The poisson noise map must be generated for the psf blurred image, which is was not before.
        # TODO : Should be using the @properties but not sure how to get it to call in the simulate classmethod

        if include_poisson_noise is True:
            array += generate_poisson_noise(array, effective_exposure_time, seed)
            # The poisson noise map does not include the background sky, so this estimate below removes it
            if background_sky_map is not None:
                array_counts = np.multiply(array - background_sky_map, effective_exposure_time)
            elif background_sky_map is None:
                array_counts = np.multiply(array, effective_exposure_time)
            # TODO: What if background_sky_map is None? array_counts doesn't exist
            poisson_noise = np.sqrt(array_counts)
            poisson_noise = np.divide(poisson_noise, effective_exposure_time)
        else:
            poisson_noise = None

        # The final image is background subtracted.
        if background_sky_map is not None:
            array -= background_sky_map

        return Image(array, effective_exposure_time=effective_exposure_time, pixel_scale=pixel_scale, psf=psf,
                     background_noise=background_noise, poisson_noise=poisson_noise)

    def background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning image_to_pixel located at the edge(s) of an image
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

    def apply_psf(self, psf):
        """
        Convolve a two-dimensional array with a two-dimensional kernel (e.g. a PSF)

        NOTE1 : The PSF kernel must be size odd x odd to avoid ambiguities with convolution offsets.

        NOTE2 : SciPy has multiple 'mode' options for the size of the output array (e.g. does it include zero padding).
        We require the output array to be the same size as the input image.

        Parameters
        ----------
        psf : ndarray
            A point spread function to apply to this image.
        """

        if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
            raise KernelException("PSF Kernel must be odd")

        return self.new_with_array(scipy.signal.convolve2d(self, psf, mode='same'))

    def electrons_per_second_to_counts(self, array):
        """
        For an array (in electrons per second) and exposure time array, return an array in units counts.

        Parameters
        ----------
        array : ndarray
            The image from which the Poisson signal_to_noise_ratio map is estimated.
        """
        return np.multiply(array, self.effective_exposure_time)

    def counts_to_electrons_per_second(self, array):
        """
        For an array (in counts) and exposure time array, convert the array to units electrons per second

        Parameters
        ----------
        array : ndarray
            The image from which the Poisson signal_to_noise_ratio map is estimated.
        """
        return np.divide(array, self.effective_exposure_time)

    @property
    def counts_array(self):
        return self.electrons_per_second_to_counts(self)

    @property
    def background_noise_counts_array(self):
        return self.electrons_per_second_to_counts(self.background_noise)

    @property
    def estimated_noise_counts(self):
        return np.sqrt(self.counts_array + np.square(self.background_noise_counts_array))

    @property
    def estimated_noise(self):
        return self.counts_to_electrons_per_second(self.estimated_noise_counts)


class PSF(Array):

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

    def renormalize(self):
        """Renormalize the PSF such that its data values sum to unity."""
        return np.divide(self, np.sum(self))

    def convolve(self, array):
        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(array, self, mode='same')


class KernelException(Exception):
    pass


def generate_poisson_noise(image, exposure_time, seed=-1):
    """
    Generate a two-dimensional background noise-map for an image, generating values from a Gaussian
    distribution with mean 0.0.

    Parameters
    ----------
    image : ndarray
        The 2D image background noise is added to.
    exposure_time : Union(ndarray, int)
        The 2D array of pixel exposure times.
    seed : int
        The seed of the random number generator, used for the random noise maps.

    Returns
    -------
    poisson_noise: ndarray
        An array describing simulated poisson noise
    """
    setup_random_seed(seed)
    image_counts = np.multiply(image, exposure_time)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time)


def generate_background_noise(image, sigma, seed=-1):
    setup_random_seed(seed)
    background_noise_map = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    return background_noise_map


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is positive,
    that seed is used for all runs, thereby giving reproducible files

    Parameters
    ----------
    seed : int
        The seed of the random number generator, used for the random signal_to_noise_ratio maps.
    """
    if seed == -1:
        seed = np.random.randint(0, int(1e9))  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)
