from autolens.imaging.scaled_array import ScaledArray, AbstractArray
import numpy as np
from scipy.stats import norm
import scipy.signal
from autolens import exc


# TODO : The idea is that we don't need functions to estimate the noise or the exposure time once we set up an image
# TODO : so we'll leave this functionality to a class that loads images with what we're given.

class PrepatoryImage(ScaledArray):

    def __init__(self, array, pixel_scale, psf, noise=None, background_noise=None, poisson_noise=None,
                 effective_exposure_time=None):
        """
        A 2d array representing a real or simulated image.

        Parameters
        ----------
        array: ndarray
            An array of image pixels in gray-scale
        noise: ndarray
            An array describing the noise in the image
        effective_exposure_time: Union(ndarray, float)
            A float or array representing the effective exposure time of the whole image or each pixel.
        pixel_scale: float
            The scale of each pixel in arc seconds
        psf: PSF
            An array describing the PSF
        """
        super(PrepatoryImage, self).__init__(array, pixel_scale)
        self.psf = psf
        self.noise = noise
        self.background_noise = background_noise
        self.poisson_noise = poisson_noise
        self.effective_exposure_time = effective_exposure_time

    @classmethod
    def simulate(cls, array, pixel_scale, effective_exposure_time, psf=None, background_sky_map=None,
                 include_poisson_noise=False, seed=-1):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        array: ndarray
            A plain image
        effective_exposure_time: Union(ndarray, float)
            A float or array representing the effective exposure time of the whole image or each pixel.
        pixel_scale: float
            The scale of each pixel in arc seconds
        psf: PSF
            An array describing the PSF
        background_sky_map
        include_poisson_noise: Bool
            If True poisson noise is simulated and added to the image
        seed: int
            A seed for random noise generation

        Returns
        -------
        image: PrepatoryImage
            A simulated image
        """

        array_counts = None

        if background_sky_map is not None:
            array += background_sky_map
            background_noise_counts = np.sqrt(np.multiply(background_sky_map, effective_exposure_time))
            background_noise = np.divide(background_noise_counts, effective_exposure_time)
        else:
            background_noise_counts = None
            background_noise = None

        if psf is not None:
            array = psf.convolve(array)
            array = cls.trim_psf_edges(array, psf)
            effective_exposure_time = cls.trim_psf_edges(effective_exposure_time, psf)
            if background_sky_map is not None:
                background_sky_map = cls.trim_psf_edges(background_sky_map, psf)
            if background_noise_counts is not None:
                background_noise_counts = cls.trim_psf_edges(background_noise_counts, psf)
                background_noise = cls.trim_psf_edges(background_noise, psf)

        if include_poisson_noise is True:

            array += generate_poisson_noise(array, effective_exposure_time, seed)

            # The poisson noise map does not include the background sky, so this estimate below removes it
            if background_sky_map is not None:
                array_counts = np.multiply(np.subtract(array, background_sky_map), effective_exposure_time)
            elif background_sky_map is None:
                array_counts = np.multiply(array, effective_exposure_time)

            poisson_noise = np.divide(np.sqrt(array_counts), effective_exposure_time)

        else:

            poisson_noise = None

        # The final image is background subtracted.
        if background_sky_map is not None:
            array -= background_sky_map
        if background_sky_map is not None and include_poisson_noise is False:
            noise = np.divide(background_noise_counts, effective_exposure_time)
        elif background_sky_map is None and include_poisson_noise is True:
            noise = np.divide(array_counts, effective_exposure_time)
        elif background_sky_map is not None and include_poisson_noise is True:
            noise = np.divide(np.sqrt(array_counts + np.square(background_noise_counts)), effective_exposure_time)
        else:
            noise = None

        if noise is not None:
            if (np.isnan(noise)).any():
                raise exc.MaskException('Nan found in poisson noise - increase exposure time.')

        return PrepatoryImage(array, pixel_scale=pixel_scale, noise=noise, psf=psf, background_noise=background_noise,
                              poisson_noise=poisson_noise, effective_exposure_time=effective_exposure_time)

    def __array_finalize__(self, obj):
        super(PrepatoryImage, self).__array_finalize__(obj)
        if isinstance(obj, PrepatoryImage):
            self.psf = obj.psf
            self.noise = obj.noise
            self.background_noise = obj.background_noise
            self.poisson_noise = obj.poisson_noise
            self.effective_exposure_time = obj.effective_exposure_time

    @staticmethod
    def trim_psf_edges(array, psf):
        psf_cut_x = np.int(np.ceil(psf.shape[0] / 2)) - 1
        psf_cut_y = np.int(np.ceil(psf.shape[1] / 2)) - 1
        array_x = np.int(array.shape[0])
        array_y = np.int(array.shape[1])
        return array[psf_cut_x:array_x - psf_cut_x, psf_cut_y:array_y - psf_cut_y]

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
        """
        Returns
        -------
        counts_array: ndarray
            An array representing the image in terms of counts
        """
        return self.electrons_per_second_to_counts(self)

    @property
    def background_noise_counts_array(self):
        """
        Returns
        -------
        background_noise_counts_array: ndarray
            An array representing the background noise in terms of counts
        """
        return self.electrons_per_second_to_counts(self.background_noise)

    @property
    def estimated_noise_counts(self):
        """
        Returns
        -------
        estimated_noise_counts: ndarray
            An array representing estimated noise in terms of counts
        """
        return np.sqrt(self.counts_array + np.square(self.background_noise_counts_array))

    @property
    def estimated_noise(self):
        """
        Returns
        -------
        estimated_noise: ndarray
            An array representing estimated noise
        """
        return self.counts_to_electrons_per_second(self.estimated_noise_counts)

    def background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning data_to_image located at the edge(s) of an image
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


class Image(ScaledArray):

    def __init__(self, array, pixel_scale, noise, psf):
        """
        A 2d array representing a real or simulated image.

        Parameters
        ----------
        array: ndarray
            An array of image pixels in gray-scale
        noise: ndarray
            An array describing the noise in the image
        pixel_scale: float
            The scale of each pixel in arc seconds
        psf: PSF
            An array describing the PSF
        """
        super(Image, self).__init__(array, pixel_scale)
        self.noise = noise
        self.psf = psf

    def __array_finalize__(self, obj):
        super(Image, self).__array_finalize__(obj)
        if isinstance(obj, Image):
            self.psf = obj.psf
            self.noise = obj.noise


class PSF(AbstractArray):

    # noinspection PyUnusedLocal
    def __init__(self, array, renormalize=True):
        """
        Class storing a 2D Point Spread Function (PSF), including its weighted_data and coordinate grid_coords.

        Parameters
        ----------
        array : ndarray
            The psf weighted_data.
        renormalize : bool
            Renormalize the PSF such that its value added up to 1.0?
        """

        # noinspection PyArgumentList
        super().__init__()
        if renormalize:
            self.renormalize()

    @classmethod
    def from_fits_renormalized(cls, file_path, hdu, pixel_scale):
        """
        Loads a PSF from fits and renormalizes it

        Parameters
        ----------
        file_path: String
            The path to the file containing the PSF
        hdu: int
            HDU ??
        pixel_scale: float
            The scale of a pixel in arcseconds

        Returns
        -------
        psf: PSF
            A renormalized PSF instance
        """
        psf = PSF.from_fits(file_path, hdu, pixel_scale)
        psf.renormalize()
        return psf

    def renormalize(self):
        """Renormalize the PSF such that its weighted_data values sum to unity."""
        return np.divide(self, np.sum(self))

    def convolve(self, array):
        """
        Convolve an array with this PSF

        Parameters
        ----------
        array: ndarray
            An array representing an image

        Returns
        -------
        convolved_array: ndarray
            An array representing an image that has been convolved with this PSF

        Raises
        ------
        KernelException if either PSF psf dimension is odd
        """
        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise exc.KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(array, self, mode='same')


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is positive,
    that seed is used for all runs, thereby giving reproducible nlo

    Parameters
    ----------
    seed : int
        The seed of the random number generator, used for the random signal_to_noise_ratio maps.
    """
    if seed == -1:
        seed = np.random.randint(0,
                                 int(1e9))  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


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
