import numpy as np
import scipy.signal
from scipy.stats import norm

from autolens import exc
from autolens.imaging import imaging_util
from autolens.imaging.scaled_array import ScaledSquarePixelArray, Array

import logging


logger = logging.getLogger(__name__)


class Image(ScaledSquarePixelArray):

    def __init__(self, array, pixel_scale, psf, noise_map, background_noise_map=None):
        """
        A 2d array representing a real or simulated _datas.

        Parameters
        ----------
        array : ndarray
            An array of the _datas.
        pixel_scale : float
            The scale of each pixel in arc seconds
        psf : PSF
            An array describing the PSF of the _datas.
        noise_map : Array
            An array describing the total noise_map in each _datas pixel.
        background_noise_map : Array
            An array describing the background noise_map in each _datas pixel (used for hyper_image background noise_map
            scaling).
        """
        super(Image, self).__init__(array, pixel_scale)
        self.psf = psf
        self.noise_map = noise_map
        self.background_noise_map = background_noise_map

    def __array_finalize__(self, obj):
        super(Image, self).__array_finalize__(obj)
        if isinstance(obj, Image):
            try:
                self.psf = obj.psf
                self.noise_map = obj.noise_map
                self.background_noise_map = obj.background_noise_map
            except AttributeError:
                logger.debug("Original object in Image.__array_finalize__ missing one or more attributes")

    def trim_image_and_noise_around_centre(self, new_shape):

        if self.background_noise_map is not None:
            background_noise_map = self.background_noise_map.trim_around_centre(new_shape)
        else:
            background_noise_map = None

        return Image(array=self.trim_around_centre(new_shape), pixel_scale=self.pixel_scale, psf=self.psf,
                     noise_map=self.noise_map.trim_around_centre(new_shape), background_noise_map=background_noise_map)

    def trim_image_and_noise_around_region(self, x0, x1, y0, y1):

        if self.background_noise_map is not None:
            background_noise_map = self.background_noise_map.trim_around_region(x0, x1, y0, y1)
        else:
            background_noise_map = None

        return Image(array=self.trim_around_region(x0, x1, y0, y1), pixel_scale=self.pixel_scale, psf=self.psf,
                     noise_map=self.noise_map.trim_around_region(x0, x1, y0, y1),
                     background_noise_map=background_noise_map)

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_map mappers of the _datas."""
        signal_to_noise_map = np.divide(self, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_map in an _datas pixel in the _datas's signal-to-noise_map mappers"""
        return np.max(self.signal_to_noise_map)


class PreparatoryImage(Image):

    def __init__(self, array, pixel_scale, psf, noise_map=None, background_noise_map=None, poisson_noise_map=None,
                 exposure_time=None, effective_exposure_map=None):
        """
        A 2d array representing an _datas, including preparatory components which are not needed for the actual lens
        analysis but help set up the noise_map, background sky, etc.

        Parameters
        ----------
        array : ndarray
            An array of the _datas.
        pixel_scale : float
            The scale of each pixel in arc seconds
        psf : PSF
            An array describing the PSF of the _datas.
        noise_map : Array
            An array describing the total noise_map in each _datas pixel.
        background_noise_map : Array
            An array describing the background noise_map in each _datas pixel (used for hyper_image background noise_map
            scaling).
        poisson_noise_map : ndarray
            An array describing the poisson noise_map in each _datas pixel (used for checking the _datas units are
            sensible).
        exposure_time : float
            The overall exposure time of the _datas.
        effective_exposure_map : ndarray
            An array representing the effective exposure time of each pixel.
        """
        super(PreparatoryImage, self).__init__(array, pixel_scale, psf, noise_map, background_noise_map)
        self.poisson_noise_map = poisson_noise_map
        self.exposure_time = exposure_time
        self.effective_exposure_map = effective_exposure_map

    @classmethod
    def simulate(cls, array, pixel_scale, exposure_time, psf=None, background_sky_level=None,
                 add_noise=False, seed=-1):

        effective_exposure_map = ScaledSquarePixelArray.single_value(value=exposure_time, shape=array.shape,
                                                                     pixel_scale=pixel_scale)
        if background_sky_level is not None:
            background_sky_map = ScaledSquarePixelArray.single_value(value=background_sky_level, shape=array.shape,
                                                                     pixel_scale=pixel_scale)
        else:
            background_sky_map = None

        return cls.simulate_variable_arrays(array=array, pixel_scale=pixel_scale,
                                            effective_exposure_map=effective_exposure_map, psf=psf,
                                            background_sky_map=background_sky_map,
                                            add_noise=add_noise, seed=seed)

    @classmethod
    def simulate_variable_arrays(cls, array, pixel_scale, effective_exposure_map, psf=None, background_sky_map=None,
                                 add_noise=True, seed=-1):
        """
        Create a realistic simulated _datas by applying effects to a plain simulated _datas.

        Parameters
        ----------
        array: ndarray
            The _datas before simulating (e.g. the lens and source galaxies before optics blurring and CCD read-out).
        pixel_scale: float
            The scale of each pixel in arc seconds
        effective_exposure_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated _datas is blurred with.
        background_sky_map : ndarray
            The value of background sky in every _datas pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_map is simulated and added to the _datas, based on the total counts in each _datas
            pixel
        seed: int
            A seed for random noise_map generation
        """

        if background_sky_map is not None:
            array += background_sky_map

        if psf is not None:
            array = psf.convolve(array)
            array = cls.trim_psf_edges(array, psf)
            effective_exposure_map = cls.trim_psf_edges(effective_exposure_map, psf)
            if background_sky_map is not None:
                background_sky_map = cls.trim_psf_edges(background_sky_map, psf)

        if add_noise is True:
            array += generate_poisson_noise(array, effective_exposure_map, seed)
            array_counts = np.multiply(array, effective_exposure_map)
            noise_map = np.divide(np.sqrt(array_counts), effective_exposure_map)
        else:
            noise_map = None

        if background_sky_map is not None:
            array -= background_sky_map

        # ESTIMATE THE BACKGROUND NOISE MAP FROM THE IMAGE

        if background_sky_map is not None:
            background_noise_map_counts = np.sqrt(np.multiply(background_sky_map, effective_exposure_map))
            background_noise_map = np.divide(background_noise_map_counts, effective_exposure_map)
        else:
            background_noise_map = None

        # ESTIMATE HTE POISSON NOISE MAP FROM THE IMAGE

        array_counts = np.multiply(array, effective_exposure_map)
        poisson_noise_map = np.divide(np.sqrt(np.abs(array_counts)), effective_exposure_map)

        return PreparatoryImage(array, pixel_scale=pixel_scale, noise_map=noise_map, psf=psf,
                                background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                                effective_exposure_map=effective_exposure_map)

    @classmethod
    def simulate_to_target_signal_to_noise(cls, array, pixel_scale, target_signal_to_noise, effective_exposure_map,
                                           psf=None, background_sky_map=None, seed=-1):
        """
        Create a realistic simulated _datas by applying effects to a plain simulated _datas.

        Parameters
        ----------
        target_signal_to_noise
        array: ndarray
            The _datas before simulating (e.g. the lens and source galaxies before optics blurring and CCD read-out).
        pixel_scale: float
            The scale of each pixel in arc seconds
        effective_exposure_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated _datas is blurred with.
        background_sky_map : ndarray
            The value of background sky in every _datas pixel (electrons per second).
        seed: int
            A seed for random noise_map generation
        """

        max_index = np.unravel_index(array.argmax(), array.shape)
        max_array = array[max_index]
        max_effective_exposure_time = effective_exposure_map[max_index]
        max_array_counts = np.multiply(max_array, max_effective_exposure_time)
        if background_sky_map is not None:
            max_background_sky_map = background_sky_map[max_index]
            max_background_sky_map_counts = np.multiply(max_background_sky_map, max_effective_exposure_time)
        else:
            max_background_sky_map_counts = None

        scale_factor = 1.

        if background_sky_map is None:
            scale_factor = target_signal_to_noise ** 2.0 / max_array_counts
        elif background_sky_map is not None:
            scale_factor = (max_array_counts + max_background_sky_map_counts) * target_signal_to_noise ** 2.0 \
                           / max_array_counts ** 2.0

        scaled_effective_exposure_time = np.multiply(scale_factor, effective_exposure_map)

        return cls.simulate_variable_arrays(array=array, pixel_scale=pixel_scale,
                                            effective_exposure_map=scaled_effective_exposure_time,
                                            psf=psf, background_sky_map=background_sky_map,
                                            add_noise=True, seed=seed)

    def __array_finalize__(self, obj):
        super(PreparatoryImage, self).__array_finalize__(obj)
        if isinstance(obj, PreparatoryImage):
            self.psf = obj.psf
            self.noise_map = obj.noise_map
            self.background_noise_map = obj.background_noise_map
            self.poisson_noise_map = obj.poisson_noise_map
            self.effective_exposure_map = obj.effective_exposure_map

    @staticmethod
    def trim_psf_edges(array, psf):
        if psf is not None:
            psf_cut_x = np.int(np.ceil(psf.shape[0] / 2)) - 1
            psf_cut_y = np.int(np.ceil(psf.shape[1] / 2)) - 1
            array_x = np.int(array.shape[0])
            array_y = np.int(array.shape[1])
            return array[psf_cut_x:array_x - psf_cut_x, psf_cut_y:array_y - psf_cut_y]
        else:
            return array

    def electrons_per_second_to_counts(self, array):
        """
        For an array (in electrons per second) and an exposure time mappers, return an array in units counts.

        Parameters
        ----------
        array : ndarray
            The array the values are to be converted from electrons per seconds to counts.
        """
        return np.multiply(array, self.effective_exposure_map)

    def counts_to_electrons_per_second(self, array):
        """
        For an array (in counts) and an exposure time mappers, convert the array to units electrons per second

        Parameters
        ----------
        array : ndarray
            The array the values are to be converted from counts to electrons per second.
        """
        return np.divide(array, self.effective_exposure_map)

    @property
    def image_counts(self):
        """The _datas in units of counts."""
        return self.electrons_per_second_to_counts(self)

    @property
    def background_noise_map_counts(self):
        """ The background noise_map mappers in units of counts."""
        return self.electrons_per_second_to_counts(self.background_noise_map)

    @property
    def estimated_noise_map_counts(self):
        """ The estimated noise_map mappers of the _datas (using its background noise_map mappers and _datas values
        in counts) in counts.
        """
        return np.sqrt((np.abs(self.image_counts) + np.square(self.background_noise_map_counts)))

    @property
    def estimated_noise_map(self):
        """ The estimated noise_map mappers of the _datas (using its background noise_map mappers and _datas values
        in counts) in electrons per second.
        """
        return self.counts_to_electrons_per_second(self.estimated_noise_map_counts)

    def background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning data_to_image located at the edge(s) of an _datas
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


class NoiseMap(ScaledSquarePixelArray):

    @classmethod
    def from_weight_map(cls, pixel_scale, weight_map):
        np.seterr(divide='ignore')
        noise_map = 1.0 / np.sqrt(weight_map)
        noise_map[noise_map == np.inf] = 1.0e8
        return NoiseMap(array=noise_map, pixel_scale=pixel_scale)


class PSF(ScaledSquarePixelArray):

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale, renormalize=False):
        """
        Class storing a 2D Point Spread Function (PSF), including its blurring kernel.

        Parameters
        ----------
        array : ndarray
            The 2d PSF blurring kernel.
        renormalize : bool
            Renormalize the PSF such that he sum of kernel values total 1.0?
        """

        # noinspection PyArgumentList
        super().__init__(array=array, pixel_scale=pixel_scale)
        if renormalize:
            self.renormalize()

    @classmethod
    def simulate_as_gaussian(cls, shape, pixel_scale, sigma, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """Simulate the PSF as an elliptical Gaussian profile."""
        from autolens.profiles.light_profiles import EllipticalGaussian
        gaussian = EllipticalGaussian(centre=centre, axis_ratio=axis_ratio, phi=phi, intensity=1.0, sigma=sigma)
        grid_1d = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=np.full(shape, False),
                                                                               pixel_scales=(pixel_scale, pixel_scale))
        gaussian_1d = gaussian.intensities_from_grid(grid=grid_1d)
        gaussian_2d = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=gaussian_1d,
                                                                                             shape=shape)
        return PSF(array=gaussian_2d, pixel_scale=pixel_scale, renormalize=True)

    @classmethod
    def from_fits_renormalized(cls, file_path, hdu, pixel_scale):
        """Loads a PSF from fits and renormalizes it

        Parameters
        ----------
        pixel_scale
        file_path: String
            The path to the file containing the PSF
        hdu : int
            The HDU the PSF is stored in the .fits file.

        Returns
        -------
        psf: PSF
            A renormalized PSF instance
        """
        psf = PSF.from_fits_with_scale(file_path, hdu, pixel_scale)
        psf.renormalize()
        return psf

    @classmethod
    def from_fits_with_scale(cls, file_path, hdu, pixel_scale):
        """
        Loads the PSF from a .fits file.

        Parameters
        ----------
        pixel_scale
        file_path: String
            The path to the file containing the PSF
        hdu : int
            The HDU the PSF is stored in the .fits file.
        """
        return cls(array=imaging_util.numpy_array_from_fits(file_path, hdu), pixel_scale=pixel_scale)

    def renormalize(self):
        """Renormalize the PSF such that its data_vector values sum to unity."""
        self[:, :] = np.divide(self, np.sum(self))

    def convolve(self, array):
        """
        Convolve an array with this PSF

        Parameters
        ----------
        array: ndarray
            An array representing the _datas the PSF is convolved with.

        Returns
        -------
        convolved_array: ndarray
            An array representing the _datas after convolution.

        Raises
        ------
        KernelException if either PSF psf dimension is odd
        """
        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise exc.KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(array, self, mode='same')


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is \
    positive, that seed is used for all runs, thereby giving reproducible results.

    Parameters
    ----------
    seed : int
        The seed of the random number generator.
    """
    if seed == -1:
        seed = np.random.randint(0,
                                 int(1e9))  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


def generate_poisson_noise(image, effective_exposure_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_map-mappers from an _datas.

    Values are computed from a Poisson distribution using the _datas's input values in units of counts.

    Parameters
    ----------
    image : ndarray
        The 2D _datas, whose values in counts are used to draw Poisson noise_map values.
    effective_exposure_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed : int
        The seed of the random number generator, used for the random noise_map maps.

    Returns
    -------
    poisson_noise_map: ndarray
        An array describing simulated poisson noise_map
    """
    setup_random_seed(seed)
    image_counts = np.multiply(image, effective_exposure_map)
    return image - np.divide(np.random.poisson(image_counts, image.shape), effective_exposure_map)


def load_imaging_from_fits(image_path, noise_map_path, psf_path, pixel_scale, image_hdu=0, noise_map_hdu=0, psf_hdu=0,
                           image_shape=None, psf_shape=None, noise_map_is_weight_map=False):
    data = ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=image_path, hdu=image_hdu,
                                                             pixel_scale=pixel_scale)

    noise_map = None

    if not noise_map_is_weight_map:
        noise_map = NoiseMap.from_fits_with_pixel_scale(file_path=noise_map_path, hdu=noise_map_hdu,
                                                        pixel_scale=pixel_scale)
    elif noise_map_is_weight_map:
        weight_map = Array.from_fits(file_path=noise_map_path, hdu=noise_map_hdu)
        noise_map = NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=pixel_scale)

    if image_shape is not None:
        data = data.resize_around_centre(new_shape=image_shape)
        noise_map = noise_map.resize_around_centre(new_shape=image_shape)

    psf = PSF.from_fits_with_scale(file_path=psf_path, hdu=psf_hdu, pixel_scale=pixel_scale)

    if psf_shape is not None:
        psf = psf.resize_around_centre(new_shape=psf_shape)

    return Image(array=data, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map)


def load_imaging_from_path(image_path, noise_map_path, psf_path, pixel_scale, psf_trimmed_shape=None):
    data = ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=image_path, hdu=0, pixel_scale=pixel_scale)
    noise = ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=noise_map_path, hdu=0, pixel_scale=pixel_scale)
    psf = PSF.from_fits_with_scale(file_path=psf_path, hdu=0, pixel_scale=pixel_scale)
    if psf_trimmed_shape is not None:
        psf = psf.trim_around_centre(psf_trimmed_shape)
    return Image(array=data, pixel_scale=pixel_scale, psf=psf, noise_map=noise)


def output_imaging_to_fits(image, image_path, noise_map_path, psf_path, overwrite=False):
    imaging_util.numpy_array_to_fits(array=image, path=image_path, overwrite=overwrite)
    imaging_util.numpy_array_to_fits(array=image.noise_map, path=noise_map_path, overwrite=overwrite)
    imaging_util.numpy_array_to_fits(array=image.psf, path=psf_path, overwrite=overwrite)
