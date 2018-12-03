import numpy as np
import scipy.signal
from scipy.stats import norm

from autolens import exc
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util, array_util
from autolens.data.array.scaled_array import ScaledSquarePixelArray, Array

import logging


logger = logging.getLogger(__name__)


class Image(ScaledSquarePixelArray):

    def __init__(self, array, pixel_scale, psf, noise_map=None, background_noise_map=None, poisson_noise_map=None,
                 exposure_time_map=None, background_sky_map=None, **kwargs):
        """
        A 2d array representing a real or simulated data.

        Parameters
        ----------
        array : ndarray
            The array of the regular data, preferably in units of electrons per second.
        pixel_scale : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the data.
        noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        background_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the background sky noise,
            preferably in units of electrons per second.
        poisson_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the Poisson counts of the source,
            preferably in units of electrons per second.
        exposure_time_map : ScaledSquarePixelArray
            An array describing the effective exposure time in each regular pixel.
        background_sky_map : ScaledSquarePixelArray
            An array describing the background sky.
        """
        super(Image, self).__init__(array=array, pixel_scale=pixel_scale)
        self.psf = psf
        self.noise_map = noise_map
        self.background_noise_map = background_noise_map
        self.poisson_noise_map = poisson_noise_map
        self.exposure_time_map = exposure_time_map
        self.background_sky_map = background_sky_map
        self.origin = (0.0, 0.0)

    @classmethod
    def simulate(cls, array, pixel_scale, exposure_time, psf=None, background_sky_level=None,
                 add_noise=False, seed=-1):

        exposure_time_map = ScaledSquarePixelArray.single_value(value=exposure_time, shape=array.shape,
                                                                     pixel_scale=pixel_scale)
        if background_sky_level is not None:
            background_sky_map = ScaledSquarePixelArray.single_value(value=background_sky_level, shape=array.shape,
                                                                     pixel_scale=pixel_scale)
        else:
            background_sky_map = None

        return cls.simulate_variable_arrays(array=array, pixel_scale=pixel_scale,
                                            exposure_time_map=exposure_time_map, psf=psf,
                                            background_sky_map=background_sky_map,
                                            add_noise=add_noise, seed=seed)

    @classmethod
    def simulate_variable_arrays(cls, array, pixel_scale, exposure_time_map, psf=None, background_sky_map=None,
                                 add_noise=True, seed=-1):
        """
        Create a realistic simulated data by applying effects to a plain simulated data.

        Parameters
        ----------
        array: ndarray
            The data before simulating (e.g. the lens and source galaxies before optics blurring and CCD read-out).
        pixel_scale: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated data is blurred with.
        background_sky_map : ndarray
            The value of background sky in every data pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_map is simulated and added to the data, based on the total counts in each data
            pixel
        seed: int
            A seed for random noise_map generation
        """

        if background_sky_map is not None:
            array += background_sky_map

        if psf is not None:
            array = psf.convolve(array)
            array = cls.trim_psf_edges(array, psf)
            exposure_time_map = cls.trim_psf_edges(exposure_time_map, psf)
            if background_sky_map is not None:
                background_sky_map = cls.trim_psf_edges(background_sky_map, psf)

        if add_noise is True:
            array += generate_poisson_noise(array, exposure_time_map, seed)
            array_counts = np.multiply(array, exposure_time_map)
            noise_map = np.divide(np.sqrt(array_counts), exposure_time_map)
        else:
            noise_map = None

        if background_sky_map is not None:
            array -= background_sky_map

        # ESTIMATE THE BACKGROUND NOISE MAP FROM THE IMAGE

        if background_sky_map is not None:
            background_noise_map_counts = np.sqrt(np.multiply(background_sky_map, exposure_time_map))
            background_noise_map = np.divide(background_noise_map_counts, exposure_time_map)
        else:
            background_noise_map = None

        # ESTIMATE THE POISSON NOISE MAP FROM THE IMAGE

        array_counts = np.multiply(array, exposure_time_map)
        poisson_noise_map = np.divide(np.sqrt(np.abs(array_counts)), exposure_time_map)

        noise_map = NoiseMap(array=noise_map, pixel_scale=pixel_scale)

        return Image(array, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map,
                     background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                     exposure_time_map=exposure_time_map, background_sky_map=background_sky_map)

    @classmethod
    def simulate_to_target_signal_to_noise(cls, array, pixel_scale, target_signal_to_noise, exposure_time_map,
                                           psf=None, background_sky_map=None, seed=-1):
        """
        Create a realistic simulated data by applying effects to a plain simulated data.

        Parameters
        ----------
        target_signal_to_noise
        array: ndarray
            The data before simulating (e.g. the lens and source galaxies before optics blurring and CCD read-out).
        pixel_scale: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated data is blurred with.
        background_sky_map : ndarray
            The value of background sky in every data pixel (electrons per second).
        seed: int
            A seed for random noise_map generation
        """

        max_index = np.unravel_index(array.argmax(), array.shape)
        max_array = array[max_index]
        max_effective_exposure_time = exposure_time_map[max_index]
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

        scaled_effective_exposure_time = np.multiply(scale_factor, exposure_time_map)

        return cls.simulate_variable_arrays(array=array, pixel_scale=pixel_scale,
                                            exposure_time_map=scaled_effective_exposure_time,
                                            psf=psf, background_sky_map=background_sky_map,
                                            add_noise=True, seed=seed)

    def new_image_with_resized_arrays(self, new_shape, new_centre_pixels=None, new_centre_arc_seconds=None):
        
        array = self.resize_scaled_array(scaled_array=self, new_shape=new_shape, new_centre_pixels=new_centre_pixels,
                                         new_centre_arc_seconds=new_centre_arc_seconds)

        noise_map = self.resize_scaled_array(scaled_array=self.noise_map, new_shape=new_shape,
                                             new_centre_pixels=new_centre_pixels,
                                             new_centre_arc_seconds=new_centre_arc_seconds)

        background_noise_map = self.resize_scaled_array(scaled_array=self.background_noise_map, new_shape=new_shape,
                                                        new_centre_pixels=new_centre_pixels,
                                                        new_centre_arc_seconds=new_centre_arc_seconds)

        poisson_noise_map = self.resize_scaled_array(scaled_array=self.poisson_noise_map, new_shape=new_shape,
                                                     new_centre_pixels=new_centre_pixels,
                                                     new_centre_arc_seconds=new_centre_arc_seconds)

        exposure_time_map = self.resize_scaled_array(scaled_array=self.exposure_time_map, new_shape=new_shape,
                                                     new_centre_pixels=new_centre_pixels,
                                                     new_centre_arc_seconds=new_centre_arc_seconds)

        background_sky_map = self.resize_scaled_array(scaled_array=self.background_sky_map, new_shape=new_shape,
                                                      new_centre_pixels=new_centre_pixels,
                                                      new_centre_arc_seconds=new_centre_arc_seconds)

        return Image(array=array, pixel_scale=self.pixel_scale, psf=self.psf, noise_map=noise_map,
                     background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                     exposure_time_map=exposure_time_map, background_sky_map=background_sky_map)

    def new_image_with_resized_psf(self, new_shape):
        psf = self.resize_scaled_array(scaled_array=self.psf, new_shape=new_shape)
        return Image(array=self, pixel_scale=self.pixel_scale, psf=psf, noise_map=self.noise_map,
                     background_noise_map=self.background_noise_map, poisson_noise_map=self.poisson_noise_map,
                     exposure_time_map=self.exposure_time_map, background_sky_map=self.background_sky_map)

    @staticmethod
    def resize_scaled_array(scaled_array, new_shape, new_centre_pixels=None, new_centre_arc_seconds=None):
        if scaled_array is not None:
            return scaled_array.resized_scaled_array_from_array(new_shape=new_shape,
                        new_centre_pixels=new_centre_pixels, new_centre_arc_seconds=new_centre_arc_seconds)
        else:
            return None

    def new_image_with_poisson_noise_added(self, seed=-1):

        image_with_sky = self + self.background_sky_map

        image_with_sky_and_noise = image_with_sky + generate_poisson_noise(image=image_with_sky,
                                            exposure_time_map=self.exposure_time_map, seed=seed)

        image_with_noise = image_with_sky_and_noise - self.background_sky_map

        return Image(array=image_with_noise, pixel_scale=self.pixel_scale, psf=self.psf,
                     noise_map=self.noise_map, background_noise_map=self.background_noise_map,
                     poisson_noise_map=self.poisson_noise_map)

    def new_image_converted_from_counts(self):

        array = self.counts_to_electrons_per_second(array=self)
        noise_map = self.counts_to_electrons_per_second(array=self.noise_map)
        background_noise_map = self.counts_to_electrons_per_second(array=self.background_noise_map)
        poisson_noise_map = self.counts_to_electrons_per_second(array=self.poisson_noise_map)
        background_sky_map = self.counts_to_electrons_per_second(array=self.background_sky_map)

        return Image(array=array, pixel_scale=self.pixel_scale, psf=self.psf, noise_map=noise_map,
                     background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                     exposure_time_map=self.exposure_time_map, background_sky_map=background_sky_map)

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_map mappers of the data."""
        signal_to_noise_map = np.divide(self, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_map in an data pixel in the data's signal-to-noise_map mappers"""
        return np.max(self.signal_to_noise_map)

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
        return np.multiply(array, self.exposure_time_map)

    def counts_to_electrons_per_second(self, array):
        """
        For an array (in counts) and an exposure time mappers, convert the array to units electrons per second

        Parameters
        ----------
        array : ndarray
            The array the values are to be converted from counts to electrons per second.
        """
        if array is not None:
            return np.divide(array, self.exposure_time_map)
        else:
            return None

    @property
    def image_counts(self):
        """The data in units of counts."""
        return self.electrons_per_second_to_counts(self)

    @property
    def background_noise_map_counts(self):
        """ The background noise_map mappers in units of counts."""
        return self.electrons_per_second_to_counts(self.background_noise_map)

    @property
    def estimated_noise_map_counts(self):
        """ The estimated noise_map mappers of the data (using its background noise_map mappers and data values
        in counts) in counts.
        """
        return np.sqrt((np.abs(self.image_counts) + np.square(self.background_noise_map_counts)))

    @property
    def estimated_noise_map(self):
        """ The estimated noise_map mappers of the data (using its background noise_map mappers and data values
        in counts) in electrons per second.
        """
        return self.counts_to_electrons_per_second(self.estimated_noise_map_counts)

    def background_noise_from_edges(self, no_edges):
        """Estimate the background signal_to_noise_ratio by binning data_to_image located at the edge(s) of an data
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

    def __array_finalize__(self, obj):
        super(Image, self).__array_finalize__(obj)
        if isinstance(obj, Image):
            try:
                self.psf = obj.psf
                self.noise_map = obj.noise_map
                self.background_noise_map = obj.background_noise_map
                self.poisson_noise_map = obj.poisson_noise_map
                self.exposure_time_map = obj.exposure_time_map
                self.background_sky_map = obj.background_sky_map
                self.origin = obj.origin
            except AttributeError:
                logger.debug("Original object in Image.__array_finalize__ missing one or more attributes")


class NoiseMap(ScaledSquarePixelArray):

    @classmethod
    def from_weight_map(cls, pixel_scale, weight_map):
        """Setup the noise-map from a weight map, which is a form of noise-map that comes via HST data-reduction and \
        the software package MultiDrizzle.

        The variance in each pixel is computed as:

        Variance = 1.0 / sqrt(weight_map).

        The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
        the analysis.

        Parameters
        -----------
        pixel_scale : float
            The size of each pixel in arc seconds.
        weight_map : ndarray
            The weight-value of each pixel which is converted to a variance.
        """
        np.seterr(divide='ignore')
        noise_map = 1.0 / np.sqrt(weight_map)
        noise_map[noise_map == np.inf] = 1.0e8
        return NoiseMap(array=noise_map, pixel_scale=pixel_scale)

    @classmethod
    def from_inverse_noise_map(cls, pixel_scale, inverse_noise_map):
        """Setup the noise-map from an root-mean square standard deviation map, which is a form of noise-map that \
        comes via HST data-reduction and the software package MultiDrizzle.

        The variance in each pixel is computed as:

        Variance = 1.0 / inverse_std_map.

        The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
        the analysis.

        Parameters
        -----------
        pixel_scale : float
            The size of each pixel in arc seconds.
        inverse_noise_map : ndarray
            The inverse noise value of each pixel which is converted to a variance.
        """
        noise_map = 1.0 / inverse_noise_map
        return NoiseMap(array=noise_map, pixel_scale=pixel_scale)

    @classmethod
    def from_image_and_background_noise_map(cls, pixel_scale, image, background_noise_map, exposure_time_map,
                                            convert_from_counts=False):
        if not convert_from_counts:
            return NoiseMap(array=np.sqrt(np.abs((background_noise_map*exposure_time_map)**2.0 +
                                          image*exposure_time_map)) / exposure_time_map,
                            pixel_scale=pixel_scale)
        elif convert_from_counts:
            return NoiseMap(array=np.sqrt(np.abs(background_noise_map**2.0 + image)), pixel_scale=pixel_scale)


class PoissonNoiseMap(NoiseMap):

    @classmethod
    def from_image_and_exposure_time_map(cls, pixel_scale, image, exposure_time_map, convert_from_counts=False):
        if not convert_from_counts:
            return PoissonNoiseMap(array=np.sqrt(image*exposure_time_map), pixel_scale=pixel_scale)
        elif convert_from_counts:
            return PoissonNoiseMap(array=np.sqrt(image), pixel_scale=pixel_scale)


class PSF(ScaledSquarePixelArray):

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale, renormalize=False, **kwargs):
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
        from autolens.model.profiles.light_profiles import EllipticalGaussian
        gaussian = EllipticalGaussian(centre=centre, axis_ratio=axis_ratio, phi=phi, intensity=1.0, sigma=sigma)
        grid_1d = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full(shape, False),
                                                                                     pixel_scales=(pixel_scale, pixel_scale))
        gaussian_1d = gaussian.intensities_from_grid(grid=grid_1d)
        gaussian_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=gaussian_1d,
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
        return cls(array=array_util.numpy_array_from_fits(file_path, hdu), pixel_scale=pixel_scale)

    def renormalize(self):
        """Renormalize the PSF such that its data_vector values sum to unity."""
        self[:, :] = np.divide(self, np.sum(self))

    def convolve(self, array):
        """
        Convolve an array with this PSF

        Parameters
        ----------
        array: ndarray
            An array representing the data the PSF is convolved with.

        Returns
        -------
        convolved_array: ndarray
            An array representing the data after convolution.

        Raises
        ------
        KernelException if either PSF psf dimension is odd
        """
        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise exc.KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(array, self, mode='same')


class ExposureTimeMap(ScaledSquarePixelArray):

    @classmethod
    def from_exposure_time_and_background_noise_map(cls, pixel_scale, exposure_time, background_noise_map):
        relative_background_noise_map = background_noise_map / np.max(background_noise_map)
        return ExposureTimeMap(array=np.abs(exposure_time * (relative_background_noise_map)), pixel_scale=pixel_scale)


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


def generate_poisson_noise(image, exposure_time_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_map-mappers from an data.

    Values are computed from a Poisson distribution using the data's input values in units of counts.

    Parameters
    ----------
    image : ndarray
        The 2D data, whose values in counts are used to draw Poisson noise_map values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed : int
        The seed of the random number generator, used for the random noise_map maps.

    Returns
    -------
    poisson_noise_map: ndarray
        An array describing simulated poisson noise_map
    """
    setup_random_seed(seed)
    image_counts = np.multiply(image, exposure_time_map)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time_map)


def load_imaging_from_fits(image_path, pixel_scale, image_hdu=0,
                           resized_image_shape=None, resized_image_origin_pixels=None,
                           resized_image_origin_arc_seconds=None,
                           psf_path=None, psf_hdu=0, resized_psf_shape=None, renormalize_psf=False,
                           noise_map_path=None, noise_map_hdu=0,
                           noise_map_from_image_and_background_noise_map=False,
                           convert_noise_map_from_weight_map=False,
                           convert_noise_map_from_inverse_noise_map=False,
                           background_noise_map_path=None, background_noise_map_hdu=0,
                           convert_background_noise_map_from_weight_map=False,
                           convert_background_noise_map_from_inverse_noise_map=False,
                           poisson_noise_map_path=None, poisson_noise_map_hdu=0,
                           poisson_noise_map_from_image=False,
                           convert_poisson_noise_map_from_weight_map=False,
                           convert_poisson_noise_map_from_inverse_noise_map=False,
                           exposure_time_map_path=None, exposure_time_map_hdu=0,
                           exposure_time_map_from_single_value=None,
                           exposure_time_map_from_background_noise_map=False,
                           background_sky_map_path=None, background_sky_map_hdu=0,
                           convert_from_counts=False):
    """Factory for loading the imaging data from a .fits file, as well as computing other properties like the noise-map,
    exposure-time map, etc. from the existing data.

    This factory also includes a number of routines for converting images from other units (e.g. counts) to electrons \
    per second.

    Parameters
    ----------
    image_path : str
        The path and filename of the .fits data containing the regular.
    pixel_scale : float
        The size of each pixel in arc seconds.
    image_hdu : int
        The hdu the regular is contained in the .fits file that *image_path* points too.
    resized_image_shape : (int, int) | None
        If input, the regular (and arrays that are regular sized, e.g. the noise-maps) are resized to these dimensions.
    resized_image_origin_pixels : (int, int) | None
        If the regular is resized, this defines a new origin in pixels around which recentering occurs.
    resized_image_origin_arc_seconds : (float, float) | None
        If the regular is resized, this defines a new origin in arc-seconds around which recentering occurs.
    psf_path : str
        The path and filename of the .fits data containing the PSF.
    psf_hdu : int
        The hdu the psf is contained in the .fits file that *psf_path* points too.
    resized_psf_shape : (int, int) | None
        If input, the psf is resized to these dimensions.
    renormalize_psf : bool
        If True, the PSF is renoralized such that all elements sum to 1.0.
    noise_map_path : str
        The path and filename of the .fits data containing the noise-map.
    noise_map_hdu : int
        The hdu the noise-map is contained in the .fits file that *noise_map_path* points too.
    noise_map_from_image_and_background_noise_map : bool
        If True, the noise-map is computed from the observed regular and background noise-map \
        (see NoiseMap.from_image_and_background_noise_map).
    convert_noise_map_from_weight_map : bool
        If True, the noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_noise_map_from_inverse_noise_map : bool
        If True, the noise-map loaded from the .fits file is converted from an inverse noise-map to a noise-map (see \
        *NoiseMap.from_inverse_noise_map).
    background_noise_map_path : str
        The path and filename of the .fits data containing the background noise-map.
    background_noise_map_hdu : int
        The hdu the background noise-map is contained in the .fits file that *background_noise_map_path* points too.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    poisson_noise_map_path : str
        The path and filename of the .fits data containing the Poisson noise-map.
    poisson_noise_map_hdu : int
        The hdu the Poisson noise-map is contained in the .fits file that *poisson_noise_map_path* points too.
    poisson_noise_map_from_image : bool
        If True, the Poisson noise-map is estimated using the regular-data.
    convert_poisson_noise_map_from_weight_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_poisson_noise_map_from_inverse_noise_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    exposure_time_map_path : str
        The path and filename of the .fits data containing the exposure time map.
    exposure_time_map_hdu : int
        The hdu the exposure time map is contained in the .fits file that *exposure_time_map_path* points too.
    exposure_time_map_from_single_value : float
        The exposure time of the regular, which computes the exposure-time map as a single value \
        (see *ExposureTimeMap.from_single_value*).
    exposure_time_map_from_background_noise_map : bool
        If True, the exposure-time map is computed from the background noise map \
        (see *ExposureTimeMap.from_background_noise_map*)
    backgrond_sky_map_path : str
        The path and filename of the .fits data containing the backgrond sky map.
    backgrond_sky_map_hdu : int
        The hdu the backgrond sky map is contained in the .fits file that *backgrond_sky_map_path* points too.
    """

    image = load_image(image_path=image_path, image_hdu=image_hdu, pixel_scale=pixel_scale)

    background_noise_map = load_background_noise_map(background_noise_map_path=background_noise_map_path,
             background_noise_map_hdu=background_noise_map_hdu,
             pixel_scale=pixel_scale,
             convert_background_noise_map_from_weight_map=convert_background_noise_map_from_weight_map,
             convert_background_noise_map_from_inverse_noise_map=convert_background_noise_map_from_inverse_noise_map)

    exposure_time_map = load_exposure_time_map(exposure_time_map_path=exposure_time_map_path,
                           exposure_time_map_hdu=exposure_time_map_hdu,
                           pixel_scale=pixel_scale, shape=image.shape,
                           exposure_time=exposure_time_map_from_single_value,
                           exposure_time_map_from_background_noise_map=exposure_time_map_from_background_noise_map,
                           background_noise_map=background_noise_map)

    poisson_noise_map = load_poisson_noise_map(poisson_noise_map_path=poisson_noise_map_path,
                   poisson_noise_map_hdu=poisson_noise_map_hdu,
                   pixel_scale=pixel_scale,
                   convert_poisson_noise_map_from_weight_map=convert_poisson_noise_map_from_weight_map,
                   convert_poisson_noise_map_from_inverse_noise_map=convert_poisson_noise_map_from_inverse_noise_map,
                   image=image, exposure_time_map=exposure_time_map,
                   poisson_noise_map_from_image=poisson_noise_map_from_image,
                   convert_from_counts=convert_from_counts)

    noise_map = load_noise_map(noise_map_path=noise_map_path, noise_map_hdu=noise_map_hdu, pixel_scale=pixel_scale,
                       image=image, background_noise_map=background_noise_map,
                       exposure_time_map=exposure_time_map,
                       convert_noise_map_from_weight_map=convert_noise_map_from_weight_map,
                       convert_noise_map_from_inverse_noise_map=convert_noise_map_from_inverse_noise_map,
                       noise_map_from_image_and_background_noise_map=noise_map_from_image_and_background_noise_map,
                       convert_from_counts=convert_from_counts)

    psf = load_psf(psf_path=psf_path, psf_hdu=psf_hdu, pixel_scale=pixel_scale, renormalize=renormalize_psf)

    background_sky_map = load_background_sky_map(background_sky_map_path=background_sky_map_path,
                                                 background_sky_map_hdu=background_sky_map_hdu,
                                                 pixel_scale=pixel_scale)

    image = Image(array=image, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map,
                  background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                  exposure_time_map=exposure_time_map, background_sky_map=background_sky_map)

    if resized_image_shape is not None:
        image = image.new_image_with_resized_arrays(new_shape=resized_image_shape,
                                                    new_centre_pixels=resized_image_origin_pixels,
                                                    new_centre_arc_seconds=resized_image_origin_arc_seconds)

    if resized_psf_shape is not None:
        image = image.new_image_with_resized_psf(new_shape=resized_psf_shape)

    if convert_from_counts:
        image = image.new_image_converted_from_counts()

    return image

def load_image(image_path, image_hdu, pixel_scale):
    """Factory for loading the regular from a .fits file

    Parameters
    ----------
    image_path : str
        The image_path and filename of the .fits data containing the regular.
    image_hdu : int
        The image_hdu the regular is contained in the .fits file that *image_path* points too
    pixel_scale : float
        The size of each pixel in arc seconds..
    """
    return ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=image_path, hdu=image_hdu, pixel_scale=pixel_scale)

def load_noise_map(noise_map_path, noise_map_hdu, pixel_scale, image, background_noise_map, exposure_time_map,
                   convert_noise_map_from_weight_map, convert_noise_map_from_inverse_noise_map,
                   noise_map_from_image_and_background_noise_map, convert_from_counts):
    """Factory for loading the noise-map from a .fits file.

    This factory also includes a number of routines for converting the noise-map from from other units (e.g. \
    a weight map) or computing the noise-map from other images (e.g. the imaging data and background noise-map).

    Parameters
    ----------
    noise_map_path : str
        The path and filename of the .fits data containing the noise-map.
    noise_map_hdu : int
        The hdu the noise-map is contained in the .fits file that *noise_map_path* points too.
    pixel_scale : float
        The size of each pixel in arc seconds.
    image : ndarray
        The regular-data, which the noise-map can be calculated using.
    background_noise_map : ndarray
        The background noise-map, which the noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the noise-map can be calculated using.
    convert_noise_map_from_weight_map : bool
        If True, the noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_noise_map_from_inverse_noise_map : bool
        If True, the noise-map loaded from the .fits file is converted from an inverse noise-map to a noise-map (see \
        *NoiseMap.from_inverse_noise_map).
    background_noise_map_path : str
        The path and filename of the .fits data containing the background noise-map.
    background_noise_map_hdu : int
        The hdu the background noise-map is contained in the .fits file that *background_noise_map_path* points too.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    noise_map_from_image_and_background_noise_map : bool
        If True, the noise-map is computed from the observed regular and background noise-map \
        (see NoiseMap.from_image_and_background_noise_map).
    """
    noise_map_options = sum([convert_noise_map_from_weight_map,
                             convert_noise_map_from_inverse_noise_map,
                             noise_map_from_image_and_background_noise_map])

    if noise_map_options > 1:
        raise exc.ImagingException('You have specified more than one method to load the noise map, e.g.:'
                                   'convert_noise_map_from_weight_map | '
                                   'convert_noise_map_from_inverse_noise_map |'
                                   'noise_map_from_image_and_background_noise_map')


    if noise_map_options == 0 and noise_map_path is not None:
        return NoiseMap.from_fits_with_pixel_scale(file_path=noise_map_path, hdu=noise_map_hdu, pixel_scale=pixel_scale)
    elif convert_noise_map_from_weight_map and noise_map_path is not None:
        weight_map = Array.from_fits(file_path=noise_map_path, hdu=noise_map_hdu)
        return NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=pixel_scale)
    elif convert_noise_map_from_inverse_noise_map and noise_map_path is not None:
        inverse_noise_map = Array.from_fits(file_path=noise_map_path, hdu=noise_map_hdu)
        return NoiseMap.from_inverse_noise_map(inverse_noise_map=inverse_noise_map, pixel_scale=pixel_scale)
    elif noise_map_from_image_and_background_noise_map:
        return NoiseMap.from_image_and_background_noise_map(pixel_scale=pixel_scale, image=image,
                                                            background_noise_map=background_noise_map,
                                                            exposure_time_map=exposure_time_map,
                                                            convert_from_counts=convert_from_counts)
    else:
        raise exc.ImagingException('A noise map was not loaded, specify a noise_map_path of option to compute a noise map.')

def load_background_noise_map(background_noise_map_path, background_noise_map_hdu, pixel_scale,
                              convert_background_noise_map_from_weight_map,
                              convert_background_noise_map_from_inverse_noise_map):
    """Factory for loading the background noise-map from a .fits file.

    This factory also includes a number of routines for converting the background noise-map from from other units (e.g. \
    a weight map).

    Parameters
    ----------
    background_noise_map_path : str
        The path and filename of the .fits data containing the background noise-map.
    background_noise_map_hdu : int
        The hdu the background noise-map is contained in the .fits file that *background_noise_map_path* points too.
    pixel_scale : float
        The size of each pixel in arc seconds.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    """
    background_noise_map_options = sum([convert_background_noise_map_from_weight_map,
                                        convert_background_noise_map_from_inverse_noise_map])

    if background_noise_map_options == 0 and background_noise_map_path is not None:
        return NoiseMap.from_fits_with_pixel_scale(file_path=background_noise_map_path, hdu=background_noise_map_hdu, pixel_scale=pixel_scale)
    elif convert_background_noise_map_from_weight_map and background_noise_map_path is not None:
        weight_map = Array.from_fits(file_path=background_noise_map_path, hdu=background_noise_map_hdu)
        return NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=pixel_scale)
    elif convert_background_noise_map_from_inverse_noise_map and background_noise_map_path is not None:
        inverse_noise_map = Array.from_fits(file_path=background_noise_map_path, hdu=background_noise_map_hdu)
        return NoiseMap.from_inverse_noise_map(inverse_noise_map=inverse_noise_map, pixel_scale=pixel_scale)
    else:
        return None

def load_poisson_noise_map(poisson_noise_map_path, poisson_noise_map_hdu, pixel_scale,
                           convert_poisson_noise_map_from_weight_map,
                           convert_poisson_noise_map_from_inverse_noise_map,
                           poisson_noise_map_from_image,
                           image, exposure_time_map, convert_from_counts):
    """Factory for loading the Poisson noise-map from a .fits file.

    This factory also includes a number of routines for converting the Poisson noise-map from from other units (e.g. \
    a weight map) or computing the Poisson noise from other images (e.g. the imaging data).

    Parameters
    ----------
    poisson_noise_map_path : str
        The path and filename of the .fits data containing the Poisson noise-map.
    poisson_noise_map_hdu : int
        The hdu the Poisson noise-map is contained in the .fits file that *poisson_noise_map_path* points too.
    pixel_scale : float
        The size of each pixel in arc seconds.
    convert_poisson_noise_map_from_weight_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_poisson_noise_map_from_inverse_noise_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    poisson_noise_map_from_image : bool
        If True, the Poisson noise-map is estimated using the regular-data.
    image : ndarray
        The regular-data, which the Poisson noise-map can be calculated using.
    background_noise_map : ndarray
        The background noise-map, which the Poisson noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the Poisson noise-map can be calculated using.
    """
    poisson_noise_map_options = sum([convert_poisson_noise_map_from_weight_map,
                                     convert_poisson_noise_map_from_inverse_noise_map,
                                     poisson_noise_map_from_image])

    if poisson_noise_map_options == 0 and poisson_noise_map_path is not None:
        return PoissonNoiseMap.from_fits_with_pixel_scale(file_path=poisson_noise_map_path, hdu=poisson_noise_map_hdu, pixel_scale=pixel_scale)
    elif poisson_noise_map_from_image:
        return PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=pixel_scale, image=image,
                                                                exposure_time_map=exposure_time_map,
                                                                convert_from_counts=convert_from_counts)
    elif convert_poisson_noise_map_from_weight_map and poisson_noise_map_path is not None:
        weight_map = Array.from_fits(file_path=poisson_noise_map_path, hdu=poisson_noise_map_hdu)
        return PoissonNoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=pixel_scale)
    elif convert_poisson_noise_map_from_inverse_noise_map and poisson_noise_map_path is not None:
        inverse_noise_map = Array.from_fits(file_path=poisson_noise_map_path, hdu=poisson_noise_map_hdu)
        return PoissonNoiseMap.from_inverse_noise_map(inverse_noise_map=inverse_noise_map, pixel_scale=pixel_scale)
    else:
        return None

def load_psf(psf_path, psf_hdu, pixel_scale, renormalize=False):
    """Factory for loading the psf from a .fits file.

    Parameters
    ----------
    psf_path : str
        The path and filename of the .fits data containing the PSF.
    psf_hdu : int
        The hdu the psf is contained in the .fits file that *psf_path* points too.
    pixel_scale : float
        The size of each pixel in arc seconds.
    renormalize : bool
        If True, the PSF is renoralized such that all elements sum to 1.0.
    """
    if renormalize:
        return PSF.from_fits_renormalized(file_path=psf_path, hdu=psf_hdu, pixel_scale=pixel_scale)
    if not renormalize:
        return PSF.from_fits_with_scale(file_path=psf_path, hdu=psf_hdu, pixel_scale=pixel_scale)

def load_exposure_time_map(exposure_time_map_path, exposure_time_map_hdu, pixel_scale, shape, exposure_time,
                           exposure_time_map_from_background_noise_map, background_noise_map):
    """Factory for loading the exposure time map from a .fits file.

    This factory also includes a number of routines for computing the exposure-time map from other images \
    (e.g. the background noise-map).

    Parameters
    ----------
    exposure_time_map_path : str
        The path and filename of the .fits data containing the exposure time map.
    exposure_time_map_hdu : int
        The hdu the exposure time map is contained in the .fits file that *exposure_time_map_path* points too.
    pixel_scale : float
        The size of each pixel in arc seconds.
    shape : (int, int)
        The shape of the regular-data, required if a single value is used to calculate the exposure time map.
    exposure_time : float
        The exposure-time used to compute the expsure-time map if only a single value is used.
    exposure_time_map_from_background_noise_map : bool
        If True, the exposure-time map is computed from the background noise map \
        (see *ExposureTimeMap.from_background_noise_map*)
    background_noise_map : ndarray
        The background noise-map, which the Poisson noise-map can be calculated using.
    """
    exposure_time_map_options = sum([exposure_time_map_from_background_noise_map])

    if exposure_time is not None and exposure_time_map_path is not None:
        raise exc.ImagingException('You have supplied both a exposure_time_map_path to an exposure time map and an exposure time. Only'
                                   'one quantity should be supplied.')

    if exposure_time_map_options == 0:

        if exposure_time is not None and exposure_time_map_path is None:
            return ExposureTimeMap.single_value(value=exposure_time, pixel_scale=pixel_scale, shape=shape)
        elif exposure_time is None and exposure_time_map_path is not None:
            return ExposureTimeMap.from_fits_with_pixel_scale(file_path=exposure_time_map_path, hdu=exposure_time_map_hdu, pixel_scale=pixel_scale)

    else:

        if exposure_time_map_from_background_noise_map:
            return ExposureTimeMap.from_exposure_time_and_background_noise_map(pixel_scale=pixel_scale,
                   exposure_time=exposure_time, background_noise_map=background_noise_map)

def load_background_sky_map(background_sky_map_path, background_sky_map_hdu, pixel_scale):
    """Factory for loading the background sky from a .fits file.

    Parameters
    ----------
    backgrond_sky_map_path : str
        The path and filename of the .fits data containing the backgrond sky map.
    backgrond_sky_map_hdu : int
        The hdu the backgrond sky map is contained in the .fits file that *backgrond_sky_map_path* points too.
    image_path : str
        The path and filename of the .fits data containing the regular.
    """
    if background_sky_map_path is not None:
        return ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=background_sky_map_path,
                                                                 hdu=background_sky_map_hdu, pixel_scale=pixel_scale)
    else:
        return None

def output_imaging_to_fits(image, image_path, psf_path, noise_map_path=None, background_noise_map_path=None,
                           poisson_noise_map_path=None, exposure_time_map_path=None, background_sky_map_path=None,
                           overwrite=False):
    array_util.numpy_array_to_fits(array=image, path=image_path, overwrite=overwrite)
    array_util.numpy_array_to_fits(array=image.psf, path=psf_path, overwrite=overwrite)

    if image.noise_map is not None and noise_map_path is not None:
        array_util.numpy_array_to_fits(array=image.noise_map, path=noise_map_path, overwrite=overwrite)

    if image.background_noise_map is not None and background_noise_map_path is not None:
        array_util.numpy_array_to_fits(array=image.background_noise_map, path=background_noise_map_path,
                                       overwrite=overwrite)

    if image.poisson_noise_map is not None and poisson_noise_map_path is not None:
        array_util.numpy_array_to_fits(array=image.poisson_noise_map, path=poisson_noise_map_path,
                                       overwrite=overwrite)

    if image.exposure_time_map is not None and exposure_time_map_path is not None:
        array_util.numpy_array_to_fits(array=image.exposure_time_map, path=exposure_time_map_path,
                                       overwrite=overwrite)

    if image.background_sky_map is not None and background_sky_map_path is not None:
        array_util.numpy_array_to_fits(array=image.background_sky_map, path=background_sky_map_path,
                                       overwrite=overwrite)