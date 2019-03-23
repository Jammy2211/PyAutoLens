import ast
import logging

import numpy as np
import scipy.signal
from astropy import units
from scipy.stats import norm
from skimage.transform import resize, rescale

from autolens import exc
from autolens.data.array.scaled_array import ScaledSquarePixelArray, Array
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util, array_util
from autolens.model.profiles.light_profiles import EllipticalGaussian

logger = logging.getLogger(__name__)


class CCDData(object):

    def __init__(self, image, pixel_scale, psf, noise_map=None, background_noise_map=None, poisson_noise_map=None,
                 exposure_time_map=None, background_sky_map=None, name=None, **kwargs):
        """A collection of 2D CCD data (an image, noise-map, psf, etc.)

        Parameters
        ----------
        image : scaled_array.ScaledArraySquarePixels
            The array of the image data, in units of electrons per second.
        pixel_scale : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the image.
        noise_map : NoiseMap | float | ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        background_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the background sky noise_map,
            preferably in units of electrons per second.
        poisson_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the Poisson counts of the source,
            preferably in units of electrons per second.
        exposure_time_map : scaled_array.ScaledSquarePixelArray
            An array describing the effective exposure time in each ccd pixel.
        background_sky_map : scaled_array.ScaledSquarePixelArray
            An array describing the background sky.
        """
        self.name = name
        self.image = image
        self.pixel_scale = pixel_scale
        self.psf = psf
        self.noise_map = noise_map
        self.background_noise_map = background_noise_map
        self.poisson_noise_map = poisson_noise_map
        self.exposure_time_map = exposure_time_map
        self.background_sky_map = background_sky_map
        self.origin = (0.0, 0.0)

    @property
    def shape(self):
        return self.image.shape

    @classmethod
    def simulate(cls, array, pixel_scale, exposure_time, psf=None, background_sky_level=None,
                 add_noise=False, seed=-1, name=None):

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
                                            add_noise=add_noise, seed=seed, name=name)

    @classmethod
    def simulate_variable_arrays(cls, array, pixel_scale, exposure_time_map, psf=None, background_sky_map=None,
                                 add_noise=True, seed=-1, name=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        array : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and CCD read-out).
        pixel_scale: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        seed: int
            A seed for random noise_maps generation
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

        array = ScaledSquarePixelArray(array=array, pixel_scale=pixel_scale)
        noise_map = NoiseMap(array=noise_map, pixel_scale=pixel_scale)

        if background_noise_map is not None:
            background_noise_map = NoiseMap(array=background_noise_map, pixel_scale=pixel_scale)

        if poisson_noise_map is not None:
            poisson_noise_map = PoissonNoiseMap(array=poisson_noise_map, pixel_scale=pixel_scale)

        return CCDData(array, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map,
                       background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                       exposure_time_map=exposure_time_map, background_sky_map=background_sky_map, name=name)

    @classmethod
    def simulate_to_target_signal_to_noise(cls, array, pixel_scale, target_signal_to_noise, exposure_time_map,
                                           psf=None, background_sky_map=None, seed=-1):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        target_signal_to_noise
        array : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and CCD read-out).
        pixel_scale: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An array representing the effective exposure time of each pixel.
        psf: PSF
            An array describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        seed: int
            A seed for random noise_maps generation
        """

        max_index = np.unravel_index(array.argmax(), array.shape)
        max_image = array[max_index]
        max_effective_exposure_time = exposure_time_map[max_index]
        max_array_counts = np.multiply(max_image, max_effective_exposure_time)
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

    def new_ccd_data_with_binned_up_arrays(self, bin_up_factor):

        image = self.bin_up_scaled_array(scaled_array=self.image, bin_up_factor=bin_up_factor, method='mean')
        psf = self.psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=1.0 / bin_up_factor, renormalize=True)
        noise_map = self.bin_up_scaled_array(scaled_array=self.noise_map, bin_up_factor=bin_up_factor,
                                             method='quadrature')
        background_noise_map = self.bin_up_scaled_array(scaled_array=self.background_noise_map,
                                                        bin_up_factor=bin_up_factor, method='quadrature')
        poisson_noise_map = self.bin_up_scaled_array(scaled_array=self.poisson_noise_map,
                                                     bin_up_factor=bin_up_factor, method='quadrature')
        exposure_time_map = self.bin_up_scaled_array(scaled_array=self.exposure_time_map,
                                                     bin_up_factor=bin_up_factor, method='sum')
        background_sky_map = self.bin_up_scaled_array(scaled_array=self.background_sky_map,
                                                      bin_up_factor=bin_up_factor, method='mean')

        return CCDData(image=image, pixel_scale=self.pixel_scale * bin_up_factor, psf=psf, noise_map=noise_map,
                       background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                       exposure_time_map=exposure_time_map, background_sky_map=background_sky_map, name=self.name)

    def new_ccd_data_with_resized_arrays(self, new_shape, new_centre_pixels=None, new_centre_arcsec=None):

        image = self.resize_scaled_array(scaled_array=self.image, new_shape=new_shape,
                                         new_centre_pixels=new_centre_pixels,
                                         new_centre_arcsec=new_centre_arcsec)

        noise_map = self.resize_scaled_array(scaled_array=self.noise_map, new_shape=new_shape,
                                             new_centre_pixels=new_centre_pixels,
                                             new_centre_arcsec=new_centre_arcsec)

        background_noise_map = self.resize_scaled_array(scaled_array=self.background_noise_map, new_shape=new_shape,
                                                        new_centre_pixels=new_centre_pixels,
                                                        new_centre_arcsec=new_centre_arcsec)

        poisson_noise_map = self.resize_scaled_array(scaled_array=self.poisson_noise_map, new_shape=new_shape,
                                                     new_centre_pixels=new_centre_pixels,
                                                     new_centre_arcsec=new_centre_arcsec)

        exposure_time_map = self.resize_scaled_array(scaled_array=self.exposure_time_map, new_shape=new_shape,
                                                     new_centre_pixels=new_centre_pixels,
                                                     new_centre_arcsec=new_centre_arcsec)

        background_sky_map = self.resize_scaled_array(scaled_array=self.background_sky_map, new_shape=new_shape,
                                                      new_centre_pixels=new_centre_pixels,
                                                      new_centre_arcsec=new_centre_arcsec)

        return CCDData(image=image, pixel_scale=self.pixel_scale, psf=self.psf, noise_map=noise_map,
                       background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                       exposure_time_map=exposure_time_map, background_sky_map=background_sky_map, name=self.name)

    def new_ccd_data_with_resized_psf(self, new_shape):
        psf = self.resize_scaled_array(scaled_array=self.psf, new_shape=new_shape)
        return CCDData(image=self.image, pixel_scale=self.pixel_scale, psf=psf, noise_map=self.noise_map,
                       background_noise_map=self.background_noise_map, poisson_noise_map=self.poisson_noise_map,
                       exposure_time_map=self.exposure_time_map, background_sky_map=self.background_sky_map,
                       name=self.name)

    @staticmethod
    def bin_up_scaled_array(scaled_array, bin_up_factor, method):
        if scaled_array is not None:
            return scaled_array.binned_up_array_from_array(bin_up_factor=bin_up_factor, method=method)
        else:
            return None

    @staticmethod
    def resize_scaled_array(scaled_array, new_shape, new_centre_pixels=None, new_centre_arcsec=None):
        if scaled_array is not None:
            return scaled_array.resized_scaled_array_from_array(new_shape=new_shape,
                                                                new_centre_pixels=new_centre_pixels,
                                                                new_centre_arcsec=new_centre_arcsec)
        else:
            return None

    def new_ccd_data_with_modified_image(self, modified_image):

        return CCDData(image=modified_image, pixel_scale=self.pixel_scale, psf=self.psf,
                       noise_map=self.noise_map, background_noise_map=self.background_noise_map,
                       poisson_noise_map=self.poisson_noise_map, exposure_time_map=self.exposure_time_map,
                       background_sky_map=self.background_sky_map, name=self.name)

    def new_ccd_data_with_poisson_noise_added(self, seed=-1):

        image_with_sky = self.image + self.background_sky_map

        image_with_sky_and_noise = image_with_sky + generate_poisson_noise(image=image_with_sky,
                                                                           exposure_time_map=self.exposure_time_map,
                                                                           seed=seed)

        image_with_noise = image_with_sky_and_noise - self.background_sky_map

        return CCDData(image=image_with_noise, pixel_scale=self.pixel_scale, psf=self.psf,
                       noise_map=self.noise_map, background_noise_map=self.background_noise_map,
                       poisson_noise_map=self.poisson_noise_map, name=self.name)

    def new_ccd_data_converted_from_electrons(self):

        image = self.array_from_counts_to_electrons_per_second(array=self.image)
        noise_map = self.array_from_counts_to_electrons_per_second(array=self.noise_map)
        background_noise_map = self.array_from_counts_to_electrons_per_second(array=self.background_noise_map)
        poisson_noise_map = self.array_from_counts_to_electrons_per_second(array=self.poisson_noise_map)
        background_sky_map = self.array_from_counts_to_electrons_per_second(array=self.background_sky_map)

        return CCDData(image=image, pixel_scale=self.pixel_scale, psf=self.psf, noise_map=noise_map,
                       background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                       exposure_time_map=self.exposure_time_map, background_sky_map=background_sky_map,
                       name=self.name)

    def new_ccd_data_converted_from_adus(self, gain):

        image = self.array_from_adus_to_electrons_per_second(array=self.image, gain=gain)
        noise_map = self.array_from_adus_to_electrons_per_second(array=self.noise_map, gain=gain)
        background_noise_map = self.array_from_adus_to_electrons_per_second(array=self.background_noise_map, gain=gain)
        poisson_noise_map = self.array_from_adus_to_electrons_per_second(array=self.poisson_noise_map, gain=gain)
        background_sky_map = self.array_from_adus_to_electrons_per_second(array=self.background_sky_map, gain=gain)

        return CCDData(image=image, pixel_scale=self.pixel_scale, psf=self.psf, noise_map=noise_map,
                       background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                       exposure_time_map=self.exposure_time_map, background_sky_map=background_sky_map,
                       name=self.name)

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.image, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)

    @property
    def absolute_signal_to_noise_map(self):
        """The estimated absolute_signal-to-noise_maps mappers of the image."""
        return np.divide(np.abs(self.image), self.noise_map)

    @property
    def absolute_signal_to_noise_max(self):
        """The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self):
        """The potential chi-squared map of the ccd data. This represents how much each pixel can contribute to \
        the chi-squared map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return np.square(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_max(self):
        """The maximum value of the potential chi-squared map"""
        return np.max(self.potential_chi_squared_map)

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

    def array_from_electrons_per_second_to_counts(self, array):
        """
        For an array (in electrons per second) and an exposure time mappers, return an array in units counts.

        Parameters
        ----------
        array : ndarray
            The array the values are to be converted from electrons per seconds to counts.
        """
        return np.multiply(array, self.exposure_time_map)

    def array_from_counts_to_electrons_per_second(self, array):
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

    def array_from_adus_to_electrons_per_second(self, array, gain):
        """
        For an array (in counts) and an exposure time mappers, convert the array to units electrons per second

        Parameters
        ----------
        array : ndarray
            The array the values are to be converted from counts to electrons per second.
        """
        if array is not None:
            return np.divide(gain * array, self.exposure_time_map)
        else:
            return None

    @property
    def image_counts(self):
        """The image in units of counts."""
        return self.array_from_electrons_per_second_to_counts(self.image)

    @property
    def background_noise_map_counts(self):
        """ The background noise_maps mappers in units of counts."""
        return self.array_from_electrons_per_second_to_counts(self.background_noise_map)

    @property
    def estimated_noise_map_counts(self):
        """ The estimated noise_maps mappers of the image (using its background noise_maps mappers and image values
        in counts) in counts.
        """
        return np.sqrt((np.abs(self.image_counts) + np.square(self.background_noise_map_counts)))

    @property
    def estimated_noise_map(self):
        """ The estimated noise_maps mappers of the image (using its background noise_maps mappers and image values
        in counts) in electrons per second.
        """
        return self.array_from_counts_to_electrons_per_second(self.estimated_noise_map_counts)

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
            top_edge = self.image[edge_no, edge_no:self.image.shape[1] - edge_no]
            bottom_edge = self.image[self.image.shape[0] - 1 - edge_no, edge_no:self.image.shape[1] - edge_no]
            left_edge = self.image[edge_no + 1:self.image.shape[0] - 1 - edge_no, edge_no]
            right_edge = self.image[edge_no + 1:self.image.shape[0] - 1 - edge_no, self.image.shape[1] - 1 - edge_no]

            edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

        return norm.fit(edges)[1]

    def __array_finalize__(self, obj):
        if isinstance(obj, CCDData):
            try:
                self.image = obj.image
                self.pixel_scale = obj.pixel_scale
                self.psf = obj.psf
                self.noise_map = obj.noise_map
                self.background_noise_map = obj.background_noise_map
                self.poisson_noise_map = obj.poisson_noise_map
                self.exposure_time_map = obj.exposure_time_map
                self.background_sky_map = obj.background_sky_map
                self.origin = obj.origin
            except AttributeError:
                logger.debug("Original object in CCD.__array_finalize__ missing one or more attributes")


class NoiseMap(ScaledSquarePixelArray):

    @classmethod
    def from_weight_map(cls, pixel_scale, weight_map):
        """Setup the noise-map from a weight map, which is a form of noise-map that comes via HST image-reduction and \
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
        comes via HST image-reduction and the software package MultiDrizzle.

        The variance in each pixel is computed as:

        Variance = 1.0 / inverse_std_map.

        The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
        the analysis.

        Parameters
        -----------
        pixel_scale : float
            The size of each pixel in arc seconds.
        inverse_noise_map : ndarray
            The inverse noise_map value of each pixel which is converted to a variance.
        """
        noise_map = 1.0 / inverse_noise_map
        return NoiseMap(array=noise_map, pixel_scale=pixel_scale)

    @classmethod
    def from_image_and_background_noise_map(cls, pixel_scale, image, background_noise_map, exposure_time_map, gain=None,
                                            convert_from_electrons=False, convert_from_adus=False):

        if not convert_from_electrons and not convert_from_adus:
            return NoiseMap(array=np.sqrt(np.abs(((background_noise_map) * exposure_time_map) ** 2.0 +
                                                 (image) * exposure_time_map)) / (exposure_time_map),
                            pixel_scale=pixel_scale)
        elif convert_from_electrons:
            return NoiseMap(array=np.sqrt(np.abs(background_noise_map ** 2.0 + image)), pixel_scale=pixel_scale)
        elif convert_from_adus:
            return NoiseMap(array=np.sqrt(np.abs((gain * background_noise_map) ** 2.0 + gain * image)) / gain,
                            pixel_scale=pixel_scale)


class PoissonNoiseMap(NoiseMap):

    @classmethod
    def from_image_and_exposure_time_map(cls, pixel_scale, image, exposure_time_map, gain=None,
                                         convert_from_electrons=False, convert_from_adus=False):
        if not convert_from_electrons and not convert_from_adus:
            return PoissonNoiseMap(array=np.sqrt(np.abs(image) * exposure_time_map) / (exposure_time_map),
                                   pixel_scale=pixel_scale)
        elif convert_from_electrons:
            return PoissonNoiseMap(array=np.sqrt(np.abs(image)), pixel_scale=pixel_scale)
        elif convert_from_adus:
            return NoiseMap(array=np.sqrt(gain * np.abs(image)) / gain, pixel_scale=pixel_scale)


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
            self[:, :] = np.divide(self, np.sum(self))

    @classmethod
    def simulate_as_gaussian(cls, shape, pixel_scale, sigma, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0):
        """Simulate the PSF as an elliptical Gaussian profile."""
        from autolens.model.profiles.light_profiles import EllipticalGaussian
        gaussian = EllipticalGaussian(centre=centre, axis_ratio=axis_ratio, phi=phi, intensity=1.0, sigma=sigma)
        grid_1d = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full(shape, False),
                                                                                     pixel_scales=(
                                                                                         pixel_scale, pixel_scale))
        gaussian_1d = gaussian.intensities_from_grid(grid=grid_1d)
        gaussian_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=gaussian_1d,
                                                                                             shape=shape)
        return PSF(array=gaussian_2d, pixel_scale=pixel_scale, renormalize=True)

    @classmethod
    def simulate_as_gaussian_via_alma_fits_header_parameters(cls, shape, pixel_scale, y_stddev, x_stddev, theta,
                                                             centre=(0.0, 0.0)):

        x_stddev = x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        y_stddev = y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        axis_ratio = x_stddev / y_stddev

        gaussian = EllipticalGaussian(centre=centre, axis_ratio=axis_ratio, phi=90.0 - theta, intensity=1.0,
                                      sigma=y_stddev)

        grid_1d = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full(shape, False),
                                                                                     pixel_scales=(
                                                                                         pixel_scale, pixel_scale))
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
        psf[:, :] = np.divide(psf, np.sum(psf))
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
        return cls(array=array_util.numpy_array_2d_from_fits(file_path, hdu), pixel_scale=pixel_scale)

    def new_psf_with_rescaled_odd_dimensioned_array(self, rescale_factor, renormalize=True):
        psf_rescaled = rescale(self, rescale_factor, anti_aliasing=False, mode='constant', multichannel=False)

        if psf_rescaled.shape[0] % 2 == 0 and psf_rescaled.shape[1] % 2 == 0:
            psf_rescaled = resize(psf_rescaled, output_shape=(psf_rescaled.shape[0] + 1, psf_rescaled.shape[1] + 1),
                                  anti_aliasing=False, mode='constant')
        elif psf_rescaled.shape[0] % 2 == 0 and psf_rescaled.shape[1] % 2 != 0:
            psf_rescaled = resize(psf_rescaled, output_shape=(psf_rescaled.shape[0] + 1, psf_rescaled.shape[1]),
                                  anti_aliasing=False, mode='constant')
        elif psf_rescaled.shape[0] % 2 != 0 and psf_rescaled.shape[1] % 2 == 0:
            psf_rescaled = resize(psf_rescaled, output_shape=(psf_rescaled.shape[0], psf_rescaled.shape[1] + 1),
                                  anti_aliasing=False, mode='constant')

        pixel_scale_factors = (self.shape[0] / psf_rescaled.shape[0], self.shape[1] / psf_rescaled.shape[1])
        pixel_scale = (self.pixel_scale * pixel_scale_factors[0], self.pixel_scale * pixel_scale_factors[1])
        return PSF(array=psf_rescaled, pixel_scale=np.max(pixel_scale), renormalize=renormalize)

    def new_psf_with_renormalized_array(self):
        """Renormalize the PSF such that its data_vector values sum to unity."""
        return PSF(array=self, pixel_scale=self.pixel_scale, renormalize=True)

    def convolve(self, array):
        """
        Convolve an array with this PSF

        Parameters
        ----------
        image : ndarray
            An array representing the image the PSF is convolved with.

        Returns
        -------
        convolved_image : ndarray
            An array representing the image after convolution.

        Raises
        ------
        KernelException if either PSF psf dimension is odd
        """
        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise exc.KernelException("PSF Kernel must be odd")

        return scipy.signal.convolve2d(array, self, mode='same')


class ExposureTimeMap(ScaledSquarePixelArray):

    @classmethod
    def from_exposure_time_and_inverse_noise_map(cls, pixel_scale, exposure_time, inverse_noise_map):
        relative_background_noise_map = inverse_noise_map / np.max(inverse_noise_map)
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
    Generate a two-dimensional poisson noise_maps-mappers from an image.

    Values are computed from a Poisson distribution using the image's input values in units of counts.

    Parameters
    ----------
    image : ndarray
        The 2D image, whose values in counts are used to draw Poisson noise_maps values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed : int
        The seed of the random number generator, used for the random noise_maps maps.

    Returns
    -------
    poisson_noise_map: ndarray
        An array describing simulated poisson noise_maps
    """
    setup_random_seed(seed)
    image_counts = np.multiply(image, exposure_time_map)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time_map)


def load_ccd_data_from_fits(image_path, pixel_scale, image_hdu=0,
                            resized_ccd_shape=None, resized_ccd_origin_pixels=None,
                            resized_ccd_origin_arcsec=None,
                            psf_path=None, psf_hdu=0, resized_psf_shape=None, renormalize_psf=True,
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
                            exposure_time_map_from_inverse_noise_map=False,
                            background_sky_map_path=None, background_sky_map_hdu=0,
                            convert_from_electrons=False,
                            gain=None, convert_from_adus=False, lens_name=None):
    """Factory for loading the ccd data from .fits files, as well as computing properties like the noise-map,
    exposure-time map, etc. from the ccd-data.

    This factory also includes a number of routines for converting the ccd-data from units not supported by PyAutoLens \
    (e.g. adus, electrons) to electrons per second.

    Parameters
    ----------
    lens_name
    image_path : str
        The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
    pixel_scale : float
        The size of each pixel in arc seconds.
    image_hdu : int
        The hdu the image is contained in the .fits file specified by *image_path*.        
    image_hdu : int
        The hdu the image is contained in the .fits file that *image_path* points too.
    resized_ccd_shape : (int, int) | None
        If input, the ccd arrays that are image sized, e.g. the image, noise-maps) are resized to these dimensions.
    resized_ccd_origin_pixels : (int, int) | None
        If the ccd arrays are resized, this defines a new origin (in pixels) around which recentering occurs.
    resized_ccd_origin_arcsec : (float, float) | None
        If the ccd arrays are resized, this defines a new origin (in arc-seconds) around which recentering occurs.
    psf_path : str
        The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits')        
    psf_hdu : int
        The hdu the psf is contained in the .fits file specified by *psf_path*.
    resized_psf_shape : (int, int) | None
        If input, the psf is resized to these dimensions.
    renormalize_psf : bool
        If True, the PSF is renoralized such that all elements sum to 1.0.
    noise_map_path : str
        The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')        
    noise_map_hdu : int
        The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
    noise_map_from_image_and_background_noise_map : bool
        If True, the noise-map is computed from the observed image and background noise-map \
        (see NoiseMap.from_image_and_background_noise_map).
    convert_noise_map_from_weight_map : bool
        If True, the noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_noise_map_from_inverse_noise_map : bool
        If True, the noise-map loaded from the .fits file is converted from an inverse noise-map to a noise-map (see \
        *NoiseMap.from_inverse_noise_map).
    background_noise_map_path : str
        The path to the background_noise_map .fits file containing the background noise-map \ 
        (e.g. '/path/to/background_noise_map.fits')        
    background_noise_map_hdu : int
        The hdu the background_noise_map is contained in the .fits file specified by *background_noise_map_path*.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    poisson_noise_map_path : str
        The path to the poisson_noise_map .fits file containing the Poisson noise-map \
         (e.g. '/path/to/poisson_noise_map.fits')        
    poisson_noise_map_hdu : int
        The hdu the poisson_noise_map is contained in the .fits file specified by *poisson_noise_map_path*.
    poisson_noise_map_from_image : bool
        If True, the Poisson noise-map is estimated using the image.
    convert_poisson_noise_map_from_weight_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_poisson_noise_map_from_inverse_noise_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    exposure_time_map_path : str
        The path to the exposure_time_map .fits file containing the exposure time map \ 
        (e.g. '/path/to/exposure_time_map.fits')        
    exposure_time_map_hdu : int
        The hdu the exposure_time_map is contained in the .fits file specified by *exposure_time_map_path*.
    exposure_time_map_from_single_value : float
        The exposure time of the ccd imaging, which is used to compute the exposure-time map as a single value \
        (see *ExposureTimeMap.from_single_value*).
    exposure_time_map_from_inverse_noise_map : bool
        If True, the exposure-time map is computed from the background noise_map map \
        (see *ExposureTimeMap.from_background_noise_map*)
    background_sky_map_path : str
        The path to the background_sky_map .fits file containing the background sky map \
        (e.g. '/path/to/background_sky_map.fits').
    background_sky_map_hdu : int
        The hdu the background_sky_map is contained in the .fits file specified by *background_sky_map_path*.
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in units of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in units of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """

    image = load_image(image_path=image_path, image_hdu=image_hdu, pixel_scale=pixel_scale)

    background_noise_map = load_background_noise_map(background_noise_map_path=background_noise_map_path,
                                                     background_noise_map_hdu=background_noise_map_hdu,
                                                     pixel_scale=pixel_scale,
                                                     convert_background_noise_map_from_weight_map=convert_background_noise_map_from_weight_map,
                                                     convert_background_noise_map_from_inverse_noise_map=convert_background_noise_map_from_inverse_noise_map)

    if background_noise_map is not None:
        inverse_noise_map = 1.0 / background_noise_map
    else:
        inverse_noise_map = None

    exposure_time_map = load_exposure_time_map(exposure_time_map_path=exposure_time_map_path,
                                               exposure_time_map_hdu=exposure_time_map_hdu,
                                               pixel_scale=pixel_scale, shape=image.shape,
                                               exposure_time=exposure_time_map_from_single_value,
                                               exposure_time_map_from_inverse_noise_map=exposure_time_map_from_inverse_noise_map,
                                               inverse_noise_map=inverse_noise_map)

    poisson_noise_map = load_poisson_noise_map(poisson_noise_map_path=poisson_noise_map_path,
                                               poisson_noise_map_hdu=poisson_noise_map_hdu,
                                               pixel_scale=pixel_scale,
                                               convert_poisson_noise_map_from_weight_map=convert_poisson_noise_map_from_weight_map,
                                               convert_poisson_noise_map_from_inverse_noise_map=convert_poisson_noise_map_from_inverse_noise_map,
                                               image=image, exposure_time_map=exposure_time_map,
                                               poisson_noise_map_from_image=poisson_noise_map_from_image,
                                               convert_from_electrons=convert_from_electrons, gain=gain,
                                               convert_from_adus=convert_from_adus)

    noise_map = load_noise_map(noise_map_path=noise_map_path, noise_map_hdu=noise_map_hdu, pixel_scale=pixel_scale,
                               image=image, background_noise_map=background_noise_map,
                               exposure_time_map=exposure_time_map,
                               convert_noise_map_from_weight_map=convert_noise_map_from_weight_map,
                               convert_noise_map_from_inverse_noise_map=convert_noise_map_from_inverse_noise_map,
                               noise_map_from_image_and_background_noise_map=noise_map_from_image_and_background_noise_map,
                               convert_from_electrons=convert_from_electrons, gain=gain,
                               convert_from_adus=convert_from_adus)

    psf = load_psf(psf_path=psf_path, psf_hdu=psf_hdu, pixel_scale=pixel_scale, renormalize=renormalize_psf)

    background_sky_map = load_background_sky_map(background_sky_map_path=background_sky_map_path,
                                                 background_sky_map_hdu=background_sky_map_hdu,
                                                 pixel_scale=pixel_scale)

    image = CCDData(image=image, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map,
                    background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                    exposure_time_map=exposure_time_map, background_sky_map=background_sky_map, gain=gain,
                    name=lens_name)

    if resized_ccd_shape is not None:
        image = image.new_ccd_data_with_resized_arrays(new_shape=resized_ccd_shape,
                                                       new_centre_pixels=resized_ccd_origin_pixels,
                                                       new_centre_arcsec=resized_ccd_origin_arcsec)

    if resized_psf_shape is not None:
        image = image.new_ccd_data_with_resized_psf(new_shape=resized_psf_shape)

    if convert_from_electrons:
        image = image.new_ccd_data_converted_from_electrons()
    elif convert_from_adus:
        image = image.new_ccd_data_converted_from_adus(gain=gain)

    return image


def load_image(image_path, image_hdu, pixel_scale):
    """Factory for loading the image from a .fits file

    Parameters
    ----------
    image_path : str
        The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
    image_hdu : int
        The hdu the image is contained in the .fits file specified by *image_path*.
    pixel_scale : float
        The size of each pixel in arc seconds..
    """
    return ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=image_path, hdu=image_hdu,
                                                             pixel_scale=pixel_scale)


def load_noise_map(noise_map_path, noise_map_hdu, pixel_scale, image, background_noise_map, exposure_time_map,
                   convert_noise_map_from_weight_map, convert_noise_map_from_inverse_noise_map,
                   noise_map_from_image_and_background_noise_map, convert_from_electrons, gain, convert_from_adus):
    """Factory for loading the noise-map from a .fits file.

    This factory also includes a number of routines for converting the noise-map from from other units (e.g. \
    a weight map) or computing the noise-map from other unblurred_image_1d (e.g. the ccd image and background noise-map).

    Parameters
    ----------
    noise_map_path : str
        The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
    noise_map_hdu : int
        The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
    pixel_scale : float
        The size of each pixel in arc seconds.
    image : ndarray
        The image-image, which the noise-map can be calculated using.
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
        The path and filename of the .fits image containing the background noise-map.
    background_noise_map_hdu : int
        The hdu the background noise-map is contained in the .fits file that *background_noise_map_path* points too.
    convert_background_noise_map_from_weight_map : bool
        If True, the bacground noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_background_noise_map_from_inverse_noise_map : bool
        If True, the background noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    noise_map_from_image_and_background_noise_map : bool
        If True, the noise-map is computed from the observed image and background noise-map \
        (see NoiseMap.from_image_and_background_noise_map).
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in units of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in units of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """
    noise_map_options = sum([convert_noise_map_from_weight_map,
                             convert_noise_map_from_inverse_noise_map,
                             noise_map_from_image_and_background_noise_map])

    if noise_map_options > 1:
        raise exc.ImagingException('You have specified more than one method to load the noise_map map, e.g.:'
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

        if background_noise_map is None:
            raise exc.ImagingException('Cannot compute the noise-map from the image and background noise_map map if a '
                                       'background noise_map map is not supplied.')

        if not (convert_from_electrons or convert_from_adus) and exposure_time_map is None:
            raise exc.ImagingException('Cannot compute the noise-map from the image and background noise_map map if an '
                                       'exposure-time (or exposure time map) is not supplied to convert to adus')

        if convert_from_adus and gain is None:
            raise exc.ImagingException('Cannot compute the noise-map from the image and background noise_map map if a'
                                       'gain is not supplied to convert from adus')

        return NoiseMap.from_image_and_background_noise_map(pixel_scale=pixel_scale, image=image,
                                                            background_noise_map=background_noise_map,
                                                            exposure_time_map=exposure_time_map,
                                                            convert_from_electrons=convert_from_electrons,
                                                            gain=gain, convert_from_adus=convert_from_adus)
    else:
        raise exc.ImagingException(
            'A noise_map map was not loaded, specify a noise_map_path or option to compute a noise_map map.')


def load_background_noise_map(background_noise_map_path, background_noise_map_hdu, pixel_scale,
                              convert_background_noise_map_from_weight_map,
                              convert_background_noise_map_from_inverse_noise_map):
    """Factory for loading the background noise-map from a .fits file.

    This factory also includes a number of routines for converting the background noise-map from from other units (e.g. \
    a weight map).

    Parameters
    ----------
    background_noise_map_path : str
        The path to the background_noise_map .fits file containing the background noise-map \
        (e.g. '/path/to/background_noise_map.fits')
    background_noise_map_hdu : int
        The hdu the background_noise_map is contained in the .fits file specified by *background_noise_map_path*.
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
        return NoiseMap.from_fits_with_pixel_scale(file_path=background_noise_map_path, hdu=background_noise_map_hdu,
                                                   pixel_scale=pixel_scale)
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
                           image, exposure_time_map, convert_from_electrons, gain, convert_from_adus):
    """Factory for loading the Poisson noise-map from a .fits file.

    This factory also includes a number of routines for converting the Poisson noise-map from from other units (e.g. \
    a weight map) or computing the Poisson noise_map from other unblurred_image_1d (e.g. the ccd image).

    Parameters
    ----------
    poisson_noise_map_path : str
        The path to the poisson_noise_map .fits file containing the Poisson noise-map \
         (e.g. '/path/to/poisson_noise_map.fits')
    poisson_noise_map_hdu : int
        The hdu the poisson_noise_map is contained in the .fits file specified by *poisson_noise_map_path*.
    pixel_scale : float
        The size of each pixel in arc seconds.
    convert_poisson_noise_map_from_weight_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from a weight-map to a noise-map (see \
        *NoiseMap.from_weight_map).
    convert_poisson_noise_map_from_inverse_noise_map : bool
        If True, the Poisson noise-map loaded from the .fits file is converted from an inverse noise-map to a \
        noise-map (see *NoiseMap.from_inverse_noise_map).
    poisson_noise_map_from_image : bool
        If True, the Poisson noise-map is estimated using the image.
    image : ndarray
        The image, which the Poisson noise-map can be calculated using.
    background_noise_map : ndarray
        The background noise-map, which the Poisson noise-map can be calculated using.
    exposure_time_map : ndarray
        The exposure-time map, which the Poisson noise-map can be calculated using.
    convert_from_electrons : bool
        If True, the input unblurred_image_1d are in units of electrons and all converted to electrons / second using the exposure \
        time map.
    gain : float
        The image gain, used for convert from ADUs.
    convert_from_adus : bool
        If True, the input unblurred_image_1d are in units of adus and all converted to electrons / second using the exposure \
        time map and gain.
    """
    poisson_noise_map_options = sum([convert_poisson_noise_map_from_weight_map,
                                     convert_poisson_noise_map_from_inverse_noise_map,
                                     poisson_noise_map_from_image])

    if poisson_noise_map_options == 0 and poisson_noise_map_path is not None:
        return PoissonNoiseMap.from_fits_with_pixel_scale(file_path=poisson_noise_map_path, hdu=poisson_noise_map_hdu,
                                                          pixel_scale=pixel_scale)
    elif poisson_noise_map_from_image:

        if not (convert_from_electrons or convert_from_adus) and exposure_time_map is None:
            raise exc.ImagingException('Cannot compute the Poisson noise-map from the image if an '
                                       'exposure-time (or exposure time map) is not supplied to convert to adus')

        if convert_from_adus and gain is None:
            raise exc.ImagingException('Cannot compute the Poisson noise-map from the image if a'
                                       'gain is not supplied to convert from adus')

        return PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=pixel_scale, image=image,
                                                                exposure_time_map=exposure_time_map,
                                                                convert_from_electrons=convert_from_electrons,
                                                                gain=gain,
                                                                convert_from_adus=convert_from_adus)

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
        The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits')
    psf_hdu : int
        The hdu the psf is contained in the .fits file specified by *psf_path*.
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
                           exposure_time_map_from_inverse_noise_map, inverse_noise_map):
    """Factory for loading the exposure time map from a .fits file.

    This factory also includes a number of routines for computing the exposure-time map from other unblurred_image_1d \
    (e.g. the background noise-map).

    Parameters
    ----------
    exposure_time_map_path : str
        The path to the exposure_time_map .fits file containing the exposure time map \
        (e.g. '/path/to/exposure_time_map.fits')
    exposure_time_map_hdu : int
        The hdu the exposure_time_map is contained in the .fits file specified by *exposure_time_map_path*.
    pixel_scale : float
        The size of each pixel in arc seconds.
    shape : (int, int)
        The shape of the image, required if a single value is used to calculate the exposure time map.
    exposure_time : float
        The exposure-time used to compute the expsure-time map if only a single value is used.
    exposure_time_map_from_inverse_noise_map : bool
        If True, the exposure-time map is computed from the background noise_map map \
        (see *ExposureTimeMap.from_background_noise_map*)
    inverse_noise_map : ndarray
        The background noise-map, which the Poisson noise-map can be calculated using.
    """
    exposure_time_map_options = sum([exposure_time_map_from_inverse_noise_map])

    if exposure_time is not None and exposure_time_map_path is not None:
        raise exc.ImagingException(
            'You have supplied both a exposure_time_map_path to an exposure time map and an exposure time. Only'
            'one quantity should be supplied.')

    if exposure_time_map_options == 0:

        if exposure_time is not None and exposure_time_map_path is None:
            return ExposureTimeMap.single_value(value=exposure_time, pixel_scale=pixel_scale, shape=shape)
        elif exposure_time is None and exposure_time_map_path is not None:
            return ExposureTimeMap.from_fits_with_pixel_scale(file_path=exposure_time_map_path,
                                                              hdu=exposure_time_map_hdu, pixel_scale=pixel_scale)

    else:

        if exposure_time_map_from_inverse_noise_map:
            return ExposureTimeMap.from_exposure_time_and_inverse_noise_map(pixel_scale=pixel_scale,
                                                                            exposure_time=exposure_time,
                                                                            inverse_noise_map=inverse_noise_map)


def load_background_sky_map(background_sky_map_path, background_sky_map_hdu, pixel_scale):
    """Factory for loading the background sky from a .fits file.

    Parameters
    ----------
    background_sky_map_path : str
        The path to the background_sky_map .fits file containing the background sky map \
        (e.g. '/path/to/background_sky_map.fits').
    background_sky_map_hdu : int
        The hdu the background_sky_map is contained in the .fits file specified by *background_sky_map_path*.
    pixel_scale : float
        The size of each pixel in arc seconds.
    """
    if background_sky_map_path is not None:
        return ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=background_sky_map_path,
                                                                 hdu=background_sky_map_hdu, pixel_scale=pixel_scale)
    else:
        return None


def output_ccd_data_to_fits(ccd_data, image_path, psf_path, noise_map_path=None, background_noise_map_path=None,
                            poisson_noise_map_path=None, exposure_time_map_path=None, background_sky_map_path=None,
                            overwrite=False):
    array_util.numpy_array_2d_to_fits(array_2d=ccd_data.image, file_path=image_path, overwrite=overwrite)
    array_util.numpy_array_2d_to_fits(array_2d=ccd_data.psf, file_path=psf_path, overwrite=overwrite)

    if ccd_data.noise_map is not None and noise_map_path is not None:
        array_util.numpy_array_2d_to_fits(array_2d=ccd_data.noise_map, file_path=noise_map_path, overwrite=overwrite)

    if ccd_data.background_noise_map is not None and background_noise_map_path is not None:
        array_util.numpy_array_2d_to_fits(array_2d=ccd_data.background_noise_map, file_path=background_noise_map_path,
                                          overwrite=overwrite)

    if ccd_data.poisson_noise_map is not None and poisson_noise_map_path is not None:
        array_util.numpy_array_2d_to_fits(array_2d=ccd_data.poisson_noise_map, file_path=poisson_noise_map_path,
                                          overwrite=overwrite)

    if ccd_data.exposure_time_map is not None and exposure_time_map_path is not None:
        array_util.numpy_array_2d_to_fits(array_2d=ccd_data.exposure_time_map, file_path=exposure_time_map_path,
                                          overwrite=overwrite)

    if ccd_data.background_sky_map is not None and background_sky_map_path is not None:
        array_util.numpy_array_2d_to_fits(array_2d=ccd_data.background_sky_map, file_path=background_sky_map_path,
                                          overwrite=overwrite)


def load_positions(positions_path):
    """Load the positions of an image.

    Positions correspond to a set of pixels in the lensed source galaxy that are anticipated to come from the same \
    multiply-imaged region of the source-plane. Mass models which do not trace the pixels within a threshold value of \
    one another are resampled during the non-linear search.

    Positions are stored in a .dat file, where each line of the file gives a list of list of (y,x) positions which \
    correspond to the same region of the source-plane. Thus, multiple source-plane regions can be input over multiple \
    lines of the same positions file.

    Parameters
    ----------
    positions_path : str
        The path to the positions .dat file containing the positions (e.g. '/path/to/positions.dat')
    """
    with open(positions_path) as f:
        position_string = f.readlines()

    positions = []

    for line in position_string:
        position_list = ast.literal_eval(line)
        positions.append(position_list)

    return positions


def output_positions(positions, positions_path):
    """Output the positions of an image to a positions.dat file.

    Positions correspond to a set of pixels in the lensed source galaxy that are anticipated to come from the same \
    multiply-imaged region of the source-plane. Mass models which do not trace the pixels within a threshold value of \
    one another are resampled during the non-linear search.

    Positions are stored in a .dat file, where each line of the file gives a list of list of (y,x) positions which \
    correspond to the same region of the source-plane. Thus, multiple source-plane regions can be input over multiple \
    lines of the same positions file.

    Parameters
    ----------
    positions : [[[]]]
        The lists of positions (e.g. [[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
    positions_path : str
        The path to the positions .dat file containing the positions (e.g. '/path/to/positions.dat')
    """
    with open(positions_path, 'w') as f:
        for position in positions:
            f.write("%s\n" % position)
