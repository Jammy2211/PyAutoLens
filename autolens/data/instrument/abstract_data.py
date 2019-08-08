import ast
import numpy as np

import scipy.signal
from astropy import units
from skimage.transform import resize, rescale

from autolens import exc
from autolens.data.array.util import array_util, grid_util, mapping_util
from autolens.data.array import scaled_array
from autolens.model.profiles.light_profiles import EllipticalGaussian


class AbstractData(object):
    def __init__(
        self,
        image,
        pixel_scale,
        psf,
        noise_map,
        exposure_time_map=None,
        origin=(0.0, 0.0),
    ):
        """A collection of abstract 2D for different instrument classes (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        image : scaled_array.ScaledArraySquarePixels
            The array of the image instrument, in units of electrons per second.
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
        self.image = image
        self.pixel_scale = pixel_scale
        self.psf = psf
        self.noise_map = noise_map
        self.exposure_time_map = exposure_time_map
        self.origin = origin

    @property
    def shape(self):
        return self.image.shape

    @staticmethod
    def bin_up_scaled_array(scaled_array, bin_up_factor, method):
        if scaled_array is not None:
            return scaled_array.binned_up_array_from_array(
                bin_up_factor=bin_up_factor, method=method
            )
        else:
            return None

    @staticmethod
    def resize_scaled_array(
        scaled_array, new_shape, new_centre_pixels=None, new_centre_arcsec=None
    ):
        if scaled_array is not None:
            return scaled_array.resized_scaled_array_from_array(
                new_shape=new_shape,
                new_centre_pixels=new_centre_pixels,
                new_centre_arcsec=new_centre_arcsec,
            )
        else:
            return None

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
        """The potential chi-squared map of the ccd instrument. This represents how much each pixel can contribute to \
        the chi-squared map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return np.square(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_max(self):
        """The maximum value of the potential chi-squared map"""
        return np.max(self.potential_chi_squared_map)

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


class AbstractNoiseMap(scaled_array.ScaledSquarePixelArray):
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
        np.seterr(divide="ignore")
        noise_map = 1.0 / np.sqrt(weight_map)
        noise_map[noise_map == np.inf] = 1.0e8
        return cls(array=noise_map, pixel_scale=pixel_scale)

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
        return cls(array=noise_map, pixel_scale=pixel_scale)


class PSF(scaled_array.ScaledSquarePixelArray):

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
    def from_no_blurring_kernel(cls, pixel_scale):

        array = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        return PSF(array=array, pixel_scale=pixel_scale, renormalize=False)

    @classmethod
    def from_gaussian(
        cls, shape, pixel_scale, sigma, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0
    ):
        """Simulate the PSF as an elliptical Gaussian profile."""
        from autolens.model.profiles.light_profiles import EllipticalGaussian

        gaussian = EllipticalGaussian(
            centre=centre, axis_ratio=axis_ratio, phi=phi, intensity=1.0, sigma=sigma
        )
        grid_1d = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full(shape, False),
            pixel_scales=(pixel_scale, pixel_scale),
            sub_grid_size=1,
        )
        gaussian_1d = gaussian.intensities_from_grid(grid=grid_1d)

        gaussian_2d = mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_grid_size(
            sub_array_1d=gaussian_1d,
            mask=np.full(fill_value=False, shape=shape),
            sub_grid_size=1,
        )

        return PSF(array=gaussian_2d, pixel_scale=pixel_scale, renormalize=True)

    @classmethod
    def from_as_gaussian_via_alma_fits_header_parameters(
        cls, shape, pixel_scale, y_stddev, x_stddev, theta, centre=(0.0, 0.0)
    ):

        x_stddev = (
            x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        axis_ratio = x_stddev / y_stddev

        gaussian = EllipticalGaussian(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=90.0 - theta,
            intensity=1.0,
            sigma=y_stddev,
        )

        grid_1d = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full(shape, False),
            pixel_scales=(pixel_scale, pixel_scale),
            sub_grid_size=1,
        )

        gaussian_1d = gaussian.intensities_from_grid(grid=grid_1d)
        gaussian_2d = mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_grid_size(
            sub_array_1d=gaussian_1d,
            mask=np.full(fill_value=False, shape=shape),
            sub_grid_size=1,
        )

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
        return cls(
            array=array_util.numpy_array_2d_from_fits(file_path, hdu),
            pixel_scale=pixel_scale,
        )

    def new_psf_with_rescaled_odd_dimensioned_array(
        self, rescale_factor, renormalize=True
    ):

        psf_rescaled = rescale(
            self,
            rescale_factor,
            anti_aliasing=False,
            mode="constant",
            multichannel=False,
        )

        if psf_rescaled.shape[0] % 2 == 0 and psf_rescaled.shape[1] % 2 == 0:
            psf_rescaled = resize(
                psf_rescaled,
                output_shape=(psf_rescaled.shape[0] + 1, psf_rescaled.shape[1] + 1),
                anti_aliasing=False,
                mode="constant",
            )
        elif psf_rescaled.shape[0] % 2 == 0 and psf_rescaled.shape[1] % 2 != 0:
            psf_rescaled = resize(
                psf_rescaled,
                output_shape=(psf_rescaled.shape[0] + 1, psf_rescaled.shape[1]),
                anti_aliasing=False,
                mode="constant",
            )
        elif psf_rescaled.shape[0] % 2 != 0 and psf_rescaled.shape[1] % 2 == 0:
            psf_rescaled = resize(
                psf_rescaled,
                output_shape=(psf_rescaled.shape[0], psf_rescaled.shape[1] + 1),
                anti_aliasing=False,
                mode="constant",
            )

        pixel_scale_factors = (
            self.shape[0] / psf_rescaled.shape[0],
            self.shape[1] / psf_rescaled.shape[1],
        )
        pixel_scale = (
            self.pixel_scale * pixel_scale_factors[0],
            self.pixel_scale * pixel_scale_factors[1],
        )
        return PSF(
            array=psf_rescaled, pixel_scale=np.max(pixel_scale), renormalize=renormalize
        )

    def new_psf_with_renormalized_array(self):
        """Renormalize the PSF such that its data_vector values sum to unity."""
        return PSF(array=self, pixel_scale=self.pixel_scale, renormalize=True)

    def convolve(self, array_2d):
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

        return scipy.signal.convolve2d(array_2d, self, mode="same")


class ExposureTimeMap(scaled_array.ScaledSquarePixelArray):
    @classmethod
    def from_exposure_time_and_inverse_noise_map(
        cls, pixel_scale, exposure_time, inverse_noise_map
    ):
        relative_background_noise_map = inverse_noise_map / np.max(inverse_noise_map)
        return ExposureTimeMap(
            array=np.abs(exposure_time * (relative_background_noise_map)),
            pixel_scale=pixel_scale,
        )


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
    return scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(
        file_path=image_path, hdu=image_hdu, pixel_scale=pixel_scale
    )


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
        return PSF.from_fits_renormalized(
            file_path=psf_path, hdu=psf_hdu, pixel_scale=pixel_scale
        )
    if not renormalize:
        return PSF.from_fits_with_scale(
            file_path=psf_path, hdu=psf_hdu, pixel_scale=pixel_scale
        )


def load_exposure_time_map(
    exposure_time_map_path,
    exposure_time_map_hdu,
    pixel_scale,
    shape=None,
    exposure_time=None,
    exposure_time_map_from_inverse_noise_map=False,
    inverse_noise_map=None,
):
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
        raise exc.DataException(
            "You have supplied both a exposure_time_map_path to an exposure time map and an exposure time. Only"
            "one quantity should be supplied."
        )

    if exposure_time_map_options == 0:

        if exposure_time is not None and exposure_time_map_path is None:
            return ExposureTimeMap.single_value(
                value=exposure_time, pixel_scale=pixel_scale, shape=shape
            )
        elif exposure_time is None and exposure_time_map_path is not None:
            return ExposureTimeMap.from_fits_with_pixel_scale(
                file_path=exposure_time_map_path,
                hdu=exposure_time_map_hdu,
                pixel_scale=pixel_scale,
            )

    else:

        if exposure_time_map_from_inverse_noise_map:
            return ExposureTimeMap.from_exposure_time_and_inverse_noise_map(
                pixel_scale=pixel_scale,
                exposure_time=exposure_time,
                inverse_noise_map=inverse_noise_map,
            )


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
    with open(positions_path, "w") as f:
        for position in positions:
            f.write("%s\n" % position)
