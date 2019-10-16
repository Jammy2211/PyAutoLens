import ast
import numpy as np

import autoarray as aa
from autolens import exc


class AbstractData(object):
    def __init__(self, data, noise_map, exposure_time_map=None):
        """A collection of abstract 2D for different data_type classes (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        data : aa.Array
            The array of the image data_type, in units of electrons per second.
        pixel_scales : float
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
        exposure_time_map : aa.Array
            An array describing the effective exposure time in each imaging pixel.
        background_sky_map : aa.Scaled
            An array describing the background sky.
        """
        self.data = data
        self.noise_map = noise_map
        self.exposure_time_map = exposure_time_map

    @property
    def mapping(self):
        return self.data.mask.mapping

    @property
    def geometry(self):
        return self.data.mask.geometry

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)

    @property
    def absolute_signal_to_noise_map(self):
        """The estimated absolute_signal-to-noise_maps mappers of the image."""
        return self.mapping.array_from_array_1d(array_1d=np.divide(np.abs(self.data), self.noise_map))

    @property
    def absolute_signal_to_noise_max(self):
        """The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self):
        """The potential chi-squared map of the imaging data_type. This represents how much each pixel can contribute to \
        the chi-squared map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return self.mapping.array_from_array_1d(array_1d=np.square(self.absolute_signal_to_noise_map))

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
        return self.array_from_electrons_per_second_to_counts(self.data)


class AbstractNoiseMap(aa.Array):
    @classmethod
    def from_weight_map(cls, weight_map):
        """Setup the noise-map from a weight map, which is a form of noise-map that comes via HST image-reduction and \
        the software package MultiDrizzle.

        The variance in each pixel is computed as:

        Variance = 1.0 / sqrt(weight_map).

        The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
        the analysis.

        Parameters
        -----------
        pixel_scales : float
            The size of each pixel in arc seconds.
        weight_map : ndarray
            The weight-value of each pixel which is converted to a variance.
        """
        np.seterr(divide="ignore")
        noise_map = 1.0 / np.sqrt(weight_map)
        noise_map[noise_map > 1.0e8] = 1.0e8
        return noise_map

    @classmethod
    def from_inverse_noise_map(cls, inverse_noise_map):
        """Setup the noise-map from an root-mean square standard deviation map, which is a form of noise-map that \
        comes via HST image-reduction and the software package MultiDrizzle.

        The variance in each pixel is computed as:

        Variance = 1.0 / inverse_std_map.

        The weight map may contain zeros, in which cause the variances are converted to large values to omit them from \
        the analysis.

        Parameters
        -----------
        pixel_scales : float
            The size of each pixel in arc seconds.
        inverse_noise_map : ndarray
            The inverse noise_map value of each pixel which is converted to a variance.
        """
        return 1.0 / inverse_noise_map


class ExposureTimeMap(aa.Array):
    @classmethod
    def from_exposure_time_and_inverse_noise_map(
        cls, exposure_time, inverse_noise_map
    ):
        relative_background_noise_map = inverse_noise_map / np.max(
            inverse_noise_map
        )
        return np.abs(exposure_time * (relative_background_noise_map))


def load_image(image_path, image_hdu, pixel_scales):
    """Factory for loading the image from a .fits file

    Parameters
    ----------
    image_path : str
        The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
    image_hdu : int
        The hdu the image is contained in the .fits file specified by *image_path*.
    pixel_scales : float
        The size of each pixel in arc seconds..
    """
    return aa.array.from_fits(
        file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
    )


def load_exposure_time_map(
    exposure_time_map_path,
    exposure_time_map_hdu,
    pixel_scales,
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
    pixel_scales : float
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
            return ExposureTimeMap.full(
                fill_value=exposure_time, pixel_scales=pixel_scales, shape_2d=shape
            )
        elif exposure_time is None and exposure_time_map_path is not None:
            return ExposureTimeMap.from_fits(
                file_path=exposure_time_map_path,
                hdu=exposure_time_map_hdu,
                pixel_scales=pixel_scales,
            )

    else:

        if exposure_time_map_from_inverse_noise_map:
            return ExposureTimeMap.from_exposure_time_and_inverse_noise_map(
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
