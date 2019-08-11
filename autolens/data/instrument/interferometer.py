import numpy as np
import scipy.signal
from skimage.transform import resize, rescale

from autolens import exc
from autolens.data.array.util import array_util, grid_util, mapping_util
from autolens.data.instrument import abstract_data
from autolens.data.array import scaled_array


class InterferometerData(abstract_data.AbstractData):
    def __init__(
        self,
        image,
        pixel_scale,
        psf,
        noise_map,
        real_visibilities,
        imaginary_visibilities,
        visibilities_noise_map,
        u_wavelengths,
        v_wavelengths,
        primary_beam,
        exposure_time_map=None,
    ):

        super(InterferometerData, self).__init__(
            image=image,
            pixel_scale=pixel_scale,
            psf=psf,
            noise_map=noise_map,
            exposure_time_map=exposure_time_map,
            origin=(0.0, 0.0),
        )

        self.real_visibilities = real_visibilities
        self.imaginary_visibilities = imaginary_visibilities
        self.visibilities = np.sqrt(
            np.square(real_visibilities) + np.square(imaginary_visibilities)
        )
        self.visibilities_noise_map = visibilities_noise_map
        self.u_wavelengths = u_wavelengths
        self.v_wavelengths = v_wavelengths
        self.primary_beam = primary_beam

    def new_interferometer_data_with_resized_arrays(
        self, new_shape, new_centre_pixels=None, new_centre_arcsec=None
    ):

        image = self.resize_scaled_array(
            scaled_array=self.image,
            new_shape=new_shape,
            new_centre_pixels=new_centre_pixels,
            new_centre_arcsec=new_centre_arcsec,
        )

        noise_map = self.resize_scaled_array(
            scaled_array=self.noise_map,
            new_shape=new_shape,
            new_centre_pixels=new_centre_pixels,
            new_centre_arcsec=new_centre_arcsec,
        )

        exposure_time_map = self.resize_scaled_array(
            scaled_array=self.exposure_time_map,
            new_shape=new_shape,
            new_centre_pixels=new_centre_pixels,
            new_centre_arcsec=new_centre_arcsec,
        )

        return InterferometerData(
            image=image,
            pixel_scale=self.pixel_scale,
            psf=self.psf,
            noise_map=noise_map,
            exposure_time_map=exposure_time_map,
            real_visibilities=self.real_visibilities,
            imaginary_visibilities=self.imaginary_visibilities,
            visibilities_noise_map=self.visibilities_noise_map,
            u_wavelengths=self.u_wavelengths,
            v_wavelengths=self.v_wavelengths,
            primary_beam=self.primary_beam,
        )

    def new_interferometer_data_with_resized_psf(self, new_shape):
        psf = self.resize_scaled_array(scaled_array=self.psf, new_shape=new_shape)
        return InterferometerData(
            image=self.image,
            pixel_scale=self.pixel_scale,
            psf=psf,
            noise_map=self.noise_map,
            exposure_time_map=self.exposure_time_map,
            real_visibilities=self.real_visibilities,
            imaginary_visibilities=self.imaginary_visibilities,
            visibilities_noise_map=self.visibilities_noise_map,
            u_wavelengths=self.u_wavelengths,
            v_wavelengths=self.v_wavelengths,
            primary_beam=self.primary_beam,
        )

    def new_interferometer_data_with_resized_primary_beam(self, new_shape):
        primary_beam = self.resize_scaled_array(
            scaled_array=self.primary_beam, new_shape=new_shape
        )
        return InterferometerData(
            image=self.image,
            pixel_scale=self.pixel_scale,
            psf=self.psf,
            noise_map=self.noise_map,
            exposure_time_map=self.exposure_time_map,
            real_visibilities=self.real_visibilities,
            imaginary_visibilities=self.imaginary_visibilities,
            visibilities_noise_map=self.visibilities_noise_map,
            u_wavelengths=self.u_wavelengths,
            v_wavelengths=self.v_wavelengths,
            primary_beam=primary_beam,
        )

    def new_interferometer_data_converted_from_electrons(self):

        image = self.array_from_counts_to_electrons_per_second(array=self.image)
        noise_map = self.array_from_counts_to_electrons_per_second(array=self.noise_map)

        return InterferometerData(
            image=image,
            pixel_scale=self.pixel_scale,
            psf=self.psf,
            noise_map=noise_map,
            exposure_time_map=self.exposure_time_map,
            real_visibilities=self.real_visibilities,
            imaginary_visibilities=self.imaginary_visibilities,
            visibilities_noise_map=self.visibilities_noise_map,
            u_wavelengths=self.u_wavelengths,
            v_wavelengths=self.v_wavelengths,
            primary_beam=self.primary_beam,
        )

    def new_interferometer_data_converted_from_adus(self, gain):

        image = self.array_from_adus_to_electrons_per_second(
            array=self.image, gain=gain
        )
        noise_map = self.array_from_adus_to_electrons_per_second(
            array=self.noise_map, gain=gain
        )

        return InterferometerData(
            image=image,
            pixel_scale=self.pixel_scale,
            psf=self.psf,
            noise_map=noise_map,
            exposure_time_map=self.exposure_time_map,
            real_visibilities=self.real_visibilities,
            imaginary_visibilities=self.imaginary_visibilities,
            visibilities_noise_map=self.visibilities_noise_map,
            u_wavelengths=self.u_wavelengths,
            v_wavelengths=self.v_wavelengths,
            primary_beam=self.primary_beam,
        )


class NoiseMap(abstract_data.AbstractNoiseMap):

    pass


class PrimaryBeam(scaled_array.ScaledSquarePixelArray):

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale, renormalize=False, **kwargs):
        """
        Class storing a 2D Point Spread Function (PrimaryBeam), including its blurring kernel.

        Parameters
        ----------
        array : ndarray
            The 2d PrimaryBeam blurring kernel.
        renormalize : bool
            Renormalize the PrimaryBeam such that he sum of kernel values total 1.0?
        """

        # noinspection PyArgumentList
        super().__init__(array=array, pixel_scale=pixel_scale)
        if renormalize:
            self[:, :] = np.divide(self, np.sum(self))

    @classmethod
    def from_gaussian(
        cls, shape, pixel_scale, sigma, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0
    ):
        """Simulate the primary beam as an elliptical Gaussian profile."""
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

        return PrimaryBeam(array=gaussian_2d, pixel_scale=pixel_scale, renormalize=True)

    @classmethod
    def from_fits_renormalized(cls, file_path, hdu, pixel_scale):
        """Loads a primary beam from fits and renormalizes it

        Parameters
        ----------
        pixel_scale
        file_path: String
            The path to the file containing the PrimaryBeam
        hdu : int
            The HDU the PrimaryBeam is stored in the .fits file.

        Returns
        -------
        primary_beam: PrimaryBeam
            A renormalized PrimaryBeam instance
        """
        primary_beam = PrimaryBeam.from_fits_with_scale(file_path, hdu, pixel_scale)
        primary_beam[:, :] = np.divide(primary_beam, np.sum(primary_beam))
        return primary_beam

    @classmethod
    def from_fits_with_scale(cls, file_path, hdu, pixel_scale):
        """
        Loads the PrimaryBeam from a .fits file.

        Parameters
        ----------
        pixel_scale
        file_path: String
            The path to the file containing the PrimaryBeam
        hdu : int
            The HDU the PrimaryBeam is stored in the .fits file.
        """
        return cls(
            array=array_util.numpy_array_2d_from_fits(file_path, hdu),
            pixel_scale=pixel_scale,
        )

    def new_primary_beam_with_rescaled_odd_dimensioned_array(
        self, rescale_factor, renormalize=True
    ):

        primary_beam_rescaled = rescale(
            self,
            rescale_factor,
            anti_aliasing=False,
            mode="constant",
            multichannel=False,
        )

        if (
            primary_beam_rescaled.shape[0] % 2 == 0
            and primary_beam_rescaled.shape[1] % 2 == 0
        ):
            primary_beam_rescaled = resize(
                primary_beam_rescaled,
                output_shape=(
                    primary_beam_rescaled.shape[0] + 1,
                    primary_beam_rescaled.shape[1] + 1,
                ),
                anti_aliasing=False,
                mode="constant",
            )
        elif (
            primary_beam_rescaled.shape[0] % 2 == 0
            and primary_beam_rescaled.shape[1] % 2 != 0
        ):
            primary_beam_rescaled = resize(
                primary_beam_rescaled,
                output_shape=(
                    primary_beam_rescaled.shape[0] + 1,
                    primary_beam_rescaled.shape[1],
                ),
                anti_aliasing=False,
                mode="constant",
            )
        elif (
            primary_beam_rescaled.shape[0] % 2 != 0
            and primary_beam_rescaled.shape[1] % 2 == 0
        ):
            primary_beam_rescaled = resize(
                primary_beam_rescaled,
                output_shape=(
                    primary_beam_rescaled.shape[0],
                    primary_beam_rescaled.shape[1] + 1,
                ),
                anti_aliasing=False,
                mode="constant",
            )

        pixel_scale_factors = (
            self.shape[0] / primary_beam_rescaled.shape[0],
            self.shape[1] / primary_beam_rescaled.shape[1],
        )
        pixel_scale = (
            self.pixel_scale * pixel_scale_factors[0],
            self.pixel_scale * pixel_scale_factors[1],
        )
        return PrimaryBeam(
            array=primary_beam_rescaled,
            pixel_scale=np.max(pixel_scale),
            renormalize=renormalize,
        )

    def new_primary_beam_with_renormalized_array(self):
        """Renormalize the PrimaryBeam such that its data_vector values sum to unity."""
        return PrimaryBeam(array=self, pixel_scale=self.pixel_scale, renormalize=True)

    def convolve(self, array_2d):
        """
        Convolve an array with this PrimaryBeam

        Parameters
        ----------
        image : ndarray
            An array representing the image the PrimaryBeam is convolved with.

        Returns
        -------
        convolved_image : ndarray
            An array representing the image after convolution.

        Raises
        ------
        KernelException if either PrimaryBeam primary_beam dimension is odd
        """
        if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0:
            raise exc.KernelException("PrimaryBeam Kernel must be odd")

        return scipy.signal.convolve2d(array_2d, self, mode="same")


def load_interferometer_data_from_fits(
    image_path,
    pixel_scale,
    real_visibilities_path=None,
    imaginary_visibilities_path=None,
    visibilities_noise_map_path=None,
    u_wavelengths_path=None,
    v_wavelengths_path=None,
    image_hdu=0,
    resized_interferometer_shape=None,
    resized_interferometer_origin_pixels=None,
    resized_interferometer_origin_arcsec=None,
    psf_path=None,
    psf_hdu=0,
    resized_psf_shape=None,
    renormalize_psf=True,
    resized_primary_beam_shape=None,
    renormalize_primary_beam=True,
    noise_map_path=None,
    noise_map_hdu=0,
    convert_noise_map_from_weight_map=False,
    convert_noise_map_from_inverse_noise_map=False,
    exposure_time_map_path=None,
    exposure_time_map_hdu=0,
    exposure_time_map_from_single_value=None,
    real_visibilities_hdu=0,
    imaginary_visibilities_hdu=0,
    visibilities_noise_map_hdu=0,
    u_wavelengths_hdu=0,
    v_wavelengths_hdu=0,
    primary_beam_path=None,
    primary_beam_hdu=0,
    convert_from_electrons=False,
    gain=None,
    convert_from_adus=False,
):
    """Factory for loading the interferometer instrument from .fits files, as well as computing properties like the noise-map,
    exposure-time map, etc. from the interferometer-instrument.

    This factory also includes a number of routines for converting the interferometer-instrument from units not supported by PyAutoLens \
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
    resized_interferometer_shape : (int, int) | None
        If input, the interferometer arrays that are image sized, e.g. the image, noise-maps) are resized to these dimensions.
    resized_interferometer_origin_pixels : (int, int) | None
        If the interferometer arrays are resized, this defines a new origin (in pixels) around which recentering occurs.
    resized_interferometer_origin_arcsec : (float, float) | None
        If the interferometer arrays are resized, this defines a new origin (in arc-seconds) around which recentering occurs.
    primary_beam_path : str
        The path to the primary_beam .fits file containing the primary_beam (e.g. '/path/to/primary_beam.fits')        
    primary_beam_hdu : int
        The hdu the primary_beam is contained in the .fits file specified by *primary_beam_path*.
    resized_primary_beam_shape : (int, int) | None
        If input, the primary_beam is resized to these dimensions.
    renormalize_psf : bool
        If True, the PrimaryBeam is renoralized such that all elements sum to 1.0.
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
        The exposure time of the interferometer imaging, which is used to compute the exposure-time map as a single value \
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

    image = abstract_data.load_image(
        image_path=image_path, image_hdu=image_hdu, pixel_scale=pixel_scale
    )

    psf = abstract_data.load_psf(
        psf_path=psf_path,
        psf_hdu=psf_hdu,
        pixel_scale=pixel_scale,
        renormalize=renormalize_psf,
    )

    noise_map = load_noise_map(
        noise_map_path=noise_map_path,
        noise_map_hdu=noise_map_hdu,
        pixel_scale=pixel_scale,
        convert_noise_map_from_weight_map=convert_noise_map_from_weight_map,
        convert_noise_map_from_inverse_noise_map=convert_noise_map_from_inverse_noise_map,
    )

    exposure_time_map = abstract_data.load_exposure_time_map(
        exposure_time_map_path=exposure_time_map_path,
        exposure_time_map_hdu=exposure_time_map_hdu,
        pixel_scale=pixel_scale,
        shape=image.shape,
        exposure_time=exposure_time_map_from_single_value,
        exposure_time_map_from_inverse_noise_map=False,
        inverse_noise_map=None,
    )

    real_visibilities = load_visibilities(
        visibilities_path=real_visibilities_path, visibilities_hdu=real_visibilities_hdu
    )
    imaginary_visibilities = load_visibilities(
        visibilities_path=imaginary_visibilities_path,
        visibilities_hdu=imaginary_visibilities_hdu,
    )
    visibilities_noise_map = load_visibilities_noise_map(
        visibilities_noise_map_path=visibilities_noise_map_path,
        visibilities_noise_map_hdu=visibilities_noise_map_hdu,
    )
    u_wavelengths = load_visibilities(
        visibilities_path=u_wavelengths_path, visibilities_hdu=u_wavelengths_hdu
    )
    v_wavelengths = load_visibilities(
        visibilities_path=v_wavelengths_path, visibilities_hdu=v_wavelengths_hdu
    )

    primary_beam = load_primary_beam(
        primary_beam_path=primary_beam_path,
        primary_beam_hdu=primary_beam_hdu,
        pixel_scale=pixel_scale,
        renormalize=renormalize_primary_beam,
    )

    interferometer_data = InterferometerData(
        image=image,
        pixel_scale=pixel_scale,
        psf=psf,
        primary_beam=primary_beam,
        noise_map=noise_map,
        real_visibilities=real_visibilities,
        imaginary_visibilities=imaginary_visibilities,
        visibilities_noise_map=visibilities_noise_map,
        u_wavelengths=u_wavelengths,
        v_wavelengths=v_wavelengths,
        exposure_time_map=exposure_time_map,
    )

    if resized_interferometer_shape is not None:
        interferometer_data = interferometer_data.new_interferometer_data_with_resized_arrays(
            new_shape=resized_interferometer_shape,
            new_centre_pixels=resized_interferometer_origin_pixels,
            new_centre_arcsec=resized_interferometer_origin_arcsec,
        )

    if resized_psf_shape is not None:
        interferometer_data = interferometer_data.new_interferometer_data_with_resized_psf(
            new_shape=resized_psf_shape
        )

    if resized_primary_beam_shape is not None:
        interferometer_data = interferometer_data.new_interferometer_data_with_resized_primary_beam(
            new_shape=resized_primary_beam_shape
        )

    if convert_from_electrons:
        interferometer_data = (
            interferometer_data.new_interferometer_data_converted_from_electrons()
        )
    elif convert_from_adus:
        interferometer_data = interferometer_data.new_interferometer_data_converted_from_adus(
            gain=gain
        )

    return interferometer_data


def load_noise_map(
    noise_map_path,
    noise_map_hdu,
    pixel_scale,
    convert_noise_map_from_weight_map,
    convert_noise_map_from_inverse_noise_map,
):
    """Factory for loading the noise-map from a .fits file.

    This factory also includes a number of routines for converting the noise-map from from other units (e.g. \
    a weight map) or computing the noise-map from other unblurred_image_1d (e.g. the interferometer image and background noise-map).

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
    noise_map_options = sum(
        [convert_noise_map_from_weight_map, convert_noise_map_from_inverse_noise_map]
    )

    if noise_map_options > 1:
        raise exc.DataException(
            "You have specified more than one method to load the noise_map map, e.g.:"
            "convert_noise_map_from_weight_map | "
            "convert_noise_map_from_inverse_noise_map |"
            "noise_map_from_image_and_background_noise_map"
        )

    if noise_map_options == 0 and noise_map_path is not None:
        return NoiseMap.from_fits_with_pixel_scale(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scale=pixel_scale
        )
    elif convert_noise_map_from_weight_map and noise_map_path is not None:
        weight_map = scaled_array.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )
        return NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=pixel_scale)
    elif convert_noise_map_from_inverse_noise_map and noise_map_path is not None:
        inverse_noise_map = scaled_array.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )
        return NoiseMap.from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map, pixel_scale=pixel_scale
        )
    else:
        raise exc.DataException(
            "A noise_map map was not loaded, specify a noise_map_path or option to compute a noise_map map."
        )


def load_visibilities(visibilities_path, visibilities_hdu):

    if visibilities_path is not None:
        return array_util.numpy_array_1d_from_fits(
            file_path=visibilities_path, hdu=visibilities_hdu
        )


def load_visibilities_noise_map(
    visibilities_noise_map_path, visibilities_noise_map_hdu
):
    if visibilities_noise_map_path is not None:
        return array_util.numpy_array_1d_from_fits(
            file_path=visibilities_noise_map_path, hdu=visibilities_noise_map_hdu
        )


def load_baselines(baselines_path, baselines_hdu):
    if baselines_path is not None:
        return array_util.numpy_array_1d_from_fits(
            file_path=baselines_path, hdu=baselines_hdu
        )


def load_primary_beam(
    primary_beam_path, primary_beam_hdu, pixel_scale, renormalize=False
):
    """Factory for loading the primary_beam from a .fits file.

    Parameters
    ----------
    primary_beam_path : str
        The path to the primary_beam .fits file containing the primary_beam (e.g. '/path/to/primary_beam.fits')
    primary_beam_hdu : int
        The hdu the primary_beam is contained in the .fits file specified by *primary_beam_path*.
    pixel_scale : float
        The size of each pixel in arc seconds.
    renormalize : bool
        If True, the PrimaryBeam is renoralized such that all elements sum to 1.0.
    """
    if renormalize:
        return PrimaryBeam.from_fits_renormalized(
            file_path=primary_beam_path, hdu=primary_beam_hdu, pixel_scale=pixel_scale
        )
    if not renormalize:
        return PrimaryBeam.from_fits_with_scale(
            file_path=primary_beam_path, hdu=primary_beam_hdu, pixel_scale=pixel_scale
        )


def output_interferometer_data_to_fits(
    interferometer_data,
    image_path,
    psf_path,
    noise_map_path=None,
    primary_beam_path=None,
    exposure_time_map_path=None,
    real_visibilities_path=None,
    imaginary_visibilities_path=None,
    visibilities_noise_map_path=None,
    u_wavelengths_path=None,
    v_wavelengths_path=None,
    overwrite=False,
):
    array_util.numpy_array_2d_to_fits(
        array_2d=interferometer_data.image, file_path=image_path, overwrite=overwrite
    )
    array_util.numpy_array_2d_to_fits(
        array_2d=interferometer_data.psf, file_path=psf_path, overwrite=overwrite
    )

    if primary_beam_path is not None:
        array_util.numpy_array_2d_to_fits(
            array_2d=interferometer_data.primary_beam,
            file_path=primary_beam_path,
            overwrite=overwrite,
        )

    if interferometer_data.noise_map is not None and noise_map_path is not None:
        array_util.numpy_array_2d_to_fits(
            array_2d=interferometer_data.noise_map,
            file_path=noise_map_path,
            overwrite=overwrite,
        )

    if (
        interferometer_data.exposure_time_map is not None
        and exposure_time_map_path is not None
    ):
        array_util.numpy_array_2d_to_fits(
            array_2d=interferometer_data.exposure_time_map,
            file_path=exposure_time_map_path,
            overwrite=overwrite,
        )

    if (
        interferometer_data.real_visibilities is not None
        and real_visibilities_path is not None
    ):
        array_util.numpy_array_1d_to_fits(
            array_1d=interferometer_data.real_visibilities,
            file_path=real_visibilities_path,
            overwrite=overwrite,
        )

    if (
        interferometer_data.imaginary_visibilities is not None
        and imaginary_visibilities_path is not None
    ):
        array_util.numpy_array_1d_to_fits(
            array_1d=interferometer_data.imaginary_visibilities,
            file_path=imaginary_visibilities_path,
            overwrite=overwrite,
        )

    if (
        interferometer_data.visibilities_noise_map is not None
        and visibilities_noise_map_path is not None
    ):
        array_util.numpy_array_1d_to_fits(
            array_1d=interferometer_data.visibilities_noise_map,
            file_path=visibilities_noise_map_path,
            overwrite=overwrite,
        )

    if interferometer_data.u_wavelengths is not None and u_wavelengths_path is not None:
        array_util.numpy_array_1d_to_fits(
            array_1d=interferometer_data.u_wavelengths,
            file_path=u_wavelengths_path,
            overwrite=overwrite,
        )

    if interferometer_data.v_wavelengths is not None and v_wavelengths_path is not None:
        array_util.numpy_array_1d_to_fits(
            array_1d=interferometer_data.v_wavelengths,
            file_path=v_wavelengths_path,
            overwrite=overwrite,
        )
