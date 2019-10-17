import logging
import numpy as np

import autoarray as aa

from autolens import exc
from autoarray.data import abstract_data

logger = logging.getLogger(__name__)


class UVPlaneData(abstract_data.AbstractData):
    def __init__(
        self,
        shape_2d,
        pixel_scales,
        visibilities,
        noise_map,
        uv_wavelengths,
        primary_beam,
        exposure_time_map=None,
    ):

        self.shape = shape_2d

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        self.pixel_scales = pixel_scales

        super(UVPlaneData, self).__init__(
            data=visibilities,
            noise_map=noise_map,
            exposure_time_map=exposure_time_map,
        )

        self.visibilities_magnitudes = np.sqrt(
            np.square(self.visibilities[:, 0]) + np.square(self.visibilities[:, 1])
        )
        self.uv_wavelengths = uv_wavelengths
        self.primary_beam = primary_beam

    @property
    def visibilities(self):
        return self.data

    def modified_visibilities_from_visibilities(self, visibilities):

        return UVPlaneData(
            shape_2d=self.shape,
            pixel_scales=self.pixel_scales,
            visibilities=visibilities,
            noise_map=self.noise_map,
            uv_wavelengths=self.uv_wavelengths,
            primary_beam=self.primary_beam,
            exposure_time_map=self.exposure_time_map,
        )

    def resized_primary_beam_from_new_shape(self, new_shape):

        primary_beam = self.primary_beam.resized_from_new_shape(
            new_shape=new_shape
        )
        return UVPlaneData(
            shape_2d=self.shape,
            visibilities=self.data,
            pixel_scales=self.pixel_scales,
            noise_map=self.noise_map,
            exposure_time_map=self.exposure_time_map,
            uv_wavelengths=self.uv_wavelengths,
            primary_beam=primary_beam,
        )

    def data_in_electrons(self):

        real_visibilities = self.array_from_counts_to_electrons_per_second(
            array=self.data[:, 0]
        )
        imaginary_visibilities = self.array_from_counts_to_electrons_per_second(
            array=self.data[:, 1]
        )
        visibilities = np.stack((real_visibilities, imaginary_visibilities), axis=-1)
        noise_map = self.array_from_counts_to_electrons_per_second(array=self.noise_map)

        return UVPlaneData(
            shape_2d=self.shape,
            visibilities=visibilities,
            pixel_scales=self.pixel_scales,
            noise_map=noise_map,
            exposure_time_map=self.exposure_time_map,
            uv_wavelengths=self.uv_wavelengths,
            primary_beam=self.primary_beam,
        )

    def data_in_adus_from_gain(self, gain):

        real_visibilities = self.array_from_adus_to_electrons_per_second(
            array=self.data[:, 0], gain=gain
        )
        imaginary_visibilities = self.array_from_adus_to_electrons_per_second(
            array=self.data[:, 1], gain=gain
        )
        visibilities = np.stack((real_visibilities, imaginary_visibilities), axis=-1)

        noise_map = self.array_from_adus_to_electrons_per_second(
            array=self.noise_map, gain=gain
        )

        return UVPlaneData(
            shape_2d=self.shape,
            visibilities=visibilities,
            pixel_scales=self.pixel_scales,
            noise_map=noise_map,
            exposure_time_map=self.exposure_time_map,
            uv_wavelengths=self.uv_wavelengths,
            primary_beam=self.primary_beam,
        )


class SimulatedUVPlaneData(UVPlaneData):
    def __init__(
        self,
        shape_2d,
        pixel_scales,
        visibilities,
        noise_map,
        uv_wavelengths,
        primary_beam,
        noise_map_realization,
        exposure_time_map=None,
        **kwargs
    ):

        super(SimulatedUVPlaneData, self).__init__(
            shape_2d=shape_2d,
            visibilities=visibilities,
            pixel_scales=pixel_scales,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            primary_beam=primary_beam,
            exposure_time_map=exposure_time_map,
        )

        self.noise_map_realization = noise_map_realization

    @classmethod
    def from_deflections_galaxies_and_exposure_arrays(
        cls,
        deflections,
        pixel_scales,
        galaxies,
        exposure_time,
        transformer,
        primary_beam=None,
        exposure_time_map=None,
        background_sky_level=0.0,
        background_sky_map=None,
        noise_sigma=None,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):

        grid = aa.grid.uniform(
            shape_2d=deflections.shape_2d, pixel_scales=pixel_scales, sub_size=1
        )

        deflected_grid_1d = grid.in_1d - deflections.in_1d

        image_2d = sum(
            map(lambda g: g.profile_image_from_grid(grid=deflected_grid_1d), galaxies)
        )

        return cls.from_image_and_exposure_arrays(
            image=image_2d,
            pixel_scales=pixel_scales,
            exposure_time=exposure_time,
            exposure_time_map=exposure_time_map,
            background_sky_level=background_sky_level,
            background_sky_map=background_sky_map,
            transformer=transformer,
            primary_beam=primary_beam,
            noise_sigma=noise_sigma,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def from_tracer_grid_and_exposure_arrays(
        cls,
        tracer,
        grid,
        pixel_scales,
        exposure_time,
        transformer,
        primary_beam=None,
        exposure_time_map=None,
        background_sky_level=0.0,
        background_sky_map=None,
        noise_sigma=None,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and UVPlane read-out).
        pixel_scales: float
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
        noise_seed: int
            A seed for random noise_maps generation
        """

        image_2d = tracer.profile_image_from_grid(grid=grid)

        return cls.from_image_and_exposure_arrays(
            image=image_2d,
            pixel_scales=pixel_scales,
            exposure_time=exposure_time,
            exposure_time_map=exposure_time_map,
            background_sky_level=background_sky_level,
            background_sky_map=background_sky_map,
            transformer=transformer,
            primary_beam=primary_beam,
            noise_sigma=noise_sigma,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def from_image_and_exposure_arrays(
        cls,
        image,
        pixel_scales,
        exposure_time,
        transformer,
        primary_beam=None,
        exposure_time_map=None,
        background_sky_level=0.0,
        background_sky_map=None,
        noise_sigma=None,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and UVPlane read-out).
        pixel_scales: float
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
        noise_seed: int
            A seed for random noise_maps generation
        """

        if exposure_time_map is None:

            exposure_time_map = aa.array.full(
                fill_value=exposure_time, shape_2d=image.shape_2d, pixel_scales=pixel_scales
            )

        if background_sky_map is None:

            background_sky_map = aa.array.full(
                fill_value=background_sky_level, shape_2d=image.shape_2d, pixel_scales=pixel_scales
            )

        image += background_sky_map

        visibilities = transformer.visibilities_from_image(image=image)

        if noise_sigma is not None:
            noise_map_realization = gaussian_noise_map_from_shape_and_sigma(
                shape=visibilities.shape, sigma=noise_sigma, noise_seed=noise_seed
            )
            visibilities = visibilities + noise_map_realization
            noise_map = np.full(
                fill_value=noise_sigma, shape=visibilities.shape,
            )
        else:
            noise_map = np.full(
                fill_value=noise_if_add_noise_false,
                shape=visibilities.shape,
            )
            noise_map_realization = None

        if np.isnan(noise_map).any():
            raise exc.DataException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        image -= background_sky_map

        return SimulatedUVPlaneData(
            shape_2d=image.shape,
            visibilities=visibilities,
            pixel_scales=pixel_scales,
            noise_map=noise_map,
            uv_wavelengths=transformer.uv_wavelengths,
            primary_beam=primary_beam,
            noise_map_realization=noise_map_realization,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            noise_realization=noise_map_realization,
        )

    def __array_finalize__(self, obj):
        if isinstance(obj, SimulatedUVPlaneData):
            try:
                self.data = obj.data
                self.pixel_scales = obj.pixel_scales
                self.psf = obj.psf
                self.noise_map = obj.noise_map
                self.background_noise_map = obj.background_noise_map
                self.poisson_noise_map = obj.poisson_noise_map
                self.exposure_time_map = obj.exposure_time_map
                self.background_sky_map = obj.background_sky_map
                self.background_noise_realization = obj.background_noise_realization
                self.poisson_noise_realization = obj.poisson_noise_realization
                self.origin = obj.origin
            except AttributeError:
                logger.debug(
                    "Original object in UVPlane.__array_finalize__ missing one or more attributes"
                )


def gaussian_noise_map_from_shape_and_sigma(shape, sigma, noise_seed=-1):
    """Generate a two-dimensional read noises-map, generating values from a Gaussian distribution with mean 0.0.

    Params
    ----------
    shape : (int, int)
        The (x,y) image_shape of the generated Gaussian noises map.
    read_noise : float
        Standard deviation of the 1D Gaussian that each noises value is drawn from
    seed : int
        The seed of the random number generator, used for the random noises maps.
    """
    if noise_seed == -1:
        # Use one seed, so all regions have identical column non-uniformity.
        noise_seed = np.random.randint(0, int(1e9))
    np.random.seed(noise_seed)
    read_noise_map = np.random.normal(loc=0.0, scale=sigma, size=shape)
    return read_noise_map


def load_uv_plane_data_from_fits(
    shape,
    pixel_scales,
    real_visibilities_path=None,
    real_visibilities_hdu=0,
    imaginary_visibilities_path=None,
    imaginary_visibilities_hdu=0,
    noise_map_path=None,
    noise_map_hdu=0,
    u_wavelengths_path=None,
    u_wavelengths_hdu=0,
    v_wavelengths_path=None,
    v_wavelengths_hdu=0,
    resized_primary_beam_shape=None,
    renormalize_primary_beam=True,
    exposure_time_map_path=None,
    exposure_time_map_hdu=0,
    exposure_time_map_from_single_value=None,
    primary_beam_path=None,
    primary_beam_hdu=0,
    convert_from_electrons=False,
    gain=None,
    convert_from_adus=False,
):
    """Factory for loading the uv_plane data_type from .fits files, as well as computing properties like the noise-map,
    exposure-time map, etc. from the uv_plane-data_type.

    This factory also includes a number of routines for converting the uv_plane-data_type from units not supported by PyAutoLens \
    (e.g. adus, electrons) to electrons per second.

    Parameters
    ----------
    lens_name
    image_path : str
        The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
    pixel_scales : float
        The size of each pixel in arc seconds.
    image_hdu : int
        The hdu the image is contained in the .fits file specified by *image_path*.        
    image_hdu : int
        The hdu the image is contained in the .fits file that *image_path* points too.
    resized_uv_plane_shape : (int, int) | None
        If input, the uv_plane structures that are image sized, e.g. the image, noise-maps) are resized to these dimensions.
    resized_uv_plane_origin_pixels : (int, int) | None
        If the uv_plane structures are resized, this defines a new origin (in pixels) around which recentering occurs.
    resized_uv_plane_origin_arcsec : (float, float) | None
        If the uv_plane structures are resized, this defines a new origin (in arc-seconds) around which recentering occurs.
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
        The exposure time of the uv_plane imaging, which is used to compute the exposure-time map as a single value \
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

    real_visibilities = load_visibilities(
        visibilities_path=real_visibilities_path, visibilities_hdu=real_visibilities_hdu
    )
    imaginary_visibilities = load_visibilities(
        visibilities_path=imaginary_visibilities_path,
        visibilities_hdu=imaginary_visibilities_hdu,
    )

    visibilities = np.stack((real_visibilities, imaginary_visibilities), axis=-1)

    exposure_time_map = load_exposure_time_map(
        exposure_time_map_path=exposure_time_map_path,
        exposure_time_map_hdu=exposure_time_map_hdu,
        shape_1d=real_visibilities.shape,
        exposure_time=exposure_time_map_from_single_value,
    )

    noise_map = load_visibilities_noise_map(
        noise_map_path=noise_map_path, noise_map_hdu=noise_map_hdu
    )
    u_wavelengths = load_visibilities(
        visibilities_path=u_wavelengths_path, visibilities_hdu=u_wavelengths_hdu
    )
    v_wavelengths = load_visibilities(
        visibilities_path=v_wavelengths_path, visibilities_hdu=v_wavelengths_hdu
    )

    uv_wavelengths = np.stack((u_wavelengths, v_wavelengths), axis=-1)

    primary_beam = load_primary_beam(
        primary_beam_path=primary_beam_path,
        primary_beam_hdu=primary_beam_hdu,
        pixel_scales=pixel_scales,
        renormalize=renormalize_primary_beam,
    )

    uv_plane_data = UVPlaneData(
        shape_2d=shape,
        visibilities=visibilities,
        pixel_scales=pixel_scales,
        primary_beam=primary_beam,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        exposure_time_map=exposure_time_map,
    )

    if resized_primary_beam_shape is not None:
        uv_plane_data = uv_plane_data.resized_primary_beam_from_new_shape(
            new_shape=resized_primary_beam_shape
        )

    if convert_from_electrons:
        uv_plane_data = uv_plane_data.data_in_electrons()
    elif convert_from_adus:
        uv_plane_data = uv_plane_data.data_in_adus_from_gain(gain=gain)

    return uv_plane_data


def load_visibilities(visibilities_path, visibilities_hdu):

    if visibilities_path is not None:
        return aa.array_util.numpy_array_1d_from_fits(
            file_path=visibilities_path, hdu=visibilities_hdu
        )


def load_visibilities_noise_map(noise_map_path, noise_map_hdu):
    if noise_map_path is not None:
        return aa.array_util.numpy_array_1d_from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )


def load_primary_beam(
    primary_beam_path, primary_beam_hdu, pixel_scales, renormalize=False
):
    """Factory for loading the primary_beam from a .fits file.

    Parameters
    ----------
    primary_beam_path : str
        The path to the primary_beam .fits file containing the primary_beam (e.g. '/path/to/primary_beam.fits')
    primary_beam_hdu : int
        The hdu the primary_beam is contained in the .fits file specified by *primary_beam_path*.
    pixel_scales : float
        The size of each pixel in arc seconds.
    renormalize : bool
        If True, the PrimaryBeam is renoralized such that all elements sum to 1.0.
    """
    return aa.kernel.from_fits(
        file_path=primary_beam_path, hdu=primary_beam_hdu, pixel_scales=pixel_scales, renormalize=renormalize
    )


def load_exposure_time_map(
    exposure_time_map_path,
    exposure_time_map_hdu,
    shape_1d=None,
    exposure_time=None,
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
    shape_1d : (int, int)
        The shape of the image, required if a single value is used to calculate the exposure time map.
    exposure_time : float
        The exposure-time used to compute the expsure-time map if only a single value is used.
    exposure_time_map_from_inverse_noise_map : bool
        If True, the exposure-time map is computed from the background noise_map map \
        (see *ExposureTimeMap.from_background_noise_map*)
    inverse_noise_map : ndarray
        The background noise-map, which the Poisson noise-map can be calculated using.
    """

    if exposure_time is not None and exposure_time_map_path is not None:
        raise exc.DataException(
            "You have supplied both a exposure_time_map_path to an exposure time map and an exposure time. Only"
            "one quantity should be supplied."
        )

    if exposure_time is not None and exposure_time_map_path is None:
        return np.full(
            fill_value=exposure_time, shape=shape_1d
        )
    elif exposure_time is None and exposure_time_map_path is not None:
        return aa.array_util.numpy_array_1d_from_fits(
            file_path=exposure_time_map_path,
            hdu=exposure_time_map_hdu,
        )


def output_uv_plane_data_to_fits(
    uv_plane_data,
    real_visibilities_path=None,
    imaginary_visibilities_path=None,
    noise_map_path=None,
    primary_beam_path=None,
    exposure_time_map_path=None,
    u_wavelengths_path=None,
    v_wavelengths_path=None,
    overwrite=False,
):

    if primary_beam_path is not None:
        aa.array_util.numpy_array_2d_to_fits(
            array_2d=uv_plane_data.primary_beam.in_2d,
            file_path=primary_beam_path,
            overwrite=overwrite,
        )

    if (
        uv_plane_data.exposure_time_map is not None
        and exposure_time_map_path is not None
    ):
        aa.array_util.numpy_array_1d_to_fits(
            array_1d=uv_plane_data.exposure_time_map,
            file_path=exposure_time_map_path,
            overwrite=overwrite,
        )

    if uv_plane_data.visibilities is not None and real_visibilities_path is not None:
        aa.array_util.numpy_array_1d_to_fits(
            array_1d=uv_plane_data.visibilities[:, 0],
            file_path=real_visibilities_path,
            overwrite=overwrite,
        )

    if (
        uv_plane_data.visibilities is not None
        and imaginary_visibilities_path is not None
    ):
        aa.array_util.numpy_array_1d_to_fits(
            array_1d=uv_plane_data.visibilities[:, 1],
            file_path=imaginary_visibilities_path,
            overwrite=overwrite,
        )

    if uv_plane_data.noise_map is not None and noise_map_path is not None:
        aa.array_util.numpy_array_1d_to_fits(
            array_1d=uv_plane_data.noise_map,
            file_path=noise_map_path,
            overwrite=overwrite,
        )

    if uv_plane_data.uv_wavelengths is not None and u_wavelengths_path is not None:
        aa.array_util.numpy_array_1d_to_fits(
            array_1d=uv_plane_data.uv_wavelengths[:, 0],
            file_path=u_wavelengths_path,
            overwrite=overwrite,
        )

    if uv_plane_data.uv_wavelengths is not None and v_wavelengths_path is not None:
        aa.array_util.numpy_array_1d_to_fits(
            array_1d=uv_plane_data.uv_wavelengths[:, 1],
            file_path=v_wavelengths_path,
            overwrite=overwrite,
        )
