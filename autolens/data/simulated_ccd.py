import logging

from autolens import exc
from autolens.data import ccd
from autolens.data.array import grids
from autolens.data.array.scaled_array import ScaledSquarePixelArray, Array
from autolens.model.galaxy.util import galaxy_util

import numpy as np

logger = logging.getLogger(__name__)


class SimulatedCCDData(ccd.CCDData):
    
    def __init__(self, image, pixel_scale, psf, noise_map=None, background_noise_map=None, poisson_noise_map=None,
                 exposure_time_map=None, background_sky_map=None, noise_realization=None, name=None, **kwargs):
        
        super(SimulatedCCDData, self).__init__(image=image, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map, 
                                               background_noise_map=background_noise_map, 
                                               poisson_noise_map=poisson_noise_map, exposure_time_map=exposure_time_map, 
                                               background_sky_map=background_sky_map, name=name, kwargs=kwargs)
        
        self.noise_realization = noise_realization

    @classmethod
    def from_deflections_source_galaxies_and_exposure_arrays(
            cls, deflections, pixel_scale, source_galaxies, exposure_time, psf=None, exposure_time_map=None,
            background_sky_level=0.0, background_sky_map=None, add_noise=True, noise_if_add_noise_false=0.1,
            noise_seed=-1, name=None):

        shape = (deflections.shape[0], deflections.shape[1])

        grid_1d = grids.RegularGrid.from_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        deflections_1d = grids.RegularGrid.from_unmasked_grid_2d(grid_2d=deflections)

        deflected_grid_1d = grid_1d - deflections_1d

        image_1d = galaxy_util.intensities_of_galaxies_from_grid(grid=deflected_grid_1d, galaxies=source_galaxies)

        image_2d = grid_1d.scaled_array_2d_from_array_1d(array_1d=image_1d)

        return cls.from_image_and_exposure_arrays(
            image=image_2d, pixel_scale=pixel_scale, exposure_time=exposure_time,
            psf=psf, exposure_time_map=exposure_time_map, background_sky_level=background_sky_level,
            background_sky_map=background_sky_map, add_noise=add_noise, noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed, name=name)

    @classmethod
    def from_tracer_and_exposure_arrays(cls, tracer, pixel_scale, exposure_time, psf=None, exposure_time_map=None,
                                       background_sky_level=0.0, background_sky_map=None, add_noise=True,
                                       noise_if_add_noise_false=0.1, noise_seed=-1, name=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
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
        noise_seed: int
            A seed for random noise_maps generation
        """
        return cls.from_image_and_exposure_arrays(
            image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=pixel_scale, exposure_time=exposure_time,
            psf=psf, exposure_time_map=exposure_time_map, background_sky_level=background_sky_level,
            background_sky_map=background_sky_map, add_noise=add_noise, noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed, name=name)

    @classmethod
    def from_image_and_exposure_arrays(cls, image, pixel_scale, exposure_time, psf=None, exposure_time_map=None,
                                       background_sky_level=0.0, background_sky_map=None, add_noise=True,
                                       noise_if_add_noise_false=0.1, noise_seed=-1, name=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
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
        noise_seed: int
            A seed for random noise_maps generation
        """

        if psf is None:
            psf = ccd.PSF.from_no_blurring_kernel(pixel_scale=pixel_scale)
            image_needs_trimming = False
        else:
            image_needs_trimming = True

        if exposure_time_map is None:

            exposure_time_map = ScaledSquarePixelArray.single_value(value=exposure_time, shape=image.shape,
                                                                    pixel_scale=pixel_scale)

        if background_sky_map is None:

            background_sky_map = ScaledSquarePixelArray.single_value(value=background_sky_level, shape=image.shape,
                                                                     pixel_scale=pixel_scale)

        image += background_sky_map

        image = psf.convolve(image)

        if image_needs_trimming:
            image = trim_psf_edges(image, psf)
            exposure_time_map = trim_psf_edges(exposure_time_map, psf)
            background_sky_map = trim_psf_edges(background_sky_map, psf)

        if add_noise is True:
            noise_realization = generate_poisson_noise(image, exposure_time_map, noise_seed)
            image += noise_realization
            image_counts = np.multiply(image, exposure_time_map)
            noise_map = np.divide(np.sqrt(image_counts), exposure_time_map)
            noise_map = ccd.NoiseMap(array=noise_map, pixel_scale=pixel_scale)
        else:
            noise_map = ccd.NoiseMap.single_value(value=noise_if_add_noise_false, shape=image.shape,
                                                  pixel_scale=pixel_scale)
            noise_realization = None

        if np.isnan(noise_map).any():
            raise exc.DataException('The noise-map has NaN values in it. This suggests your exposure time and / or'
                                       'background sky levels are too low, creating signal counts at or close to 0.0.')

        image -= background_sky_map

        # ESTIMATE THE BACKGROUND NOISE MAP FROM THE BACKGROUND SKY MAP

        background_noise_map_counts = np.sqrt(np.multiply(background_sky_map, exposure_time_map))
        background_noise_map = np.divide(background_noise_map_counts, exposure_time_map)

        # ESTIMATE THE POISSON NOISE MAP FROM THE IMAGE

        image_counts = np.multiply(image, exposure_time_map)
        poisson_noise_map = np.divide(np.sqrt(np.abs(image_counts)), exposure_time_map)

        image = ScaledSquarePixelArray(array=image, pixel_scale=pixel_scale)
        background_noise_map = ccd.NoiseMap(array=background_noise_map, pixel_scale=pixel_scale)
        poisson_noise_map = ccd.PoissonNoiseMap(array=poisson_noise_map, pixel_scale=pixel_scale)

        return SimulatedCCDData(image, pixel_scale=pixel_scale, psf=psf, noise_map=noise_map,
                                background_noise_map=background_noise_map, poisson_noise_map=poisson_noise_map,
                                exposure_time_map=exposure_time_map, background_sky_map=background_sky_map,
                                noise_realization=noise_realization, name=name)

    def __array_finalize__(self, obj):
        if isinstance(obj, SimulatedCCDData):
            try:
                self.image = obj.image
                self.pixel_scale = obj.pixel_scale
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
                logger.debug("Original object in CCD.__array_finalize__ missing one or more attributes")


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


def trim_psf_edges(array, psf):

    if psf is not None:
        psf_cut_x = np.int(np.ceil(psf.shape[0] / 2)) - 1
        psf_cut_y = np.int(np.ceil(psf.shape[1] / 2)) - 1
        array_x = np.int(array.shape[0])
        array_y = np.int(array.shape[1])
        return array[psf_cut_x:array_x - psf_cut_x, psf_cut_y:array_y - psf_cut_y]
    else:
        return array