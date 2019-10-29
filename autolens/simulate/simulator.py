import autofit as af
import autoarray as aa

from autoarray.data import imaging
from autolens.lens import ray_tracing
from autoarray.plotters import imaging_plotters


class ImagingSimulator(object):
    def __init__(
        self, shape_2d, pixel_scales, psf, exposure_time, background_sky_level, add_noise=True,        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulate a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of an observation using this data_type.
        background_sky_level : float
            The level of the background sky of an observationg using this data_type.
        """

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        self.shape = shape_2d
        self.pixel_scales = pixel_scales
        self.psf = psf
        self.exposure_time = exposure_time
        self.background_sky_level = background_sky_level
        self.add_noise = add_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    @classmethod
    def lsst(
        cls,
        shape=(101, 101),
        pixel_scales=0.2,
        psf_shape=(31, 31),
        psf_sigma=0.5,
        exposure_time=100.0,
        background_sky_level=1.0,
        add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Large Synotpic Survey Telescope.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def euclid(
        cls,
        shape=(151, 151),
        pixel_scales=0.1,
        psf_shape=(31, 31),
        psf_sigma=0.1,
        exposure_time=565.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Euclid space satellite.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def hst(
        cls,
        shape=(251, 251),
        pixel_scales=0.05,
        psf_shape=(31, 31),
        psf_sigma=0.05,
        exposure_time=2000.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Hubble Space Telescope.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    @classmethod
    def hst_up_sampled(
        cls,
        shape=(401, 401),
        pixel_scales=0.03,
        psf_shape=(31, 31),
        psf_sigma=0.05,
        exposure_time=2000.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation with the Hubble Space Telescope which has been upscaled to a higher \
        pixel-scale to better sample the PSF.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed
        )

    @classmethod
    def keck_adaptive_optics(
        cls,
        shape=(751, 751),
        pixel_scales=0.01,
        psf_shape=(31, 31),
        psf_sigma=0.025,
        exposure_time=1000.0,
        background_sky_level=1.0,
            add_noise=True,
            noise_if_add_noise_false=0.1,
            noise_seed=-1,
    ):
        """Default settings for an observation using Keck Adaptive Optics imaging.

        This can be customized by over-riding the default input values."""
        psf = aa.kernel.from_gaussian(
            shape_2d=psf_shape, sigma=psf_sigma, pixel_scales=pixel_scales
        )
        return ImagingSimulator(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_galaxies(
        self,
        galaxies,
        sub_size=16,
        should_plot_imaging=False,
    ):
        """Simulate Imaging data_type for this data_type, as follows:

        1)  Setup the image-plane al.ogrid of the Imaging array, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated Imaging data_type.

        3) Simulate the Imaging data_type, using a special image which ensures edge-effects don't
           degrade simulate of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the simulate to .fits format if a data_path and data_name are specified. Otherwise, return the simulated \
           imaging data_type instance."""

        grid = aa.grid.uniform(
            shape_2d=self.shape, pixel_scales=self.psf.pixel_scales, sub_size=sub_size
        )

        tracer = ray_tracing.Tracer.from_galaxies(galaxies=galaxies)

        imaging = self.from_tracer_and_grid(
            tracer=tracer,
            grid=grid,
        )

        if should_plot_imaging:
            imaging_plotters.subplot(imaging=imaging)

        return imaging

    def from_galaxies_and_write_to_fits(
        self,
        galaxies,
        data_path,
        data_name,
        sub_size=16,
        should_plot_imaging=False,
    ):
        """Simulate Imaging data_type for this data_type, as follows:

        1)  Setup the image-plane al.ogrid of the Imaging array, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated Imaging data_type.

        3) Simulate the Imaging data_type, using a special image which ensures edge-effects don't
           degrade simulate of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the simulate to .fits format if a data_path and data_name are specified. Otherwise, return the simulated \
           imaging data_type instance."""

        imaging = self.from_galaxies(
            galaxies=galaxies,
            sub_size=sub_size,
            should_plot_imaging=should_plot_imaging,
        )

        data_output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
            path=data_path, folder_names=[data_name]
        )

        imaging.output_to_fits(
            image_path=data_output_path + "image.fits",
            psf_path=data_output_path + "psf.fits",
            noise_map_path=data_output_path + "noise_map.fits",
            exposure_time_map_path=data_output_path + "exposure_time_map.fits",
            background_noise_map_path=data_output_path + "background_noise_map.fits",
            poisson_noise_map_path=data_output_path + "poisson_noise_map.fits",
            background_sky_map_path=data_output_path + "background_sky_map.fits",
            overwrite=True,
        )

    def from_deflections_and_galaxies(
        self,
        deflections,
        galaxies,
        name=None,
    ):

        grid = aa.grid.uniform(
            shape_2d=deflections.shape_2d,
            pixel_scales=deflections.pixel_scales,
            sub_size=1,
        )

        deflected_grid = grid - deflections.in_1d_binned

        image_2d = sum(
            map(lambda g: g.profile_image_from_grid(grid=deflected_grid), galaxies)
        )

        return self.from_image(
            image=image_2d,
            name=name,
        )

    def from_tracer_and_grid(
        self,
        tracer,
        grid,
        name=None,
    ):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
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

        image = tracer.padded_profile_image_from_grid_and_psf_shape(
            grid=grid, psf_shape=self.psf.shape_2d
        )

        return self.from_image(
            image=image.in_1d_binned,
            name=name,
        )

    def from_image(
        self,
        image,
        name=None,
    ):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
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
        return imaging.SimulatedImaging.simulate(
            image=image,
            exposure_time=self.exposure_time,
            psf=self.psf,
            background_sky_level=self.background_sky_level,
            add_noise=self.add_noise,
            noise_if_add_noise_false=self.noise_if_add_noise_false,
            noise_seed=self.noise_seed,
            name=name,
        )
