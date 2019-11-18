from autoarray.structures import grids
from autoarray.simulator import simulator
from autolens.lens import ray_tracing


class ImagingSimulator(simulator.ImagingSimulator):
    def __init__(
        self,
        shape_2d,
        pixel_scales,
        sub_size,
        psf,
        exposure_time,
        background_level,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        origin=(0.0, 0.0),
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of an observation using this data_type.
        background_level : float
            The level of the background sky of an observationg using this data_type.
        """

        super(ImagingSimulator, self).__init__(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            psf=psf,
            exposure_time=exposure_time,
            background_level=background_level,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
            origin=origin,
        )

    def from_tracer(self, tracer, name=None):
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
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = tracer.padded_profile_image_from_grid_and_psf_shape(
            grid=self.grid, psf_shape_2d=self.psf.shape_2d
        )

        return self.from_image(image=image.in_1d_binned, name=name)

    def from_galaxies(self, galaxies):
        """Simulate imaging data for this data_type, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data_type instance."""

        tracer = ray_tracing.Tracer.from_galaxies(galaxies=galaxies)

        return self.from_tracer(tracer=tracer)

    def from_deflections_and_galaxies(self, deflections, galaxies):

        grid = grids.Grid.uniform(
            shape_2d=deflections.shape_2d,
            pixel_scales=deflections.pixel_scales,
            sub_size=1,
        )

        deflected_grid = grid - deflections.in_1d_binned

        image = sum(
            map(lambda g: g.profile_image_from_grid(grid=deflected_grid), galaxies)
        )

        return self.from_image(image=image)


class InterferometerSimulator(simulator.InterferometerSimulator):
    def __init__(
        self,
        real_space_shape_2d,
        real_space_pixel_scales,
        uv_wavelengths,
        sub_size,
        exposure_time,
        background_level,
        primary_beam=None,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        origin=(0.0, 0.0),
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of an observation using this data_type.
        background_level : float
            The level of the background sky of an observationg using this data_type.
        """

        super(InterferometerSimulator, self).__init__(
            real_space_shape_2d=real_space_shape_2d,
            real_space_pixel_scales=real_space_pixel_scales,
            uv_wavelengths=uv_wavelengths,
            sub_size=sub_size,
            exposure_time=exposure_time,
            background_level=background_level,
            primary_beam=primary_beam,
            noise_sigma=noise_sigma,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
            origin=origin,
        )

    def from_tracer(self, tracer):
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
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = tracer.profile_image_from_grid(grid=self.grid)

        return self.from_real_space_image(real_space_image=image.in_1d_binned)

    def from_galaxies(self, galaxies):
        """Simulate imaging data for this data_type, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data_type instance."""

        tracer = ray_tracing.Tracer.from_galaxies(galaxies=galaxies)

        return self.from_tracer(tracer=tracer)

    def from_deflections_and_galaxies(self, deflections, galaxies):

        grid = grids.Grid.uniform(
            shape_2d=deflections.shape_2d,
            pixel_scales=deflections.pixel_scales,
            sub_size=1,
        )

        deflected_grid = grid - deflections.in_1d_binned

        image = sum(
            map(lambda g: g.profile_image_from_grid(grid=deflected_grid), galaxies)
        )

        return self.from_real_space_image(real_space_image=image)
