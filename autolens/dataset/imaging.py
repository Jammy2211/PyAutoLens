from autoarray.dataset import imaging
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures import kernel_2d
from autolens.lens import ray_tracing


class SimulatorImaging(imaging.SimulatorImaging):
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: kernel_2d.Kernel2D = None,
        normalize_psf: bool = True,
        read_noise: float = None,
        add_poisson_noise: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        psf : Kernel2D
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of the simulated imaging.
        background_sky_level : float
            The level of the background sky of the simulated imaging.
        normalize_psf : bool
            If `True`, the PSF kernel is normalized so all values sum to 1.0.
        read_noise : float
            The level of read-noise added to the simulated imaging by drawing from a Gaussian distribution with
            sigma equal to the value `read_noise`.
        add_poisson_noise : bool
            Whether Poisson noise corresponding to photon count statistics on the imaging observation is added.
        noise_if_add_noise_false : float
            If noise is not added to the simulated dataset a `noise_map` must still be returned. This value gives
            the value of noise assigned to every pixel in the noise-map.
        noise_seed : int
            The random seed used to add random noise, where -1 corresponds to a random seed every run.
        """

        super(SimulatorImaging, self).__init__(
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            normalize_psf=normalize_psf,
            read_noise=read_noise,
            add_poisson_noise=add_poisson_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_tracer_and_grid(self, tracer, grid, name=None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : np.ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales: float
            The scale of each pixel in arc seconds
        exposure_time_map : np.ndarray
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : np.ndarray
            The value of background sky in every image pixel (electrons per second).
        add_poisson_noise: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = tracer.padded_image_2d_from_grid_and_psf_shape(
            grid=grid, psf_shape_2d=self.psf.shape_native
        )

        imaging = self.from_image(image=image.binned, name=name)

        return imaging.trimmed_after_convolution_from(
            kernel_shape=self.psf.shape_native
        )

    def from_galaxies_and_grid(self, galaxies, grid, name=None):
        """Simulate imaging data for this data, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data instance."""

        tracer = ray_tracing.Tracer.from_galaxies(galaxies=galaxies)

        return self.from_tracer_and_grid(tracer=tracer, grid=grid, name=name)

    def from_deflections_and_galaxies(self, deflections, galaxies, name=None):

        grid = grid_2d.Grid2D.uniform(
            shape_native=deflections.shape_native,
            pixel_scales=deflections.pixel_scales,
            sub_size=1,
        )

        deflected_grid = grid - deflections.binned

        image = sum(map(lambda g: g.image_2d_from_grid(grid=deflected_grid), galaxies))

        return self.from_image(image=image, name=name)
