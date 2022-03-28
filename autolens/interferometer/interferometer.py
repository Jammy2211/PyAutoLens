import autoarray as aa

from autolens.lens.ray_tracing import Tracer


class SimulatorInterferometer(aa.SimulatorInterferometer):
    def __init__(
        self,
        uv_wavelengths,
        exposure_time: float,
        transformer_class=aa.TransformerDFT,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_native
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time_map
            The exposure time of an observation using this data.
        """

        super().__init__(
            uv_wavelengths=uv_wavelengths,
            exposure_time=exposure_time,
            transformer_class=transformer_class,
            noise_sigma=noise_sigma,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def via_tracer_from(self, tracer, grid, name=None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales
            The scale of each pixel in arc seconds
        exposure_time_map
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        add_poisson_noise: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = tracer.image_2d_from(grid=grid)

        return self.via_image_from(image=image.binned, name=name)

    def via_galaxies_from(self, galaxies, grid, name=None):
        """Simulate imaging data for this data, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data instance."""

        tracer = Tracer.from_galaxies(galaxies=galaxies)

        return self.via_tracer_from(tracer=tracer, grid=grid, name=name)

    def via_deflections_and_galaxies_from(self, deflections, galaxies, name=None):

        grid = aa.Grid2D.uniform(
            shape_native=deflections.shape_native,
            pixel_scales=deflections.pixel_scales,
            sub_size=1,
        )

        deflected_grid = grid - deflections.binned

        image = sum(map(lambda g: g.image_2d_from(grid=deflected_grid), galaxies))

        return self.via_image_from(image=image, name=name)
