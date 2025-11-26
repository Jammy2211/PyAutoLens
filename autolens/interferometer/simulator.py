from typing import List

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer


class SimulatorInterferometer(aa.SimulatorInterferometer):
    def via_tracer_from(self, tracer, grid):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales
            The scale of each pixel in arc seconds
        exposure_time_map
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        add_poisson_noise_to_data: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = tracer.image_2d_from(grid=grid)

        return self.via_image_from(image=image)

    def via_galaxies_from(self, galaxies, grid):
        """Simulate imaging data for this data, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data instance."""

        tracer = Tracer(galaxies=galaxies)

        return self.via_tracer_from(tracer=tracer, grid=grid)

    def via_deflections_and_galaxies_from(
        self, deflections: aa.VectorYX2D, galaxies: List[ag.Galaxy]
    ) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input deflection angle map and list of galaxies.

        The input deflection angle map ray-traces the image-plane coordinates from the image-plane to source-plane,
        via the lens equation.

        This traced grid is then used to evaluate the light of the list of galaxies, which therefore simulate the
        image of the strong lens.

        This function is used in situations where one has access to a deflection angle map which does not suit being
        ray-traced using a `Tracer` object (e.g. deflection angles from a cosmological simulation of a galaxy).

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        galaxies
            The galaxies used to create the tracer, which describes the ray-tracing and strong lens configuration
            used to simulate the imaging dataset.
        grid
            The image-plane 2D grid of (y,x) coordinates grid which the image of the strong lens is generated on.
        """
        grid = aa.Grid2D.uniform(
            shape_native=deflections.shape_native,
            pixel_scales=deflections.pixel_scales,
            over_sample_size=1,
        )

        deflected_grid = aa.Grid2D(
            values=grid - deflections,
            mask=grid.mask,
            over_sample_size=1,
            over_sampled=grid - deflections,
            over_sampler=grid.over_sampler,
        )

        image = sum(map(lambda g: g.image_2d_from(grid=deflected_grid), galaxies))

        return self.via_image_from(image=image)
