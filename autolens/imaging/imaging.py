from typing import List

import autoarray as aa
import autogalaxy as ag

from autolens.lens.ray_tracing import Tracer

class SimulatorImaging(aa.SimulatorImaging):

    def via_tracer_from(self, tracer : Tracer, grid : aa.type.Grid2DLike) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input tracer and grid.

        The tracer is used to perform ray-tracing and generate the image of the strong lens galaxies (e.g.
        the lens light, lensed source light, etc) which is simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        tracer
            The tracer, which describes the ray-tracing and strong lens configuration used to simulate the imaging
            dataset.
        grid
            The image-plane grid which the image of the strong lens is generated on.
        """

        tracer.set_snr_of_snr_light_profiles(
            grid=grid,
            exposure_time=self.exposure_time,
            background_sky_level=self.background_sky_level,
        )

        image = tracer.padded_image_2d_from(
            grid=grid, psf_shape_2d=self.psf.shape_native
        )

        dataset = self.via_image_from(image=image.binned)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.shape_native
        )

    def via_galaxies_from(self, galaxies : List[ag.Galaxy], grid : aa.type.Grid2DLike) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input list of galaxies and grid.

        The galaxies are used to create a tracer, which performs ray-tracing and generate the image of the strong lens
        galaxies (e.g. the lens light, lensed source light, etc) which is simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        galaxies
            The galaxies used to create the tracer, which describes the ray-tracing and strong lens configuration
            used to simulate the imaging dataset.
        grid
            The image-plane grid which the image of the strong lens is generated on.
        """

        tracer = Tracer.from_galaxies(galaxies=galaxies)

        return self.via_tracer_from(tracer=tracer, grid=grid)

    def via_deflections_and_galaxies_from(self, deflections : aa.VectorYX2D, galaxies : List[ag.Galaxy]) -> aa.Imaging:
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
            The image-plane grid which the image of the strong lens is generated on.
        """
        grid = aa.Grid2D.uniform(
            shape_native=deflections.shape_native,
            pixel_scales=deflections.pixel_scales,
            sub_size=1,
        )

        deflected_grid = grid - deflections.binned

        image = sum(map(lambda g: g.image_2d_from(grid=deflected_grid), galaxies))

        return self.via_image_from(image=image)

    def via_source_image_from(self, tracer : Tracer, grid : aa.type.Grid2DLike, source_image : aa.Array2D) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input image of a source galaxy.

        This input image is on a uniform and regular 2D array, meaning it can simulate the source's irregular
        and assymetric source galaxy morphological features.

        The typical use case is inputting the image of an irregular galaxy in the source-plane (whose values are
        on a uniform array) and using this function computing the lensed image of this source galaxy.

        The tracer is used to perform ray-tracing and generate the image of the strong lens galaxies (e.g.
        the lens light, lensed source light, etc) which is simulated.

        The source galaxy light profiles are ignored in favour of the input source image, but the emission of
        other galaxies (e.g. the lems galaxy's light) are included.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        tracer
            The tracer, which describes the ray-tracing and strong lens configuration used to simulate the imaging
            dataset.
        grid
            The image-plane grid which the image of the strong lens is generated on.
        source_image
            The image of the source-plane and source galaxy which is interpolated to compute the lensed image.
        """

        image = tracer.image_2d_via_input_plane_image_from(
            grid=grid,
            plane_image=source_image
        ).binned

        padded_image = image.padded_before_convolution_from(
            kernel_shape=self.psf.shape_native
        )
        dataset = self.via_image_from(image=padded_image)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.shape_native
        )