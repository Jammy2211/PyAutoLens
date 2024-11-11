import os
import numpy as np
from typing import Optional, Union

import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.result import ResultDataset as AgResultDataset

from autolens.analysis.positions import PositionsLHResample
from autolens.analysis.positions import PositionsLHPenalty
from autolens.point.fit.positions.source.max_separation import (
    FitPositionsSourceMaxSeparation,
)
from autolens.lens.tracer import Tracer
from autolens.point.solver import PointSolver
from autolens.point.solver.shape_solver import ShapeSolver

class Result(AgResultDataset):
    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.
        """
        return self.analysis.tracer_via_instance_from(instance=self.instance)

    @property
    def max_log_likelihood_positions_threshold(self) -> float:
        """
        If the `Analysis` has a `PositionsLH` object this add a penalty term to the likelihood of the
        `log_likelihood_function`.

        This term is computed by ray-tracing a set of multiple image positions (E.g. corresponding to the lensed
        sources brightest pixels) to the source-plane and computing their maximum separation. This separation is
        then combined with a threshold to compute the likelihood term.

        This property returns the maximum separation of this `Analysis` object's multiple image positions for the
        maximum log likelihood tracer.

        It therefore provides information on how closely the lens model was able to ray trace the multiple images to
        one another in the source plane, and can be used for setting up the `PositionsLH` object in subsequent fits.

        Returns
        -------

        """
        positions_fits = FitPositionsSourceMaxSeparation(
            data=self.analysis.positions_likelihood.positions,
            noise_map=None,
            tracer=self.max_log_likelihood_tracer,
        )

        return positions_fits.max_separation_of_source_plane_positions

    @property
    def source_plane_light_profile_centre(self) -> aa.Grid2DIrregular:
        """
        Return a light profile centre of one of the a galaxies in the maximum log likelihood `Tracer`'s source-plane.
        If there are multiple light profiles, the first light profile's centre is returned.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        centre = self.max_log_likelihood_tracer.planes[-1].extract_attribute(
            cls=ag.LightProfile, attr_name="centre"
        )
        if centre is not None:
            return aa.Grid2DIrregular(values=[np.asarray(centre[0])])

    @property
    def source_plane_centre(self) -> aa.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `LEq` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        return self.source_plane_light_profile_centre

    @property
    def image_plane_multiple_image_positions(self) -> aa.Grid2DIrregular:
        """
        Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane.

        These image-plane positions are used by the next search in a pipeline if automatic position updating is turned
        on."""

        grid = self.analysis.dataset.mask.derive_grid.all_false

        solver = PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=0.001,
        )

        multiple_images = solver.solve(
            tracer=self.max_log_likelihood_tracer,
            source_plane_coordinate=self.source_plane_centre.in_list[0],
        )

        if multiple_images.shape[0] == 1:
            return self.image_plane_multiple_image_positions_for_single_image_from(image_0=multiple_images)

        return aa.Grid2DIrregular(values=multiple_images)

    def image_plane_multiple_image_positions_for_single_image_from(self, image_0) -> aa.Grid2DIrregular:
        """
        If the standard point solver only locates one multiple image, finds a second image which is not technically
        a multiple image in the point source regime but is close to it.

        This is performed by placing a circle at the source-plane centre and growing it until the `ShapeSolver`
        finds a second multiple image which is a sufficiently far enough distance away from the first multiple image.

        Parameters
        ----------
        image_0
            The first multiple image of the source-plane centre computed via the standard point solver.
        """

        grid = self.analysis.dataset.mask.derive_grid.all_false

        solver = ShapeSolver.for_grid(
            grid=grid,
            pixel_scale_precision=0.001,
        )

        centre = self.source_plane_centre.in_list[0]

        triangles = solver.solve_triangles(
            tracer=self.max_log_likelihood_tracer,
            shape=aa.Circle(y=centre[0], x=centre[1], radius=0.01)
        )

        multiple_images = triangles.centres

        distances = np.sum((multiple_images - image_0) ** 2, axis=1)

        print(np.min(distances))

        hgjjhjh

        furthest_index = np.argmax(distances)
        image_1 = multiple_images[furthest_index]

        print(image_0[0], image_1)

        return aa.Grid2DIrregular(values=[image_0[0], image_1])

    def positions_threshold_from(
        self,
        factor=1.0,
        minimum_threshold=None,
        positions: Optional[aa.Grid2DIrregular] = None,
    ) -> float:
        """
        Compute a new position threshold from these results corresponding to the image-plane multiple image positions
        of the maximum log likelihood `Tracer` ray-traced to the source-plane.

        First, we ray-trace forward the multiple-image's to the source-plane via the mass model to determine how far
        apart they are separated. We take the maximum source-plane separation of these points and multiple this by
        the auto_positions_factor to determine a new positions threshold. This value may also be rounded up to the
        input `auto_positions_minimum_threshold`.

        This is used for non-linear search chaining, specifically updating the position threshold of a new model-fit
        using the maximum likelihood model of a previous search.

        Parameters
        ----------
        factor
            The value the computed threshold is multiplied by to make the position threshold larger or smaller than the
            maximum log likelihood model's threshold.
        minimum_threshold
            The output threshold is rounded up to this value if it is below it, to avoid extremely small threshold
            values.
        positions
            If input, these positions are used instead of the computed multiple image positions from the lens mass
            model.

        Returns
        -------
        float
            The maximum source plane separation of this results maximum likelihood `Tracer` multiple images multiplied
            by `factor` and rounded up to the `threshold`.
        """

        positions = (
            self.image_plane_multiple_image_positions
            if positions is None
            else positions
        )

        tracer = Tracer(galaxies=self.max_log_likelihood_galaxies)

        positions_fits = FitPositionsSourceMaxSeparation(
            data=positions, noise_map=None, tracer=tracer
        )

        threshold = factor * np.max(
            positions_fits.max_separation_of_source_plane_positions
        )

        if minimum_threshold is not None:
            if threshold < minimum_threshold:
                return minimum_threshold

        return threshold

    def positions_likelihood_from(
        self,
        factor=1.0,
        minimum_threshold=None,
        use_resample=False,
        positions: Optional[aa.Grid2DIrregular] = None,
    ) -> Union[PositionsLHPenalty, PositionsLHResample]:
        """
        Returns a `PositionsLH` object from the result of a lens model-fit, where the maximum log likelihood mass
        and source models are used to determine the multiple image positions in the image-plane and source-plane
        and ray-trace them to the source-plane to determine the threshold.

        In chained fits, for example the SLaM pipelines, this means that a simple initial fit (e.g. SIE mass model,
        parametric source) can be used to determine the multiple image positions and threshold for a more complex
        subsequent fit (e.g. power-law mass model, pixelized source).

        Parameters
        ----------
        factor
            The value the computed threshold is multiplied by to make the position threshold larger or smaller than the
            maximum log likelihood model's threshold.
        minimum_threshold
            The output threshold is rounded up to this value if it is below it, to avoid extremely small threshold
            values.
        use_resample
            If `False` the `PositionsLH` object is created using the `PositionsLHPenalty` class, which uses the
            threshold to apply a penalty term to the likelihood. If `True` the `PositionsLH` object is created using
            the `PositionsLHResample` class, which resamples the positions to the threshold.
        positions
            If input, these positions are used instead of the computed multiple image positions from the lens mass
            model.

        Returns
        -------
        The `PositionsLH` object used to apply a likelihood penalty or resample the positions.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return None

        positions = (
            self.image_plane_multiple_image_positions
            if positions is None
            else positions
        )
        threshold = self.positions_threshold_from(
            factor=factor, minimum_threshold=minimum_threshold, positions=positions
        )

        if not use_resample:
            return PositionsLHPenalty(positions=positions, threshold=threshold)
        return PositionsLHResample(positions=positions, threshold=threshold)


class ResultDataset(Result):
    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.

        If a dataset is fitted the adapt images of the adapt image must first be associated with each galaxy.
        """
        instance = self.analysis.instance_with_associated_adapt_images_from(
            instance=self.instance
        )

        return self.analysis.tracer_via_instance_from(instance=instance)

    @property
    def positions(self):
        """
        The (y,x) arc-second coordinates of the lensed sources brightest pixels, which are used for discarding mass
        models which do not trace within a threshold in the source-plane of one another.
        """
        if self.analysis.positions_likelihood is not None:
            return self.analysis.positions_likelihood.positions

    @property
    def source_plane_centre(self) -> aa.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `LEq` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        if self.source_plane_inversion_centre is not None:
            return self.source_plane_inversion_centre
        elif self.source_plane_light_profile_centre is not None:
            return self.source_plane_light_profile_centre

    @property
    def source_plane_inversion_centre(self) -> aa.Grid2DIrregular:
        """
        Returns the centre of the brightest source pixel(s) of an `LEq`.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        if self.max_log_likelihood_fit.inversion is not None:
            if self.max_log_likelihood_fit.inversion.has(cls=aa.AbstractMapper):
                inversion = self.max_log_likelihood_fit.inversion
                mapper = inversion.cls_list_from(cls=aa.AbstractMapper)[0]

                mapper_valued = aa.MapperValued(
                    values=inversion.reconstruction_dict[mapper],
                    mapper=mapper,
                )

                return mapper_valued.max_pixel_centre
