import os
from os import path
import numpy as np
import json
from typing import Optional, Union

import autoarray as aa
import autogalaxy as ag

from autofit.non_linear.paths.abstract import AbstractPaths
from autogalaxy.analysis.result import ResultDataset as AgResultDataset

from autolens.analysis.positions import PositionsLHResample
from autolens.analysis.positions import PositionsLHPenalty
from autolens.point.fit_point.max_separation import FitPositionsSourceMaxSeparation
from autolens.lens.ray_tracing import Tracer
from autolens.point.point_solver import PointSolver


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
            positions=self.analysis.positions_likelihood.positions,
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
        centre = self.max_log_likelihood_tracer.source_plane.extract_attribute(
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

        grid = self.analysis.dataset.mask.derive_grid.all_false_sub_1

        solver = PointSolver(
            grid=grid, pixel_scale_precision=0.001, distance_to_mass_profile_centre=0.05
        )

        multiple_images = solver.solve(
            lensing_obj=self.max_log_likelihood_tracer,
            source_plane_coordinate=self.source_plane_centre.in_list[0],
        )

        return aa.Grid2DIrregular(values=multiple_images)

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

        positions_fits = FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=None, tracer=self.max_log_likelihood_tracer
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
                return self.max_log_likelihood_fit.inversion.brightest_reconstruction_pixel_centre_list[
                    0
                ]
