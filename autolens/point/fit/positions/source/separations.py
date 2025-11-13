import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.point.solver import PointSolver


class FitPositionsSource(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: Optional[PointSolver],
        profile: Optional[ag.ps.Point] = None,
        xp=np,
    ):
        """
        Fits the positions of a a point source dataset using a `Tracer` object with a source-plane chi-squared based on
        the separation of image-plane positions ray-traced to the source-plane compared to the centre of the source
        galaxy.

        The fit performs the following steps:

        1) Determine the source-plane centre of the source-galaxy, which could be a free model parameter or computed
           as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

        2) Ray-trace the positions in the point source to the source-plane via the `Tracer`, including accounting for
           multi-plane ray-tracing.

        3) Compute the distance of each ray-traced position to the source-plane centre and compute the residuals,

        4) Compute the magnification of each image-plane position via the Hessian of the tracer's deflection angles.

        5) Compute the residuals of each position as the difference between the source-plane centre and each
           ray-traced position.

        6) Compute the chi-squared of each position as the square of the residual multiplied by the magnification and
           divided by the RMS noise-map value.

        7) Sum the chi-squared values to compute the overall log likelihood of the fit.

        Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
        point source dataset to ensure that point source datasets are fitted to the correct point source.

        This fit object is used in the `FitPointDataset` to perform position based fitting of a `PointDataset`,
        which may also fit other components of the point dataset like fluxes or time delays.

        When performing a `model-fit`via an `AnalysisPoint` object the `figure_of_merit` of this object
        is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        name
            The name of the point source dataset which is paired to a `Point` profile.
        data
            The positions of the point source in the image-plane which are fitted.
        noise_map
            The noise-map of the positions which are used to compute the log likelihood of the positions.
        tracer
            The tracer of galaxies whose point source profile are used to fit the positions.
        solver
            Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
            triangles to and from the source-plane. This is not used in this source-plane point source fit.
        profile
            Manually input the profile of the point source, which is used instead of the one extracted from the
            tracer via name pairing if that profile is not found.
        """

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
            xp=xp,
        )

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the source-plane model positions of the point source, which are the positions of the image-plane
        positions ray-traced to the source-plane.

        This calculation accounts for multi-plane ray-tracing, whereby if the tracer has more than 2 planees the
        redshift of the point source galaxy is extracted and the deflections between the image-plane and source-plane
        at its specific redshift are used.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(
                grid=self.data, xp=self._xp
            )
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, xp=self._xp, plane_i=0, plane_j=self.plane_index
            )

        return self.data.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the residuals of the point-source source-plane fit, which are the distances of each source-plane
        position from the source-plane centre.
        """
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )

    @property
    def chi_squared_map(self) -> float:
        """
        Returns the chi-squared of the point-source source-plane fit, which is the sum of the squared residuals
        multiplied by the magnifications squared, divided by the noise-map values squared.
        """

        return self.residual_map**2.0 / (
            self.magnifications_at_positions.array**-2.0 * self.noise_map.array**2.0
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the normalization of the noise-map, which is the sum of the noise-map values squared.
        """
        return self._xp.sum(
            self._xp.log(
                2
                * np.pi
                * (
                    self.magnifications_at_positions.array**-2.0
                    * self.noise_map.array**2.0
                )
            )
        )

    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of the point-source source-plane fit, which is the sum of the chi-squared values.
        """
        return -0.5 * (sum(self.chi_squared_map) + self.noise_normalization)
