from abc import ABC
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.point.fit.abstract import AbstractFitPoint
from autolens.point.solver import PointSolver
from autolens.lens.tracer import Tracer


class AbstractFitPositions(AbstractFitPoint, ABC):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: PointSolver,
        profile: Optional[ag.ps.Point] = None,
        xp=np,
    ):
        """
        Abstract class to fit the positions of a a point source dataset using a `Tracer` object, where the specific
        implementation of the chi-squared is defined in the sub-class.

        The fit performs the following steps:

        1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
           as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

        2) Using the sub-class specific chi-squared, compute the residuals of each image-plane position, chi-squared
           and overall log likelihood of the fit.

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
            triangles to and from the source-plane.
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
    def positions(self):
        return self.data
