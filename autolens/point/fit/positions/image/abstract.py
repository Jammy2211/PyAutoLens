from typing import Optional, Tuple

import autoarray as aa
import autogalaxy as ag

from autolens.point.solver import PointSolver
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.lens.tracer import Tracer


class AbstractFitPositionsImagePair(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: PointSolver,
        profile: Optional[ag.ps.Point] = None,
    ):
        """
        A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        ----------
        data : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
        )

    @staticmethod
    def square_distance(coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.
        """
        return self.solver.solve(
            tracer=self.tracer,
            source_plane_coordinate=self.source_plane_coordinate,
            source_plane_redshift=self.source_plane_redshift,
        )
