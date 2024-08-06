from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.point.fit_point.positions.abstract import AbstractFitPositionsImagePair
from autolens.point.solver import PointSolver
from autolens.lens.tracer import Tracer

from autolens import exc


class FitPositionsImagePairRepeat(AbstractFitPositionsImagePair):
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

    @property
    def residual_map(self) -> aa.ArrayIrregular:

        residual_map = []

        for position in self.data:

            distances = [
                self.square_distance(model_position, position)
                for model_position in self.model_data
            ]
            residual_map.append(min(distances))

        return aa.ArrayIrregular(values=residual_map)

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map,
        )
