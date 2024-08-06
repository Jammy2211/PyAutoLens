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
        positions: aa.Grid2DIrregular,
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
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        super().__init__(
            name=name,
            positions=positions,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
        )

    @property
    def mask(self):
        return None

    @property
    def noise_map(self):
        return self._noise_map

    @property
    def positions(self) -> aa.Grid2DIrregular:
        return self.dataset

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.

        It if common for many more image-plane positions to be computed than actual positions in the dataset. In this
        case, each data point is paired with its closest model position.
        """

        model_positions = self.solver.solve(
            tracer=self.tracer,
            source_plane_coordinate=self.source_plane_coordinate,
            source_plane_redshift=self.source_plane_redshift,
        )

        return model_positions.grid_of_closest_from(grid_pair=self.positions)

    @property
    def model_positions(self) -> aa.Grid2DIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        residual_positions = self.positions - self.model_positions

        return residual_positions.distances_to_coordinate_from(coordinate=(0.0, 0.0))

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map,
        )
