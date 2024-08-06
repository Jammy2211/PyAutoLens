from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer
from autolens.point.fit.positions.abstract import AbstractFitPositions


class FitPositionsSource(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
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
            profile=profile,
        )

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.

        It if common for many more image-plane positions to be computed than actual positions in the dataset. In this
        case, each data point is paired with its closest model position.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(grid=self.data)
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, plane_i=0, plane_j=self.source_plane_index
            )

        return self.data.grid_2d_via_deflection_grid_from(
            deflection_grid=deflections
        )

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )
