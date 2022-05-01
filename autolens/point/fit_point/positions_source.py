from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.ray_tracing import Tracer

from autolens import exc


class FitPositionsSource(aa.FitDataset):
    def __init__(
        self,
        name: str,
        positions: aa.Grid2DIrregular,
        noise_map: aa.ValuesIrregular,
        tracer: Tracer,
        point_profile: Optional[ag.ps.Point] = None,
    ):
        """
        A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        super().__init__(dataset=positions)

        self.name = name
        self._noise_map = noise_map
        self.tracer = tracer

        self.point_profile = (
            tracer.extract_profile(profile_name=name)
            if point_profile is None
            else point_profile
        )

        if self.point_profile is None:
            raise exc.PointExtractionException(
                f"For the point-source named {name} there was no matching point source profile "
                f"in the tracer (make sure your tracer's point source name is the same the dataset name."
            )

        self.source_plane_coordinate = self.point_profile.centre

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
        if len(self.tracer.planes) <= 2:

            deflections = self.tracer.deflections_yx_2d_from(grid=self.positions)

        else:

            upper_plane_index = self.tracer.extract_plane_index_of_profile(
                profile_name=self.name
            )

            deflections = self.tracer.deflections_between_planes_from(
                grid=self.positions, plane_i=0, plane_j=upper_plane_index
            )

        return self.positions.grid_2d_via_deflection_grid_from(
            deflection_grid=deflections
        )

    @property
    def model_positions(self) -> aa.Grid2DIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ValuesIrregular:

        return self.model_positions.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )
