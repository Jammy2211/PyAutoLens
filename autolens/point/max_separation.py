from typing import Optional

import autoarray as aa

from autolens.lens.tracer import Tracer


class SourceMaxSeparation:
    def __init__(
        self,
        data: aa.Grid2DIrregular,
        noise_map: Optional[aa.ArrayIrregular],
        tracer: Tracer,
    ):
        """
        Given a positions dataset, which is a list of positions with names that associated them to model source
        galaxies, use a `Tracer` to determine the traced coordinate positions in the source-plane.

        Different children of this abstract class are available which use the traced coordinates to define a chi-squared
        value in different ways.

        Parameters
        ----------
        data : Grid2DIrregular
            The (y,x) arc-second coordinates of named positions which the log_likelihood is computed using. Positions
            are paired to galaxies in the `Tracer` using their names.
        tracer : Tracer
            The object that defines the ray-tracing of the strong lens system of galaxies.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        self.data = data
        self.noise_map = noise_map

        traced_grid_2d_list = tracer.traced_grid_2d_list_from(
            grid=aa.Grid2DIrregular(data)
        )

        self.source_plane_positions = aa.Grid2DIrregular(values=traced_grid_2d_list[-1])

    @property
    def furthest_separations_of_source_plane_positions(self) -> aa.ArrayIrregular:
        """
        Returns the furthest distance of every source-plane (y,x) coordinate to the other source-plane (y,x)
        coordinates.

        For example, for the following source-plane positions:

        source_plane_positions = [[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)]

        The returned furthest distances are:

        source_plane_positions = [3.0, 2.0, 3.0]

        Returns
        -------
        aa.ArrayIrregular
            The further distances of every set of grouped source-plane coordinates the other source-plane coordinates
            that it is grouped with.
        """
        return self.source_plane_positions.furthest_distances_to_other_coordinates

    @property
    def max_separation_of_source_plane_positions(self) -> float:
        return max(self.furthest_separations_of_source_plane_positions)

    def max_separation_within_threshold(self, threshold) -> bool:
        return self.max_separation_of_source_plane_positions <= threshold
