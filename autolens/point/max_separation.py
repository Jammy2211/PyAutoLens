import numpy as np
from typing import Optional

import autoarray as aa

from autolens.lens.tracer import Tracer


class SourceMaxSeparation:
    def __init__(
        self,
        data: aa.Grid2DIrregular,
        noise_map: Optional[aa.ArrayIrregular],
        tracer: Tracer,
        plane_redshift: float = Optional[None],
        xp=np,
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
        plane_redshift
            The redshift of the plane in the `Tracer` that the source-plane positions are computed from. This is
            often the last plane in the `Tracer`, which is the source-plane.
        """

        self.data = data
        self.noise_map = noise_map

        try:
            plane_index = tracer.plane_index_via_redshift_from(redshift=plane_redshift)
        except TypeError:
            plane_index = -1

        self.plane_positions = aa.Grid2DIrregular(
            values=tracer.traced_grid_2d_list_from(grid=data, xp=xp)[plane_index], xp=xp
        )

    @property
    def furthest_separations_of_plane_positions(self) -> aa.ArrayIrregular:
        """
        Returns the furthest distance of every source-plane (y,x) coordinate to the other source-plane (y,x)
        coordinates.

        For example, for the following plane positions:

        plane_positions = [[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)]

        The returned furthest distances are:

        plane_positions = [3.0, 2.0, 3.0]

        Returns
        -------
        aa.ArrayIrregular
            The further distances of every set of grouped source-plane coordinates the other source-plane coordinates
            that it is grouped with.
        """
        return self.plane_positions.furthest_distances_to_other_coordinates

    @property
    def max_separation_of_plane_positions(self) -> float:
        return max(self.furthest_separations_of_plane_positions)

    def max_separation_within_threshold(self, threshold) -> bool:
        return self.max_separation_of_plane_positions <= threshold
