import jax.numpy as jnp
import logging
from typing import Tuple, Optional

import autoarray as aa
from autoarray.structures.triangles.shape import Point

from autofit.jax_wrapper import register_pytree_node_class
from autogalaxy import OperateDeflections
from .shape_solver import AbstractSolver


logger = logging.getLogger(__name__)


@register_pytree_node_class
class PointSolver(AbstractSolver):

    def solve(
        self,
        tracer: OperateDeflections,
        source_plane_coordinate: Tuple[float, float],
        plane_redshift: Optional[float] = None,
    ) -> aa.Grid2DIrregular:
        """
        Solve for the image plane coordinates that are traced to the source plane coordinate.

        This is done by tiling the image plane with triangles and checking if the source plane coordinate is contained
        within the triangle. The triangles are sub-sampled to increase the resolution with only the triangles that
        contain the source plane coordinate and their neighbours being kept.

        The means of the triangles  are then filtered to keep only those with an absolute magnification above the
        threshold.

        Parameters
        ----------
        source_plane_coordinate
            The plane coordinate to trace to the image plane, which by default in the source-plane coordinate
            but could be a coordinate in another plane is `plane_redshift` is input.
        tracer
            The tracer that traces the image plane coordinates to the source plane
        plane_redshift
            The redshift of the plane coordinate, which for multi-plane systems may not be the source-plane.

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        kept_triangles = super().solve_triangles(
            tracer=tracer,
            shape=Point(*source_plane_coordinate),
            plane_redshift=plane_redshift,
        )

        filtered_means = self._filter_low_magnification(
            tracer=tracer, points=kept_triangles.means
        )

        arr = aa.Grid2DIrregular([pair for pair in filtered_means])

        mask = ~jnp.isnan(arr.array).any(axis=1)
        return aa.Grid2DIrregular(arr.array[mask])
