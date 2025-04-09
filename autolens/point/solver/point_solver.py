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
        source_plane_redshift: Optional[float] = None,
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
            The source plane coordinate to trace to the image plane.
        tracer
            The tracer that traces the image plane coordinates to the source plane
        source_plane_redshift
            The redshift of the source plane coordinate.

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        kept_triangles = super().solve_triangles(
            tracer=tracer,
            shape=Point(*source_plane_coordinate),
            source_plane_redshift=source_plane_redshift,
        )

        filtered_means = self._filter_low_magnification(
            tracer=tracer, points=kept_triangles.means
        )

        return aa.Grid2DIrregular([pair for pair in filtered_means])
