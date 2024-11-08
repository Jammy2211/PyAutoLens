import logging
from typing import Tuple, Optional

from autoarray.numpy_wrapper import np

import autoarray as aa
from autoarray.numpy_wrapper import use_jax
from autoarray.structures.triangles.shape import Point

from autofit.jax_wrapper import jit, register_pytree_node_class
from autogalaxy import OperateDeflections
from .shape_solver import AbstractSolver


logger = logging.getLogger(__name__)


@register_pytree_node_class
class PointSolver(AbstractSolver):
    @jit
    def solve(
        self,
        tracer: OperateDeflections,
        source_plane_coordinate: Tuple[float, float],
        source_plane_redshift: Optional[float] = None,
        neighbor_order: int = 0,
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
        neighbor_order
            The number of times recursively add neighbors for the triangles that contain
            the source plane coordinate.
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

        for _ in range(neighbor_order):
            kept_triangles = kept_triangles.neighborhood()

        filtered_means = self._filter_low_magnification(
            tracer=tracer, points=kept_triangles.means
        )
        if use_jax:
            return aa.Grid2DIrregular([pair for pair in filtered_means])

        filtered_means = [
            pair for pair in filtered_means if not np.any(np.isnan(pair)).all()
        ]

        difference = len(kept_triangles.means) - len(filtered_means)
        if difference > 0:
            logger.debug(
                f"Filtered one multiple-image with magnification below threshold."
            )
        elif difference > 1:
            logger.warning(
                f"Filtered {difference} multiple-images with magnification below threshold."
            )

        filtered_close = []

        for mean in filtered_means:
            if any(
                np.linalg.norm(np.array(mean) - np.array(other))
                <= self.pixel_scale_precision / 2
                for other in filtered_close
            ):
                continue
            filtered_close.append(mean)

        return aa.Grid2DIrregular(
            [pair for pair in filtered_close if not np.isnan(pair).all()]
        )
