import logging

from typing import Tuple, List, Iterator, Optional

import numpy as np

import autoarray as aa

from autofit.jax_wrapper import jit, register_pytree_node_class
from .abstract_solver import AbstractSolver


from autolens.lens.tracer import Tracer
from .step import Step

logger = logging.getLogger(__name__)


@register_pytree_node_class
class PointSolver(AbstractSolver):
    # noinspection PyMethodOverriding
    def _filter_indexes(
        self,
        source_triangles: aa.AbstractTriangles,
        source_plane_coordinate: Tuple[float, float],
    ) -> np.ndarray:
        return source_triangles.containing_indices(point=source_plane_coordinate)

    @jit
    def solve(
        self,
        tracer: Tracer,
        source_plane_coordinate: Tuple[float, float],
        source_plane_redshift: Optional[float] = None,
    ) -> aa.Grid2DIrregular:
        """
        Solve for the image plane coordinates that are traced to the source plane coordinate.

        This is done by tiling the image plane with triangles and checking if the source plane coordinate is contained
        within the triangle. The triangles are subsampled to increase the resolution with only the triangles that
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
        return super().solve(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            source_plane_redshift=source_plane_redshift,
        )

    # noinspection PyMethodOverriding
    def steps(
        self,
        tracer: Tracer,
        source_plane_coordinate: Tuple[float, float],
        source_plane_redshift: Optional[float] = None,
        **kwargs,
    ) -> Iterator[Step]:
        """
        Iterate over the steps of the triangle solver algorithm.

        Parameters
        ----------
        tracer
            The tracer that traces from the image plane to the source plane.
        source_plane_coordinate
        source_plane_redshift
            The redshift of the source plane.

        Returns
        -------
        An iterator over the steps of the triangle solver algorithm.
        """
        yield from super().steps(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            source_plane_redshift=source_plane_redshift,
            **kwargs,
        )
