import logging
import math

from typing import Tuple, List, Iterator, Type, Optional

import autoarray as aa

from autoarray.structures.triangles.shape import Shape
from autofit.jax_wrapper import jit, use_jax, numpy as np, register_pytree_node_class

try:
    if use_jax:
        from autoarray.structures.triangles.jax_array import ArrayTriangles
    else:
        from autoarray.structures.triangles.array import ArrayTriangles
except ImportError:
    from autoarray.structures.triangles.array import ArrayTriangles

from autoarray.structures.triangles.abstract import AbstractTriangles

from autolens.lens.tracer import Tracer
from .step import Step

logger = logging.getLogger(__name__)


class AbstractSolver:
    # noinspection PyPep8Naming
    def __init__(
        self,
        scale: float,
        initial_triangles: AbstractTriangles,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
    ):
        """
        Determine the image plane coordinates that are traced to be a source plane coordinate.

        This is performed efficiently by iteratively subdividing the image plane into triangles and checking if the
        source plane coordinate is contained within the triangle. The triangles are subsampled to increase the
        resolution

        Parameters
        ----------
        pixel_scale_precision
            The target pixel scale of the image grid.
        """
        self.scale = scale
        self.pixel_scale_precision = pixel_scale_precision
        self.magnification_threshold = magnification_threshold

        self.initial_triangles = initial_triangles

    # noinspection PyPep8Naming
    @classmethod
    def for_grid(
        cls,
        grid: aa.Grid2D,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
        array_triangles_cls: Type[AbstractTriangles] = ArrayTriangles,
    ):
        scale = grid.pixel_scale

        y = grid[:, 0]
        x = grid[:, 1]

        y_min = y.min()
        y_max = y.max()
        x_min = x.min()
        x_max = x.max()

        initial_triangles = array_triangles_cls.for_limits_and_scale(
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            scale=scale,
        )

        return cls(
            scale=scale,
            initial_triangles=initial_triangles,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
        )

    @property
    def n_steps(self) -> int:
        """
        How many times should triangles be subdivided?
        """
        return math.ceil(math.log2(self.scale / self.pixel_scale_precision))

    @staticmethod
    def _source_plane_grid(
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
        source_plane_redshift: Optional[float] = None,
    ) -> aa.type.Grid2DLike:
        """
        Calculate the source plane grid from the image plane grid.

        Parameters
        ----------
        grid
            The image plane grid.

        Returns
        -------
        The source plane grid computed by applying the deflections to the image plane grid.
        """

        source_plane_index = -1

        if source_plane_redshift is not None:
            for redshift in tracer.plane_redshifts:
                source_plane_index += 1
                if redshift == source_plane_redshift:
                    break

        deflections = tracer.deflections_between_planes_from(
            grid=grid, plane_i=0, plane_j=source_plane_index
        )
        # noinspection PyTypeChecker
        return grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    @jit
    def solve_triangles(
        self,
        tracer: Tracer,
        shape: Shape,
        source_plane_redshift: Optional[float] = None,
    ) -> AbstractTriangles:
        """
        Solve for the image plane coordinates that are traced to the source plane coordinate.

        This is done by tiling the image plane with triangles and checking if the source plane coordinate is contained
        within the triangle. The triangles are subsampled to increase the resolution with only the triangles that
        contain the source plane coordinate and their neighbours being kept.

        The means of the triangles are then filtered to keep only those with an absolute magnification above the
        threshold.

        Parameters
        ----------
        tracer
            The tracer to use to trace the image plane coordinates to the source plane.
        shape
            The shape in the source plane for which we want to identify the image plane coordinates.
        source_plane_redshift
            The redshift of the source plane.

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        if self.n_steps == 0:
            raise ValueError(
                "The target pixel scale is too large to subdivide the triangles."
            )

        steps = list(
            self.steps(
                tracer=tracer,
                shape=shape,
                source_plane_redshift=source_plane_redshift,
            )
        )
        final_step = steps[-1]
        return final_step.filtered_triangles

    def _filter_low_magnification(
        self, tracer: Tracer, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Filter the points to keep only those with an absolute magnification above the threshold.

        Parameters
        ----------
        points
            The points to filter.

        Returns
        -------
        The points with an absolute magnification above the threshold.
        """
        points = np.array(points)
        magnifications = tracer.magnification_2d_via_hessian_from(
            grid=aa.Grid2DIrregular(points),
            buffer=self.scale,
        )
        mask = np.abs(magnifications.array) > self.magnification_threshold
        return np.where(mask[:, None], points, np.nan)

    def _filtered_triangles(
        self,
        tracer: Tracer,
        triangles: aa.AbstractTriangles,
        source_plane_redshift,
        shape: Shape,
    ):
        """
        Filter the triangles to keep only those that meet the solver condition
        """
        source_plane_grid = self._source_plane_grid(
            tracer=tracer,
            grid=aa.Grid2DIrregular(triangles.vertices),
            source_plane_redshift=source_plane_redshift,
        )
        source_triangles = triangles.with_vertices(source_plane_grid.array)

        indexes = source_triangles.containing_indices(shape=shape)

        return triangles.for_indexes(indexes=indexes)

    def steps(
        self,
        tracer: Tracer,
        shape: Shape,
        source_plane_redshift: Optional[float] = None,
    ) -> Iterator[Step]:
        """
        Iterate over the steps of the triangle solver algorithm.

        Parameters
        ----------
        tracer
            The tracer to use to trace the image plane coordinates to the source plane.
        source_plane_redshift
            The redshift of the source plane.
        shape
            The shape in the source plane for which we want to identify the image plane coordinates.

        Returns
        -------
        An iterator over the steps of the triangle solver algorithm.
        """
        initial_triangles = self.initial_triangles
        for number in range(self.n_steps):
            kept_triangles = self._filtered_triangles(
                tracer=tracer,
                triangles=initial_triangles,
                source_plane_redshift=source_plane_redshift,
                shape=shape,
            )
            neighbourhood = kept_triangles.neighborhood()
            up_sampled = neighbourhood.up_sample()

            yield Step(
                number=number,
                initial_triangles=initial_triangles,
                filtered_triangles=kept_triangles,
                neighbourhood=neighbourhood,
                up_sampled=up_sampled,
            )

            initial_triangles = up_sampled

    def tree_flatten(self):
        return (), (
            self.scale,
            self.pixel_scale_precision,
            self.magnification_threshold,
            self.initial_triangles,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            scale=aux_data[0],
            pixel_scale_precision=aux_data[1],
            magnification_threshold=aux_data[2],
            initial_triangles=aux_data[3],
        )


@register_pytree_node_class
class ShapeSolver(AbstractSolver):
    def find_magnification(
        self,
        tracer: Tracer,
        shape: Shape,
        source_plane_redshift: Optional[float] = None,
    ) -> float:
        """
        Find the magnification of the shape in the source plane.

        Parameters
        ----------
        tracer
            A tracer that traces the image plane to the source plane.
        shape
            The shape of an image plane pixel.
        source_plane_redshift
            The redshift of the source plane.

        Returns
        -------
        The magnification of the shape in the source plane.
        """
        kept_triangles = super().solve_triangles(
            tracer=tracer,
            shape=shape,
            source_plane_redshift=source_plane_redshift,
        )
        return kept_triangles.area / shape.area
