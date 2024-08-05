import logging
import math
from dataclasses import dataclass
from typing import Tuple, List, Iterator, Type

from autoarray import Grid2D, Grid2DIrregular
from autoarray.structures.triangles import array
from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.type import Grid2DLike
from autogalaxy import OperateDeflections

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """
    A step in the triangle solver algorithm.

    Attributes
    ----------
    number
        The number of the step.
    initial_triangles
        The triangles at the start of the step.
    filtered_triangles
        The triangles trace to triangles that contain the source plane coordinate.
    neighbourhood
        The neighbourhood of the filtered triangles.
    up_sampled
        The neighbourhood up-sampled to increase the resolution.
    """

    number: int
    initial_triangles: AbstractTriangles
    filtered_triangles: AbstractTriangles
    neighbourhood: AbstractTriangles
    up_sampled: AbstractTriangles


class PointSolver:
    # noinspection PyPep8Naming
    def __init__(
        self,
        pixel_scale_precision: float,
        magnification_threshold=0.1,
        ArrayTriangles: Type[AbstractTriangles] = array.ArrayTriangles,
    ):
        """
        Determine the image plane coordinates that are traced to be a source plane coordinate.

        This is performed efficiently by iteratively subdividing the image plane into triangles and checking if the
        source plane coordinate is contained within the triangle. The triangles are subsampled to increase the
        resolution

        Parameters
        ----------
        lensing_obj
            A tracer describing the lensing system.
        pixel_scale_precision
            The target pixel scale of the image grid.
        ArrayTriangles
            The class to use for the triangles.
        """

        self.pixel_scale_precision = pixel_scale_precision
        self.magnification_threshold = magnification_threshold
        self.ArrayTriangles = ArrayTriangles

    def n_steps_from(self, pixel_scale) -> int:
        """
        How many times should triangles be subdivided?
        """
        return math.ceil(math.log2(pixel_scale / self.pixel_scale_precision))

    def _source_plane_grid(self, lensing_obj : OperateDeflections, grid: Grid2DLike) -> Grid2DLike:
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
        deflections = lensing_obj.deflections_yx_2d_from(grid=grid)
        # noinspection PyTypeChecker
        return grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    def solve(
        self,
        lensing_obj : OperateDeflections,
        grid : Grid2DLike,
        source_plane_coordinate: Tuple[float, float]
    ) -> Grid2DIrregular:
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

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        if self.n_steps_from(pixel_scale=grid.pixel_scale) == 0:
            raise ValueError(
                "The target pixel scale is too large to subdivide the triangles."
            )

        steps = list(self.steps(
            lensing_obj=lensing_obj,
            grid=grid,
            source_plane_coordinate=source_plane_coordinate
        ))
        final_step = steps[-1]
        kept_triangles = final_step.filtered_triangles

        filtered_means = self._filter_low_magnification(
            lensing_obj=lensing_obj,
            pixel_scale=grid.pixel_scale,
            points=kept_triangles.means
        )

        difference = len(kept_triangles.means) - len(filtered_means)
        if difference > 0:
            logger.debug(
                f"Filtered one multiple-image with magnification below threshold."
            )
        elif difference > 1:
            logger.warning(
                f"Filtered {difference} multiple-images with magnification below threshold."
            )

        return Grid2DIrregular(values=filtered_means)

    def _filter_low_magnification(
        self,
        lensing_obj: OperateDeflections,
        pixel_scale: float,
        points: List[Tuple[float, float]]
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
        return [
            point
            for point, magnification in zip(
                points,
                lensing_obj.magnification_2d_via_hessian_from(
                    grid=Grid2DIrregular(points),
                    buffer=pixel_scale,
                ),
            )
            if abs(magnification) > self.magnification_threshold
        ]

    def _filter_triangles(
        self,
        lensing_obj: OperateDeflections,
        triangles: AbstractTriangles,
        source_plane_coordinate: Tuple[float, float],
    ):
        """
        Filter the triangles to keep only those that contain the source plane coordinate.

        Parameters
        ----------
        triangles
            A set of triangles that may contain the source plane coordinate.
        source_plane_coordinate
            The source plane coordinate to check if it is contained within the triangles.

        Returns
        -------
        The triangles that contain the source plane coordinate.
        """
        source_plane_grid = self._source_plane_grid(
            lensing_obj=lensing_obj,
            grid=Grid2DIrregular(triangles.vertices)
        )
        source_triangles = triangles.with_vertices(source_plane_grid.array)
        indexes = source_triangles.containing_indices(point=source_plane_coordinate)
        return triangles.for_indexes(indexes=indexes)

    def steps(
        self,
        lensing_obj: OperateDeflections,
        grid: Grid2DLike,
        source_plane_coordinate: Tuple[float, float],
    ) -> Iterator[Step]:
        """
        Iterate over the steps of the triangle solver algorithm.

        Parameters
        ----------
        source_plane_coordinate
            The source plane coordinate to trace to the image plane.

        Returns
        -------
        An iterator over the steps of the triangle solver algorithm.
        """

        extent = grid.geometry.extent

        initial_triangles = self.ArrayTriangles.for_limits_and_scale(
            y_min=extent[2],
            y_max=extent[3],
            x_min=extent[0],
            x_max=extent[1],
            scale=grid.pixel_scale,
        )

        for number in range(self.n_steps_from(pixel_scale=grid.pixel_scale)):
            kept_triangles = self._filter_triangles(
                lensing_obj=lensing_obj,
                triangles=initial_triangles,
                source_plane_coordinate=source_plane_coordinate,
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
