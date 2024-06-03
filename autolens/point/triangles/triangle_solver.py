import math
from typing import Tuple

from autoarray import Grid2D
from autoarray.structures.triangles.subsample_triangles import SubsampleTriangles
from autoarray.structures.triangles.triangles import Triangles
from autoarray.type import Grid2DLike
from autolens import Tracer


class TriangleSolver:
    def __init__(
        self,
        tracer: Tracer,
        grid: Grid2D,
        target_pixel_scale: float,
    ):
        self.tracer = tracer
        self.grid = grid
        self.target_pixel_scale = target_pixel_scale

    @property
    def n_steps(self):
        return math.ceil(math.log2(self.grid.pixel_scale / self.target_pixel_scale))

    def _source_plane_grid(self, grid: Grid2DLike):
        deflections = self.tracer.deflections_yx_2d_from(grid=grid)
        return grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    def solve(self, source_plane_coordinate: Tuple[float, float]):
        triangles = Triangles.for_grid(grid=self.grid)

        for _ in range(self.n_steps):
            kept_triangles = self._filter_triangles(
                triangles=triangles,
                source_plane_coordinate=source_plane_coordinate,
            )
            triangles = SubsampleTriangles(parent_triangles=kept_triangles)

        return [triangle.mean for triangle in kept_triangles]

    def _filter_triangles(
        self,
        triangles: Triangles,
        source_plane_coordinate: Tuple[float, float],
    ):
        source_plane_grid = self._source_plane_grid(grid=triangles.grid_2d)

        kept_triangles = []
        for image_triangle, source_triangle in zip(
            triangles.triangles,
            triangles.with_updated_grid(source_plane_grid),
        ):
            if source_triangle.contains(point=source_plane_coordinate):
                kept_triangles.append(image_triangle)

        return kept_triangles
