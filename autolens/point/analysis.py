from typing import List

import autofit as af
from autoarray import Grid2D
from autogalaxy.profiles.point_sources import Point
from autolens.point.triangles.triangle_solver import TriangleSolver


class Analysis(af.Analysis):
    def __init__(
        self,
        points: List[Point],
        error: float,
        grid: Grid2D,
        pixel_scale_precision=0.025,
    ):
        self.points = points
        self.error = error

        self.grid = grid
        self.pixel_scale_precision = pixel_scale_precision

    def log_likelihood_function(self, instance):
        lens = instance.lens

        solver = TriangleSolver(
            lensing_obj=lens,
            grid=self.grid,
            pixel_scale_precision=self.pixel_scale_precision,
        )
        source_plane_coordinates = instance.source.centre
        multiple_image_coordinates = solver.solve(
            source_plane_coordinate=source_plane_coordinates
        )
        return multiple_image_coordinates
