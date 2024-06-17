import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import autofit as af
from autoarray import Grid2D
from autolens.point.triangles.triangle_solver import TriangleSolver


class PointSourceAnalysis(af.Analysis, ABC):
    def __init__(
        self,
        coordinates: List[Tuple[float, float]],
        error: float,
        grid: Grid2D,
        pixel_scale_precision=0.025,
    ):
        self.observed_coordinates = coordinates
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
        predicted_coordinates = solver.solve(
            source_plane_coordinate=source_plane_coordinates
        )
        return self._likelihood_for_coordinates(predicted_coordinates)

    @abstractmethod
    def _likelihood_for_coordinates(self, predicted_coordinates):
        pass


class AllToAllPointSourceAnalysis(PointSourceAnalysis):
    def _likelihood_for_coordinates(self, predicted_coordinates):
        likelihood = 1 / (len(predicted_coordinates) ** len(self.observed_coordinates))
        for observed in self.observed_coordinates:
            likelihood *= sum(
                [
                    math.exp(
                        (
                            (predicted[0] - observed[0]) ** 2
                            + (predicted[1] - observed[1]) ** 2
                        )
                        / (2 * self.error**2)
                    )
                    for predicted in predicted_coordinates
                ]
            )
        return likelihood
