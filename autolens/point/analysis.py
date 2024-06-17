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
        """
        Abstract class for point source analysis.

        Parameters
        ----------
        coordinates
            The observed multiple image coordinates of the point source.
        error
            The error on the position of the observed coordinates.
        grid
            The grid of image plane coordinates.
        pixel_scale_precision
            The target pixel scale of the image grid. That is, how precisely the image plane is sampled.
        """
        self.observed_coordinates = coordinates
        self.error = error

        self.grid = grid
        self.pixel_scale_precision = pixel_scale_precision

    def log_likelihood_function(self, instance):
        """
        Compute the log likelihood of the model instance.

        This is done by solving the position of the multiple images of the point source
        in the image plane to a desired precision before comparing them to the observed coordinates.

        Parameters
        ----------
        instance
            The model instance. Must have a lens and source attribute.

        Returns
        -------
        The log likelihood of the model instance.
        """
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
    def _likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Compute the likelihood of the predicted coordinates.
        """


class AllToAllPointSourceAnalysis(PointSourceAnalysis):
    def _likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Compute the likelihood of the predicted coordinates by comparing the positions of
        the observed and predicted coordinates.

        This is essentially the product over all possible pairings of observed and predicted coordinates.

        Parameters
        ----------
        predicted_coordinates
            The predicted multiple image coordinates of the point source.

        Returns
        -------
        The likelihood of the predicted coordinates.
        """
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
        return math.log(likelihood)
