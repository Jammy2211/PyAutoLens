import itertools
import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import autofit as af
from autoarray import Grid2D
from autolens.point.triangles.triangle_solver import TriangleSolver
import numpy as np
from scipy.optimize import linear_sum_assignment


class AnalysisPointSource(af.Analysis, ABC):
    def __init__(
        self,
        coordinates: List[Tuple[float, float]],
        error: float,
        grid: Grid2D,
        pixel_scale_precision=0.025,
        magnification_threshold=0.1,
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
        self.magnification_threshold = magnification_threshold

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
            magnification_threshold=self.magnification_threshold,
        )
        source_plane_coordinates = instance.source.centre
        predicted_coordinates = solver.solve(
            source_plane_coordinate=source_plane_coordinates
        )
        return self._log_likelihood_for_coordinates(predicted_coordinates)

    @abstractmethod
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Compute the likelihood of the predicted coordinates.
        """

    @staticmethod
    def square_distance(coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    def error_corrected_distance(self, coord1, coord2):
        return self.square_distance(coord1, coord2) / (2 * self.error**2)


class AnalysisAllToAllPointSource(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
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
        if len(predicted_coordinates) == 0:
            raise af.exc.FitException("The number of predicted coordinates is zero.")

        likelihood = 1 / (len(predicted_coordinates) ** len(self.observed_coordinates))
        for observed in self.observed_coordinates:
            likelihood *= sum(
                [
                    math.exp(
                        -self.square_distance(predicted, observed)
                        / (2 * self.error**2)
                    )
                    for predicted in predicted_coordinates
                ]
            )
        return math.log(likelihood)


class AnalysisClosestPointSource(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Compute the likelihood of the predicted coordinates by comparing the positions of
        the observed and predicted coordinates.

        This is done by pairing the closest predicted and observed coordinates.

        Parameters
        ----------
        predicted_coordinates
            The predicted multiple image coordinates of the point source.

        Returns
        -------
        The log likelihood of the predicted coordinates.

        Raises
        ------
        FitException
            If the number of predicted coordinates is not equal to the number of observed coordinates.
        """

        if len(predicted_coordinates) != len(self.observed_coordinates):
            raise af.exc.FitException(
                "The number of predicted coordinates must be equal to the number of observed coordinates."
            )

        predicted_coordinates = set(predicted_coordinates)
        observed_coordinates = set(self.observed_coordinates)

        log_likelihood = 0.0

        while observed_coordinates:
            predicted, observed = min(
                itertools.product(predicted_coordinates, observed_coordinates),
                key=lambda x: self.square_distance(*x),
            )
            log_likelihood -= self.square_distance(predicted, observed) / (
                2 * self.error**2
            )

            predicted_coordinates.remove(predicted)
            observed_coordinates.remove(observed)

        return log_likelihood


class AnalysisBestMatch(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        log_likelihood = math.log(1 / 2 * math.pi * self.error**2)
        for observed in self.observed_coordinates:
            distances = [
                self.square_distance(predicted, observed)
                for predicted in predicted_coordinates
            ]
            minimum_distance = min(distances)
            log_likelihood -= minimum_distance / self.error**2
        return 0.5 * log_likelihood


class AnalysisBestNoRepeat(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        cost_matrix = np.linalg.norm(
            np.array(
                self.observed_coordinates,
            )[:, np.newaxis]
            - np.array(
                predicted_coordinates,
            ),
            axis=2,
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        log_likelihood = math.log(1 / 2 * math.pi * self.error**2)

        for i, j in zip(row_ind, col_ind):
            observed = self.observed_coordinates[i]
            predicted = predicted_coordinates[j]
            log_likelihood -= self.error_corrected_distance(
                observed,
                predicted,
            )

        return 0.5 * log_likelihood
