import itertools
import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import autofit as af
from autoarray import Grid2D
from autolens.point.triangles.triangle_solver import TriangleSolver
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


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


class AnalysisBestMatch(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Predict the log likelihood of the predicted coordinates by comparing each observed
        multiple image to whichever predicted multiple image is closest, allowing for repeats.

        Parameters
        ----------
        predicted_coordinates
            The predicted multiple image coordinates of the point source.

        Returns
        -------
        The log likelihood of the predicted coordinates.
        """
        log_likelihood = math.log(1 / 2 * math.pi * self.error**2)
        for observed in self.observed_coordinates:
            distances = [
                self.square_distance(predicted, observed)
                for predicted in predicted_coordinates
            ]
            minimum_distance = min(distances)
            log_likelihood -= minimum_distance / (2 * self.error**2)
        return 0.5 * log_likelihood


class AnalysisBestNoRepeat(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Predict the log likelihood of the predicted coordinates by comparing each observed
        multiple image to the closest predicted multiple image, without allowing for repeats.

        That is, each predicted multiple image is used only once. The Hungarian algorithm
        is used to find the best matching of predicted to observed coordinates.

        Parameters
        ----------
        predicted_coordinates
            The predicted multiple image coordinates of the point source.

        Returns
        -------
        The log likelihood of the predicted coordinates.
        """
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



class AnalysisMarginalizeOverAll(AnalysisPointSource):
    def _log_likelihood_for_coordinates(
        self, predicted_coordinates: List[Tuple[float, float]]
    ) -> float:
        """
        Predict the log likelihood of the predicted coordinates by comparing each observed
        multiple image to all predicted multiple images. Effectively, this marginalizes over
        all possible pairings of observed and predicted coordinates.

        Parameters
        ----------
        predicted_coordinates
            The predicted multiple image coordinates of the point source.

        Returns
        -------
        The log likelihood of the predicted coordinates.
        """
        combinations = len(predicted_coordinates) ** len(self.observed_coordinates)
        log_likelihood = -math.log(combinations)

        for observed in self.observed_coordinates:
            log_likelihood -= logsumexp(
                [
                    self.error_corrected_distance(predicted, observed)
                    for predicted in predicted_coordinates
                ]
            )

        return log_likelihood
