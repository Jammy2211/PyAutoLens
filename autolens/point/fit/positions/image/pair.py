import numpy as np
from scipy.optimize import linear_sum_assignment

import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePair(AbstractFitPositionsImagePair):
    """
    A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
    their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

    Parameters
    ----------
    data : Grid2DIrregular
        The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
    noise_value
        The noise-value assumed when computing the log likelihood.
    """

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        residual_map = []

        cost_matrix = np.linalg.norm(
            np.array(
                self.data,
            )[:, np.newaxis]
            - np.array(
                self.model_data,
            ),
            axis=2,
        )

        data_indexes, model_indexes = linear_sum_assignment(cost_matrix)

        for data_index, model_index in zip(data_indexes, model_indexes):
            distance = np.sqrt(
                self.square_distance(
                    self.data[data_index], self.model_data[model_index]
                )
            )

            residual_map.append(distance)

        return aa.ArrayIrregular(values=residual_map)
