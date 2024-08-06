import numpy as np
from scipy.optimize import linear_sum_assignment

import autoarray as aa

from autolens.point.fit_point.positions.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePairAll(AbstractFitPositionsImagePair):
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

        combinations = len(self.model_data) ** len(self.data)

        residual_map = []

        for data in self.data:
            for model_data in self.model_data:

                distance = np.sqrt(
                    self.square_distance(
                        data, model_data
                    )
                )

                residual_map.append(distance)

        return aa.ArrayIrregular(values=residual_map)
