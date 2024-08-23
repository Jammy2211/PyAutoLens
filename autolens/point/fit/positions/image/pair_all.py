import numpy as np

import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


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
    def noise_map(self):
        noise_map = []

        for i in range(len(self.data)):
            for j in range(len(self.model_data)):
                noise_map.append(self._noise_map[i])

        return aa.ArrayIrregular(values=noise_map)

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        combinations = len(self.model_data) ** len(self.data)

        residual_map = []

        for model_data in self.model_data:
            for data in self.data:
                distance = np.sqrt(self.square_distance(data, model_data))

                residual_map.append(distance)

        return aa.ArrayIrregular(values=residual_map)
