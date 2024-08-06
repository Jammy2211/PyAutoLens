import numpy as np

import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePairRepeat(AbstractFitPositionsImagePair):
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

        for position in self.data:
            distances = [
                self.square_distance(model_position, position)
                for model_position in self.model_data
            ]
            residual_map.append(np.sqrt(min(distances)))

        return aa.ArrayIrregular(values=residual_map)
