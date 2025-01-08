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


class Fit:
    def __init__(self, data, model_positions, noise_map):
        self.data = data
        self.model_positions = model_positions
        self.noise_map = noise_map

    @staticmethod
    def square_distance(coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    def log_p(self, data_position, model_position, sigma):
        chi2 = self.square_distance(data_position, model_position) / sigma**2
        return -np.log(np.sqrt(2 * np.pi * sigma**2)) - 0.5 * chi2

    def log_likelihood(self):
        n_permutations = len(self.model_positions) ** len(self.data)
        return -np.log(n_permutations) + np.sum(
            np.array(
                [
                    np.log(
                        np.sum(
                            [
                                np.exp(
                                    self.log_p(
                                        data_position,
                                        model_position,
                                        sigma,
                                    )
                                )
                                for model_position in self.model_positions
                            ]
                        )
                    )
                    for data_position, sigma in zip(self.data, self.noise_map)
                ]
            )
        )
