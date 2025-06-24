import numpy as np
from scipy.optimize import linear_sum_assignment

import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePair(AbstractFitPositionsImagePair):
    """
    Fits the positions of a point source dataset using a `Tracer` object with an image-plane chi-squared where every
    model position of the point-source is paired with its closest observed position, without allowing for repeated
    pairings of the same observed position to model positions.

    By not allowing for repeated pairings, this can produce behaviour such as a model position not being paired to
    its closest observed position, but instead being paired to a further observed position, if doing so
    means that the overall distances of pairings are reduced.

    THIS FIT CURRENTLY GIVES UNRELIABLE RESULTS, BECAUSE IT GOES TO SOLUTIONS WHERE THE NUMBER OF MODEL POSITIONS
    IS BELOW THE NUMBER OF DATA POSITIONS, REDUCING THE CHI-SQUARED TO LOW VALUES. PYAUTOLENS SHOULD BE UPDATED TO
    PENALIZE THIS BEHAVIOUR BEFORE THIS FIT CAN BE USED. THIS REISDUAL MAP PROPERTY MAY ALSO NEED TO BE EXTENDED
    TO ACCOUNT FOR NOISE.

    The fit performs the following steps:

    1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
       as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

    2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
       source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
       multi-plane ray-tracing.

    3) Pair each model position with the observed position, not allowing for repeated pairings of the same
       observed position to model positions, to compute the `residual_map`. This may result in some observed
       positions not being paired to their closest model position, if doing so reduces the overall distances of
       pairings.

    5) Compute the chi-squared of each position as the square of the residual divided by the RMS noise-map value.

    6) Sum the chi-squared values to compute the overall log likelihood of the fit.

    Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
    point source dataset to ensure that point source datasets are fitted to the correct point source.

    This fit object is used in the `FitPointDataset` to perform position based fitting of a `PointDataset`,
    which may also fit other components of the point dataset like fluxes or time delays.

    When performing a `model-fit`via an `AnalysisPoint` object the `figure_of_merit` of this object
    is called and returned in the `log_likelihood_function`.

    Parameters
    ----------
    name
        The name of the point source dataset which is paired to a `Point` profile.
    data
        The positions of the point source in the image-plane which are fitted.
    noise_map
        The noise-map of the positions which are used to compute the log likelihood of the positions.
    tracer
        The tracer of galaxies whose point source profile are used to fit the positions.
    solver
        Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
        triangles to and from the source-plane.
    profile
        Manually input the profile of the point source, which is used instead of the one extracted from the
        tracer via name pairing if that profile is not found.
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
