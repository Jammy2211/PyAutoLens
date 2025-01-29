import numpy as np

import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePairAll(AbstractFitPositionsImagePair):
    """
    Fits the positions of a a point source dataset using a `Tracer` object with an image-plane chi-squared where every
    model position of the point-source is paired with all other observed positions using the probability of each
    model posiition explaining each observed position.

    Pairing all model positions with all observed positions is a less intuitive and commonly used approach
    than other methods, for example pairing each position one-to-one. The scheme was proposed in the paper
    below and provides a number of benefits, for example being a fully Bayesian approach to the problem and
    linearizing aspects of the problem.

    https://arxiv.org/abs/2406.15280

    THIS IMPLEMENTATION DOES NOT CURRRENTLY BREAK DOWN THE CALCULATION INTO A RESIDUAL MAP AND CHI-SQUARED,
    GOING STRAIGHT TO A `log_likelihood`. FUTURE WORK WILL WORK OUT HOW TO EXPRESS THIS IN TERMS OF A CHI-SQUARED
    AND RESIDUAL MAP.

    The fit performs the following steps:

    1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
       as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

    2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
       source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
       multi-plane ray-tracing.

    3) Pair every model position with every observed position and return the overall log likelihood of the fit.

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

    def log_p(
        self,
        data_position: np.array,
        model_position: np.array,
        sigma: float,
    ) -> float:
        """
        Compute the log probability of a given model coordinate explaining a given observed coordinate.

        Accounts for noise, with noiser image coordinates having a comparatively lower log probability.

        Parameters
        ----------
        data_position
            The observed coordinate.
        model_position
            The model coordinate.
        sigma
            The noise associated with the observed coordinate.

        Returns
        -------
            The log probability of the model coordinate explaining the observed coordinate.
        """
        chi2 = self.square_distance(data_position, model_position) / sigma**2
        return -np.log(np.sqrt(2 * np.pi * sigma**2)) - 0.5 * chi2

    def all_permutations_log_likelihoods(self) -> np.array:
        """
        Compute the log likelihood for each permutation whereby the model could explain the observed image coordinates.

        For example, if there are two observed image coordinates and two model image coordinates, the log likelihood
        for each permutation is:

        P(data_0 | model_0) * P(data_1 | model_1)
        P(data_0 | model_1) * P(data_1 | model_0)
        P(data_0 | model_0) * P(data_1 | model_0)
        P(data_0 | model_1) * P(data_1 | model_1)

        This is every way in which the coordinates generated by the model can explain the observed coordinates.
        """
        return np.array(
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
                            for model_position in self.model_data
                            if not np.isnan(model_position).any()
                        ]
                    )
                )
                for data_position, sigma in zip(self.data, self.noise_map)
            ]
        )

    @property
    def chi_squared(self) -> float:
        """
        Compute the log likelihood of the model image coordinates explaining the observed image coordinates.

        This is the sum across all permutations of the observed image coordinates of the log probability of each
        model image coordinate explaining the observed image coordinate.

        For example, if there are two observed image coordinates and two model image coordinates, the log likelihood
        is the sum of the log probabilities:

        P(data_0 | model_0) * P(data_1 | model_1)
        + P(data_0 | model_1) * P(data_1 | model_0)
        + P(data_0 | model_0) * P(data_1 | model_0)
        + P(data_0 | model_1) * P(data_1 | model_1)

        This is every way in which the coordinates generated by the model can explain the observed coordinates.
        """
        n_non_nan_model_positions = np.count_nonzero(
            ~np.isnan(
                self.model_data,
            ).any(axis=1)
        )
        n_permutations = n_non_nan_model_positions ** len(self.data)
        return -2.0 * (
            -np.log(n_permutations) + np.sum(self.all_permutations_log_likelihoods())
        )
