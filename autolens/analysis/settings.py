import copy
import numpy as np
from typing import Optional, Union

import autoarray as aa
import autogalaxy as ag

from autoconf import conf

from autolens.point.fit_point.max_separation import FitPositionsSourceMaxSeparation

from autolens import exc


class SettingsLens:
    def __init__(
        self,
        threshold: Optional[float] = None,
        use_resampling: bool = False,
        use_likelihood_penalty: bool = False,
        use_likelihood_overwrite: bool = False,
        stochastic_likelihood_resamples: Optional[int] = None,
        stochastic_samples: int = 250,
        stochastic_histogram_bins: int = 10,
    ):

        self.threshold = threshold

        self.use_resampling = use_resampling
        self.use_likelihood_penalty = use_likelihood_penalty
        self.use_likelihood_overwrite = use_likelihood_overwrite

        self.stochastic_likelihood_resamples = stochastic_likelihood_resamples
        self.stochastic_samples = stochastic_samples
        self.stochastic_histogram_bins = stochastic_histogram_bins

    def log_likelihood_penalty_base_from(
        self, dataset: Union[aa.Imaging, aa.Interferometer]
    ) -> float:
        """
        The fast log likelihood penalty scheme returns an alternative penalty log likelihood for any model where the
        image-plane positions do not trace within a threshold distance of one another in the source-plane.

        This `log_likelihood_penalty` is defined as:

        log_Likelihood_penalty_base - log_likelihood_penalty_factor * (max_source_plane_separation - threshold)

        The `log_likelihood_penalty` is only used if `max_source_plane_separation > threshold`.

        This function returns the `log_likelihood_penalty_base`, which represents the lowest possible likelihood
        solutions a model-fit can give. It is the chi-squared of model-data consisting of all zeros plus
        the noise normalziation term.

        Parameters
        ----------
        dataset
            The imaging or interferometer dataset from which the penalty base is computed.
        """
        residual_map = aa.util.fit.residual_map_from(
            data=dataset.data, model_data=np.zeros(dataset.data.shape)
        )
        chi_suqared_map = aa.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=dataset.noise_map
        )
        chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_suqared_map)

        noise_normalization = aa.util.fit.noise_normalization_from(
            noise_map=dataset.noise_map
        )

        return -0.5 * (chi_squared + noise_normalization)

    def log_likelihood_penalty_from(self, positions, tracer):
        """
        The fast log likelihood penalty scheme returns an alternative penalty log likelihood for any model where the
        image-plane positions to not trace within a threshold distance of one another in the source-plane.

        This `log_likelihood_penalty` is defined as:

        log_Likelihood_penalty_base - log_likelihood_penalty_factor * (max_source_plane_separation - threshold)

        The `log_likelihood_penalty` is only used if `max_source_plane_separation > threshold`.

        Parameters
        ----------
        dataset
            The imaging or interferometer dataset from which the penalty base is computed.
        """
        if not tracer.has(cls=ag.mp.MassProfile) or len(tracer.planes) == 1:
            return

        positions_fit = FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=None, tracer=tracer
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            return 100.0 * (
                positions_fit.max_separation_of_source_plane_positions - self.threshold
            )

    def resample_if_not_within_threshold(self, positions, tracer):

        if not tracer.has(cls=ag.mp.MassProfile) or len(tracer.planes) == 1:
            return

        positions_fit = FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=None, tracer=tracer
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            if conf.instance["general"]["test"]["test_mode"]:
                return

            raise exc.RayTracingException
