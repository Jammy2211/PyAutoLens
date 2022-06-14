import numpy as np
from typing import Optional, Union

import autoarray as aa
import autofit as af

from autoconf import conf

from autolens.point.fit_point.max_separation import FitPositionsSourceMaxSeparation

from autolens import exc


class PositionsThresholder:
    def __init__(
        self,
        positions: aa.Grid2DIrregular,
        threshold: float,
        use_resampling: bool = False,
        use_likelihood_penalty: bool = False,
        use_likelihood_overwrite: bool = False,
    ):

        if len(positions) == 1:
            raise exc.PositionsException(
                f"The positions input into the PositionsThresholder have length one "
                f"(e.g. it is only one (y,x) coordinate and therefore cannot be compared with other images).\n\n"
                "Please input more positions into the PositionsThresholder."
            )

        if sum([use_resampling, use_likelihood_penalty, use_likelihood_overwrite]) == 0:
            raise exc.PositionsException(
                f"No `use_` setting has been input as True for the PositionsThresholder."
                f"Please input `use_resampling=True`, `use_likelihood_penalty=True` or `use_likelihood_overwrite=True`"
            )

        if sum([use_resampling, use_likelihood_penalty, use_likelihood_overwrite]) > 1:
            raise exc.PositionsException(
                f"More than one `use_` setting has been input as True for the PositionsThresholder."
                f"Please only input one entry out of `use_resampling=True`, `use_likelihood_penalty=True` or `use_likelihood_overwrite=True`"
            )

        self.positions = positions
        self.threshold = threshold

        self.use_resampling = use_resampling
        self.use_likelihood_penalty = use_likelihood_penalty
        self.use_likelihood_overwrite = use_likelihood_overwrite

    def resample_if_not_within_threshold(self, tracer):

        if not self.use_resampling:
            return None

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        positions_fit = FitPositionsSourceMaxSeparation(
            positions=self.positions, noise_map=None, tracer=tracer
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            if conf.instance["general"]["test"]["test_mode"]:
                return

            raise exc.RayTracingException

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
        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        positions_fit = FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=None, tracer=tracer
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            return 100.0 * (
                positions_fit.max_separation_of_source_plane_positions - self.threshold
            )

    def log_likelihood_function_positions_overwrite(
        self, instance: af.ModelInstance, tracer, fit_func, dataset
    ) -> Optional[float]:

        if not self.use_likelihood_penalty and not self.use_likelihood_overwrite:
            return None

        log_likelihood_positions_penalty = self.log_likelihood_penalty_from(
            tracer=tracer, positions=self.positions
        )

        if self.use_likelihood_penalty:

            return (
                fit_func(instance=instance).figure_of_merit
                + log_likelihood_positions_penalty
            )

        if self.use_likelihood_overwrite:

            log_likelihood_penalty_base = self.log_likelihood_penalty_base_from(
                dataset=dataset
            )

            return log_likelihood_penalty_base + log_likelihood_positions_penalty
