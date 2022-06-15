import numpy as np
from typing import Optional, Union

import autoarray as aa
import autofit as af

from autoconf import conf

from autolens.lens.ray_tracing import Tracer
from autolens.point.fit_point.max_separation import FitPositionsSourceMaxSeparation

from autolens import exc


# TODO : max sure `if not tracer.has_mass_profile or len(tracer.planes) == 1:` is used correct for all resamplers.


class AbstractPositions:
    def __init__(self, positions: aa.Grid2DIrregular, threshold: float):

        if len(positions) == 1:
            raise exc.PositionsException(
                f"The positions input into the Positions have length one "
                f"(e.g. it is only one (y,x) coordinate and therefore cannot be compared with other images).\n\n"
                "Please input more positions into the Positions."
            )

        self.positions = positions
        self.threshold = threshold

    def log_likelihood_function_positions_overwrite(
        self, instance: af.ModelInstance, analysis: "AnalysisDataset"
    ) -> Optional[float]:
        raise NotImplementedError


class PositionsResample(AbstractPositions):
    def log_likelihood_function_positions_overwrite(
        self, instance: af.ModelInstance, analysis: "AnalysisDataset"
    ) -> Optional[float]:

        tracer = analysis.tracer_via_instance_from(instance=instance)

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        positions_fit = FitPositionsSourceMaxSeparation(
            positions=self.positions, noise_map=None, tracer=tracer
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            if conf.instance["general"]["test"]["test_mode"]:
                return

            raise exc.RayTracingException


class PositionsLHOverwrite(AbstractPositions):

    def __init__(self, positions: aa.Grid2DIrregular, threshold: float, log_likelihood_penalty_factor : float = 1e8):

        super().__init__(positions=positions, threshold=threshold)

        self.log_likelihood_penalty_factor = log_likelihood_penalty_factor

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

        if isinstance(dataset, aa.Imaging):

            chi_squared_map = aa.util.fit.chi_squared_map_from(
                residual_map=residual_map, noise_map=dataset.noise_map
            )

            chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

            noise_normalization = aa.util.fit.noise_normalization_from(
                noise_map=dataset.noise_map
            )

        else:

            chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
                residual_map=residual_map, noise_map=dataset.noise_map
            )

            chi_squared = aa.util.fit.chi_squared_complex_from(
                chi_squared_map=chi_squared_map
            )

            noise_normalization = aa.util.fit.noise_normalization_complex_from(
                noise_map=dataset.noise_map
            )

        return -0.5 * (chi_squared + noise_normalization)

    def log_likelihood_penalty_from(self, tracer: Tracer) -> Optional[float]:
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
            positions=self.positions, noise_map=None, tracer=tracer
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            return self.log_likelihood_penalty_factor * (
                positions_fit.max_separation_of_source_plane_positions - self.threshold
            )

    def log_likelihood_function_positions_overwrite(
        self, instance: af.ModelInstance, analysis: "AnalysisDataset"
    ) -> Optional[float]:

        tracer = analysis.tracer_via_instance_from(instance=instance)

        log_likelihood_positions_penalty = self.log_likelihood_penalty_from(
            tracer=tracer
        )

        if log_likelihood_positions_penalty is None:
            return None

        log_likelihood_penalty_base = self.log_likelihood_penalty_base_from(
            dataset=analysis.dataset
        )

        return log_likelihood_penalty_base - log_likelihood_positions_penalty
