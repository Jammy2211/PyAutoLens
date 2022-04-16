import copy

from autoconf import conf

from autolens.point.fit_point.max_separation import FitPositionsSourceMaxSeparation

from autolens import exc


class SettingsLens:
    def __init__(
        self,
        positions_threshold=None,
        stochastic_likelihood_resamples=None,
        stochastic_samples: int = 250,
        stochastic_histogram_bins: int = 10,
    ):

        self.positions_threshold = positions_threshold
        self.stochastic_likelihood_resamples = stochastic_likelihood_resamples
        self.stochastic_samples = stochastic_samples
        self.stochastic_histogram_bins = stochastic_histogram_bins

        self.einstein_radius_estimate = None
        self.einstein_radius_count = None

    def check_positions_trace_within_threshold_via_tracer(self, positions, tracer):

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        if positions is not None and self.positions_threshold is not None:

            positions_fit = FitPositionsSourceMaxSeparation(
                positions=positions, noise_map=None, tracer=tracer
            )

            if not positions_fit.max_separation_within_threshold(
                self.positions_threshold
            ):

                if conf.instance["general"]["test"]["test_mode"]:
                    return

                raise exc.RayTracingException

    def modify_positions_threshold(self, positions_threshold):

        settings = copy.copy(self)
        settings.positions_threshold = positions_threshold
        return settings

    def modify_einstein_radius_estimate(self, einstein_radius_estimate):

        settings = copy.copy(self)
        settings.einstein_radius_estimate = einstein_radius_estimate
        settings.einstein_radius_count = 0
        return settings
