from autoconf import conf
from autoarray.structures import grids
from autolens import exc
from autolens.fit import fit_point_source

import copy


class SettingsLens:
    def __init__(
        self,
        positions_threshold=None,
        auto_positions_factor=None,
        auto_positions_minimum_threshold=None,
        auto_einstein_radius_factor: float = None,
        auto_einstein_radius_count: int = 250,
        stochastic_likelihood_resamples=None,
        stochastic_samples: int = 250,
        stochastic_histogram_bins: int = 10,
    ):

        self.positions_threshold = positions_threshold
        self.auto_positions_factor = auto_positions_factor
        self.auto_positions_minimum_threshold = auto_positions_minimum_threshold
        self.auto_einstein_radius_factor = auto_einstein_radius_factor
        self.auto_einstein_radius_count = auto_einstein_radius_count
        self.stochastic_likelihood_resamples = stochastic_likelihood_resamples
        self.stochastic_samples = stochastic_samples
        self.stochastic_histogram_bins = stochastic_histogram_bins

        self.einstein_radius_estimate = None
        self.einstein_radius_count = None

    @property
    def tag(self):
        return (
            f"{conf.instance['notation']['settings_tags']['lens']['lens']}["
            f"{self.positions_threshold_tag}"
            f"{self.stochastic_likelihood_resamples_tag}]"
        )

    @property
    def positions_threshold_tag(self):
        """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
        to trace within one another.

        This changes the phase name 'name' as follows:

        positions_threshold = 1 -> name
        positions_threshold = 2 -> phase_name_positions_threshold_2
        positions_threshold = 2 -> phase_name_positions_threshold_2
        """

        if self.positions_threshold is None:
            return conf.instance["notation"]["settings_tags"]["lens"][
                "no_positions_threshold"
            ]
        return conf.instance["notation"]["settings_tags"]["lens"]["positions_threshold"]

    @property
    def stochastic_likelihood_resamples_tag(self):
        """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
        to trace within one another.

        This changes the phase name 'name' as follows:

        positions_threshold = 1 -> name
        positions_threshold = 2 -> phase_name_positions_threshold_2
        positions_threshold = 2 -> phase_name_positions_threshold_2
        """

        if self.stochastic_likelihood_resamples is None:
            return ""
        return (
            f'__{conf.instance["notation"]["settings_tags"]["lens"]["stochastic_likelihood_resamples"]}_'
            f"{self.stochastic_likelihood_resamples}"
        )

    def check_positions_trace_within_threshold_via_tracer(self, positions, tracer):

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        if positions is not None and self.positions_threshold is not None:

            positions_fit = fit_point_source.FitPositionsSourceMaxSeparation(
                positions=positions, noise_map=None, tracer=tracer
            )

            if not positions_fit.max_separation_within_threshold(
                self.positions_threshold
            ):
                raise exc.RayTracingException

    def check_einstein_radius_with_threshold_via_tracer(self, tracer, grid):

        if self.einstein_radius_estimate is None:
            return

        if self.einstein_radius_count > self.auto_einstein_radius_count:
            return

        try:
            einstein_radius_tracer = tracer.einstein_radius_from_grid(grid=grid)
        except Exception:
            raise exc.RayTracingException

        fractional_value = (
            self.auto_einstein_radius_factor * self.einstein_radius_estimate
        )

        einstein_radius_lower = self.einstein_radius_estimate - fractional_value
        einstein_radius_upper = self.einstein_radius_estimate + fractional_value

        if (einstein_radius_tracer < einstein_radius_lower) or (
            einstein_radius_tracer > einstein_radius_upper
        ):
            raise exc.RayTracingException

        self.einstein_radius_count += 1

    def modify_positions_threshold(self, positions_threshold):

        settings = copy.copy(self)
        settings.positions_threshold = positions_threshold
        return settings

    def modify_einstein_radius_estimate(self, einstein_radius_estimate):

        settings = copy.copy(self)
        settings.einstein_radius_estimate = einstein_radius_estimate
        settings.einstein_radius_count = 0
        return settings
