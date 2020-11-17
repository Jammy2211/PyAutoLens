from autoconf import conf
from autolens import exc
from autolens.fit import fit_positions

import copy


class SettingsLens:
    def __init__(
        self,
        positions_threshold=None,
        auto_positions_factor=None,
        auto_positions_minimum_threshold=None,
    ):

        self.positions_threshold = positions_threshold
        self.auto_positions_factor = auto_positions_factor
        self.auto_positions_minimum_threshold = auto_positions_minimum_threshold

    @property
    def tag(self):
        return (
            f"{conf.instance['notation']['settings_tags']['lens']['lens']}["
            f"{self.positions_threshold_tag}]"
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

    def check_positions_trace_within_threshold_via_tracer(self, positions, tracer):

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        if positions is not None and self.positions_threshold is not None:

            positions_fit = fit_positions.FitPositionsSourcePlaneMaxSeparation(
                positions=positions, tracer=tracer, noise_value=1.0
            )

            if not positions_fit.maximum_separation_within_threshold(
                self.positions_threshold
            ):
                raise exc.RayTracingException

    def modify_positions_threshold(self, positions_threshold):

        settings = copy.copy(self)
        settings.positions_threshold = positions_threshold
        return settings
