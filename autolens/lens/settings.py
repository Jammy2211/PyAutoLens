from autoconf import conf
from autolens import exc
from autolens.fit import fit_positions

import copy


class SettingsLens:
    def __init__(
        self,
        auto_positions_factor=None,
        auto_positions_minimum_threshold=None,
        positions_threshold=None,
    ):

        self.auto_positions_factor = auto_positions_factor
        self.auto_positions_minimum_threshold = auto_positions_minimum_threshold
        self.positions_threshold = positions_threshold

    @property
    def tag(self):
        return self.auto_positions_factor_tag + self.positions_threshold_tag

    @property
    def auto_positions_factor_tag(self):
        """Generate an auto positions factor tag, to customize phase names based on the factor automated positions are
        required to trace within one another.

        This changes the phase name 'phase_name' as follows:

        auto_positions_factor = None -> phase_name
        auto_positions_factor = 2.0 -> phase_name__auto_pos_x2.00
        auto_positions_factor = 3.0 -> phase_name__auto_pos_x3.00
        """
        if self.auto_positions_factor is None:
            return ""

        if self.auto_positions_minimum_threshold is not None:
            auto_positions_minimum_threshold_tag = f"_{conf.instance.tag.get('lens', 'auto_positions_minimum_threshold')}_{str(self.auto_positions_minimum_threshold)}"
        else:
            auto_positions_minimum_threshold_tag = ""

        return (
            "__"
            + conf.instance.tag.get("lens", "auto_positions_factor")
            + "_x{0:.2f}".format(self.auto_positions_factor)
            + auto_positions_minimum_threshold_tag
        )

    @property
    def positions_threshold_tag(self):
        """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
        to trace within one another.

        This changes the phase name 'phase_name' as follows:

        positions_threshold = 1 -> phase_name
        positions_threshold = 2 -> phase_name_positions_threshold_2
        positions_threshold = 2 -> phase_name_positions_threshold_2
        """

        old_tag = conf.instance.general.get("tag", "old_tag", bool)

        if old_tag and self.auto_positions_factor is not None:
            return ""

        if self.positions_threshold is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("lens", "positions_threshold", str)
            + "_{0:.2f}".format(self.positions_threshold)
        )

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
