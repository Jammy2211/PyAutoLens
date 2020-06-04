from autoarray.structures import grids
from autoarray.operators import transformer
from autogalaxy.pipeline.phase import settings

import copy


class PhaseSettingsLens:
    def __init__(
        self,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=True,
    ):

        self.auto_positions_factor = auto_positions_factor
        self.positions_threshold = positions_threshold
        self.inversion_uses_border = inversion_uses_border

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
        return "__auto_pos_x{0:.2f}".format(self.auto_positions_factor)

    @property
    def positions_threshold_tag(self):
        """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
        to trace within one another.

        This changes the phase name 'phase_name' as follows:

        positions_threshold = 1 -> phase_name
        positions_threshold = 2 -> phase_name_positions_threshold_2
        positions_threshold = 2 -> phase_name_positions_threshold_2
        """
        if self.positions_threshold is None:
            return ""
        return "__pos_{0:.2f}".format(self.positions_threshold)


class PhaseSettingsImaging(settings.PhaseSettingsImaging, PhaseSettingsLens):
    def __init__(
        self,
        grid_class=grids.GridIterate,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
        pixel_scales_interp=None,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=True,
    ):

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
            psf_shape_2d=psf_shape_2d,
        )

        PhaseSettingsLens.__init__(
            self=self,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            inversion_uses_border=inversion_uses_border,
        )

    @property
    def phase_no_inversion_tag(self):
        return (
            "settings"
            + self.grid_no_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            "settings"
            + self.grid_with_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
        )


class PhaseSettingsInterferometer(
    settings.PhaseSettingsInterferometer, PhaseSettingsLens
):
    def __init__(
        self,
        grid_class=grids.GridIterate,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        transformer_class=transformer.TransformerNUFFT,
        primary_beam_shape_2d=None,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=True,
    ):

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
            transformer_class=transformer_class,
            primary_beam_shape_2d=primary_beam_shape_2d,
        )

        PhaseSettingsLens.__init__(
            self=self,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            inversion_uses_border=inversion_uses_border,
        )

    @property
    def phase_no_inversion_tag(self):
        return (
            "settings"
            + self.grid_no_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            "settings"
            + self.grid_with_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
        )
