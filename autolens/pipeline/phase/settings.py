from autoarray.structures import grids
from autoarray.operators import transformer
from autogalaxy.pipeline.phase import settings

import copy


class PhaseSettingsLens:
    def __init__(
        self,
        interpolation_pixel_scale=None,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=True,
    ):

        self.interpolation_pixel_scale = interpolation_pixel_scale
        self.auto_positions_factor = auto_positions_factor
        self.positions_threshold = positions_threshold
        self.inversion_uses_border = inversion_uses_border

    def edit(
        self,
        interpolation_pixel_scale=None,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=None,
    ):

        settings = copy.copy(self)

        settings.interpolation_pixel_scale = (
            self.interpolation_pixel_scale
            if interpolation_pixel_scale is None
            else interpolation_pixel_scale
        )

        settings.auto_positions_factor = (
            self.auto_positions_factor
            if auto_positions_factor is None
            else auto_positions_factor
        )

        settings.positions_threshold = (
            self.positions_threshold
            if positions_threshold is None
            else positions_threshold
        )

        settings.inversion_uses_border = (
            self.inversion_uses_border
            if inversion_uses_border is None
            else inversion_uses_border
        )

        return settings

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

    @property
    def interpolation_pixel_scale_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the interpolation \
        grid that deflection angles are computed on before interpolating to the and sub aa.

        This changes the phase name 'phase_name' as follows:

        interpolation_pixel_scale = 1 -> phase_name
        interpolation_pixel_scale = 2 -> phase_name_interpolation_pixel_scale_2
        interpolation_pixel_scale = 2 -> phase_name_interpolation_pixel_scale_2
        """
        if self.interpolation_pixel_scale is None:
            return ""
        return "__interp_{0:.3f}".format(self.interpolation_pixel_scale)


class PhaseSettingsImaging(settings.PhaseSettingsImaging, PhaseSettingsLens):
    def __init__(
        self,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
        interpolation_pixel_scale=None,
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
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
            psf_shape_2d=psf_shape_2d,
        )

        PhaseSettingsLens.__init__(
            self=self,
            interpolation_pixel_scale=interpolation_pixel_scale,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            inversion_uses_border=inversion_uses_border,
        )

    def edit(
        self,
        grid_class=None,
        grid_inversion_class=None,
        sub_size=None,
        fractional_accuracy=None,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
        interpolation_pixel_scale=None,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=None,
    ):

        settings = super().edit(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
            psf_shape_2d=psf_shape_2d,
        )

        settings = PhaseSettingsLens.edit(
            self=settings,
            interpolation_pixel_scale=interpolation_pixel_scale,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            inversion_uses_border=inversion_uses_border,
        )

        return settings

    @property
    def phase_tag(self):
        return (
            "phase_tag"
            + self.sub_size_tag
            + self.interpolation_pixel_scale_tag
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
        grid_class=grids.GridIterator,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        interpolation_pixel_scale=None,
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
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
            transformer_class=transformer_class,
            primary_beam_shape_2d=primary_beam_shape_2d,
        )

        PhaseSettingsLens.__init__(
            self=self,
            interpolation_pixel_scale=interpolation_pixel_scale,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            inversion_uses_border=inversion_uses_border,
        )

    def edit(
        self,
        grid_class=None,
        grid_inversion_class=None,
        sub_size=None,
        fractional_accuracy=None,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        transformer_class=None,
        primary_beam_shape_2d=None,
        interpolation_pixel_scale=None,
        auto_positions_factor=None,
        positions_threshold=None,
        inversion_uses_border=None,
    ):

        settings = super().edit(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
            transformer_class=transformer_class,
            primary_beam_shape_2d=primary_beam_shape_2d,
        )

        settings = PhaseSettingsLens.edit(
            self=settings,
            interpolation_pixel_scale=interpolation_pixel_scale,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            inversion_uses_border=inversion_uses_border,
        )

        return settings

    @property
    def phase_tag(self):

        return (
            "phase_tag"
            + self.transformer_tag
            + self.sub_size_tag
            + self.interpolation_pixel_scale_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
        )
