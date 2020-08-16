from autoconf import conf
from autoarray.structures import grids
from autoarray.operators import transformer
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.pipeline.phase import settings


class PhaseSettingsLens:
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
            auto_positions_minimum_threshold_tag = f"_{conf.instance.tag.get('phase', 'auto_positions_minimum_threshold')}_{str(self.auto_positions_minimum_threshold)}"
        else:
            auto_positions_minimum_threshold_tag = ""

        return (
            "__"
            + conf.instance.tag.get("phase", "auto_positions_factor")
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
        if self.positions_threshold is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("phase", "positions_threshold", str)
            + "_{0:.2f}".format(self.positions_threshold)
        )


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
        psf_shape_2d=None,
        pixel_scales_interp=None,
        auto_positions_factor=None,
        auto_positions_minimum_threshold=None,
        positions_threshold=None,
        pixelization_settings=pix.PixelizationSettings(),
        inversion_settings=inv.InversionSettings(),
        log_likelihood_cap=None,
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
            psf_shape_2d=psf_shape_2d,
            pixelization_settings=pixelization_settings,
            inversion_settings=inversion_settings,
            log_likelihood_cap=log_likelihood_cap,
        )

        PhaseSettingsLens.__init__(
            self=self,
            auto_positions_factor=auto_positions_factor,
            auto_positions_minimum_threshold=auto_positions_minimum_threshold,
            positions_threshold=positions_threshold,
        )

    @property
    def phase_no_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_no_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
            + self.pixelization.use_border_tag
            + self.pixelization.is_stochastic_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_with_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
            + self.pixelization.use_border_tag
            + self.pixelization.is_stochastic_tag
            + self.log_likelihood_cap_tag
        )


class PhaseSettingsInterferometer(
    settings.PhaseSettingsInterferometer, PhaseSettingsLens
):
    def __init__(
        self,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        transformer_class=transformer.TransformerNUFFT,
        primary_beam_shape_2d=None,
        auto_positions_factor=None,
        auto_positions_minimum_threshold=None,
        positions_threshold=None,
        pixelization_settings=pix.PixelizationSettings(),
        inversion_settings=inv.InversionSettings(),
        log_likelihood_cap=None,
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
            transformer_class=transformer_class,
            primary_beam_shape_2d=primary_beam_shape_2d,
            pixelization_settings=pixelization_settings,
            inversion_settings=inversion_settings,
            log_likelihood_cap=log_likelihood_cap,
        )

        PhaseSettingsLens.__init__(
            self=self,
            auto_positions_factor=auto_positions_factor,
            auto_positions_minimum_threshold=auto_positions_minimum_threshold,
            positions_threshold=positions_threshold,
        )

    @property
    def phase_no_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_no_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
            + self.pixelization.use_border_tag
            + self.pixelization.is_stochastic_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_with_inversion_tag
            + self.transformer_tag
            + self.inversion.use_linear_operators_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.auto_positions_factor_tag
            + self.positions_threshold_tag
            + self.pixelization.use_border_tag
            + self.pixelization.is_stochastic_tag
            + self.log_likelihood_cap_tag
        )


class PhaseSettingsPositions(PhaseSettingsLens):
    def __init__(self, positions_threshold):

        PhaseSettingsLens.__init__(self=self, positions_threshold=positions_threshold)
