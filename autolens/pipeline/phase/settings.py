from autoconf import conf
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.dataset import imaging, interferometer
from autogalaxy.pipeline.phase import settings
from autolens.lens.settings import SettingsLens


class SettingsPhaseImaging(settings.SettingsPhaseImaging):
    def __init__(
        self,
        settings_masked_imaging=imaging.SettingsMaskedImaging(),
        settings_pixelization=pix.SettingsPixelization(use_border=True),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=SettingsLens(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            settings_masked_imaging=settings_masked_imaging,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            log_likelihood_cap=log_likelihood_cap,
        )

        self.settings_lens = settings_lens

    @property
    def phase_tag_no_inversion(self):

        use_old_tag = conf.instance.general.get("tag", "old_tag", bool)

        if use_old_tag:

            return (
                conf.instance.tag.get("phase", "phase")
                + self.settings_masked_imaging.grid_tag_no_inversion
                + self.settings_masked_imaging.signal_to_noise_limit_tag
                + self.settings_masked_imaging.bin_up_factor_tag
                + self.settings_masked_imaging.psf_shape_tag
                + self.settings_lens.auto_positions_factor_tag
                + self.settings_lens.positions_threshold_tag
                + self.settings_pixelization.use_border_tag
                + self.settings_pixelization.is_stochastic_tag
                + self.log_likelihood_cap_tag
            )

        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_imaging.tag_no_inversion
            + self.settings_lens.tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_tag_with_inversion(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.settings_masked_imaging.tag_with_inversion
            + self.settings_lens.tag
            + self.settings_pixelization.tag
            + self.settings_inversion.tag
            + self.log_likelihood_cap_tag
        )


class SettingsPhaseInterferometer(settings.SettingsPhaseInterferometer):
    def __init__(
        self,
        masked_interferometer=interferometer.SettingsMaskedInterferometer(),
        settings_pixelization=pix.SettingsPixelization(use_border=True),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=SettingsLens(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            masked_interferometer=masked_interferometer,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            log_likelihood_cap=log_likelihood_cap,
        )

        self.settings_lens = settings_lens

    @property
    def phase_tag_no_inversion(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_interferometer.tag_no_inversion
            + self.settings_lens.tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_tag_with_inversion(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_interferometer.tag_with_inversion
            + self.settings_lens.tag
            + self.settings_pixelization.tag
            + self.settings_inversion.tag
            + self.log_likelihood_cap_tag
        )


class SettingsPhasePositions:
    def __init__(self, lens):

        self.lens = lens
