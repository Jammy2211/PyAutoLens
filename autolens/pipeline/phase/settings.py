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

        return (
            f"{conf.instance['notation']['settings_tags']['phase']['phase']}__"
            f"{self.settings_masked_imaging.tag_no_inversion}__"
            f"{self.settings_lens.tag}"
            f"{self.log_likelihood_cap_tag}"
        )

    @property
    def phase_tag_with_inversion(self):
        return (
            f"{conf.instance['notation']['settings_tags']['phase']['phase']}__"
            f"{self.settings_masked_imaging.tag_with_inversion}__"
            f"{self.settings_lens.tag}__"
            f"{self.settings_pixelization.tag}__"
            f"{self.settings_inversion.tag}"
            f"{self.log_likelihood_cap_tag}"
        )


class SettingsPhaseInterferometer(settings.SettingsPhaseInterferometer):
    def __init__(
        self,
        settings_masked_interferometer=interferometer.SettingsMaskedInterferometer(),
        settings_pixelization=pix.SettingsPixelization(use_border=True),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=SettingsLens(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            settings_masked_interferometer=settings_masked_interferometer,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            log_likelihood_cap=log_likelihood_cap,
        )

        self.settings_lens = settings_lens

    @property
    def phase_tag_no_inversion(self):
        return (
            f"{conf.instance['notation']['settings_tags']['phase']['phase']}__"
            f"{self.settings_masked_interferometer.tag_no_inversion}__"
            f"{self.settings_lens.tag}"
            f"{self.log_likelihood_cap_tag}"
        )

    @property
    def phase_tag_with_inversion(self):

        return (
            f"{conf.instance['notation']['settings_tags']['phase']['phase']}__"
            f"{self.settings_masked_interferometer.tag_with_inversion}__"
            f"{self.settings_lens.tag}__"
            f"{self.settings_pixelization.tag}__"
            f"{self.settings_inversion.tag}"
            f"{self.log_likelihood_cap_tag}"
        )


class SettingsPhasePositions:
    def __init__(self, lens):

        self.lens = lens
