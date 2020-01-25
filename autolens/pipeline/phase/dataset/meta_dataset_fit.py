import autofit as af
import autoarray as aa
from autolens import exc
from autoarray.operators.inversion import pixelizations as pix
from autolens.pipeline.phase.dataset.phase import isinstance_or_prior


class MetaDatasetFit:
    def __init__(
        self,
        model,
        sub_size=2,
        signal_to_noise_limit=None,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
        is_hyper_phase=False,
    ):
        self.is_hyper_phase = is_hyper_phase
        self.model = model
        self.sub_size = sub_size
        self.signal_to_noise_limit = signal_to_noise_limit
        self.positions_threshold = positions_threshold
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.inversion_uses_border = inversion_uses_border
        self.inversion_pixel_limit = (
            inversion_pixel_limit
            or af.conf.instance.general.get(
                "inversion", "inversion_pixel_limit_overall", int
            )
        )

    def mask_with_phase_sub_size_from_mask(self, mask):

        if mask.sub_size != self.sub_size:
            mask = aa.mask.manual(
                mask_2d=mask,
                pixel_scales=mask.pixel_scales,
                sub_size=self.sub_size,
                origin=mask.origin,
            )

        return mask

    def check_positions(self, positions):

        if self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

    @property
    def pixelization(self):
        for galaxy in self.model.galaxies:
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.PriorModel):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization

    @property
    def has_pixelization(self):
        if self.pixelization is not None:
            return True
        else:
            return False

    @property
    def uses_cluster_inversion(self):
        if self.model.galaxies:
            for galaxy in self.model.galaxies:
                if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                    return True
        return False

    def preload_pixelization_grids_of_planes_from_results(self, results):

        if self.is_hyper_phase:
            return None

        if (
            results is not None
            and results.last is not None
            and hasattr(results.last, "hyper_combined")
            and self.pixelization is not None
        ):
            if self.pixelization.__class__ is results.last.pixelization.__class__:
                return (
                    results.last.hyper_combined.most_likely_pixelization_grids_of_planes
                )
        return None
