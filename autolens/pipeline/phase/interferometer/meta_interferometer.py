from autogalaxy.pipeline.phase.interferometer import (
    meta_interferometer as ag_meta_interferometer,
)
from autolens.dataset import interferometer
from autolens.pipeline.phase.dataset import meta_dataset


class MetaInterferometer(
    ag_meta_interferometer.MetaInterferometer, meta_dataset.MetaLens
):
    def __init__(self, settings, model, real_space_mask, is_hyper_phase=False):
        super().__init__(
            settings=settings,
            model=model,
            real_space_mask=real_space_mask,
            is_hyper_phase=is_hyper_phase,
        )

        meta_dataset.MetaLens.__init__(
            self=self, settings=settings, is_hyper_phase=is_hyper_phase
        )

    def masked_dataset_from(self, dataset, mask, results, modified_visibilities):

        real_space_mask = self.mask_with_phase_sub_size_from_mask(
            mask=self.real_space_mask
        )

        positions = self.updated_positions_from_positions_and_results(
            positions=dataset.positions, results=results
        )

        self.settings.positions_threshold = self.updated_positions_threshold_from_positions(
            positions=positions, results=results
        )

        self.check_positions(positions=positions)

        dataset.positions = positions

        preload_sparse_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        masked_interferometer = interferometer.MaskedInterferometer(
            interferometer=dataset.modified_visibilities_from_visibilities(
                modified_visibilities
            ),
            visibilities_mask=mask,
            real_space_mask=real_space_mask,
            grid_class=self.settings.grid_class,
            grid_inversion_class=self.settings.grid_inversion_class,
            fractional_accuracy=self.settings.fractional_accuracy,
            sub_steps=self.settings.sub_steps,
            transformer_class=self.settings.transformer_class,
            primary_beam_shape_2d=self.settings.primary_beam_shape_2d,
            pixel_scales_interp=self.settings.pixel_scales_interp,
            inversion_pixel_limit=self.settings.inversion_pixel_limit,
            positions_threshold=self.settings.positions_threshold,
            inversion_uses_border=self.settings.inversion_uses_border,
            inversion_stochastic=self.settings.inversion_stochastic,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return masked_interferometer
