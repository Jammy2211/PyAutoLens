from autolens.dataset import interferometer
from autogalaxy.pipeline.phase.interferometer import (
    meta_interferometer as ag_meta_interferometer,
)
from autolens.pipeline.phase.dataset import meta_dataset


class MetaInterferometer(
    ag_meta_interferometer.MetaInterferometer, meta_dataset.MetaLens
):
    def __init__(
        self,
        model,
        real_space_mask,
        transformer_class,
        sub_size=2,
        is_hyper_phase=False,
        auto_positions_factor=None,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
        primary_beam_shape_2d=None,
    ):
        super().__init__(
            model=model,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            inversion_pixel_limit=inversion_pixel_limit,
            real_space_mask=real_space_mask,
            transformer_class=transformer_class,
            primary_beam_shape_2d=primary_beam_shape_2d,
        )

        meta_dataset.MetaLens.__init__(
            self=self,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
        )

    def masked_dataset_from(self, dataset, mask, results, modified_visibilities):

        real_space_mask = self.mask_with_phase_sub_size_from_mask(
            mask=self.real_space_mask
        )

        positions = self.updated_positions_from_positions_and_results(
            positions=dataset.positions, results=results
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
            transformer_class=self.transformer_class,
            primary_beam_shape_2d=self.primary_beam_shape_2d,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return masked_interferometer
