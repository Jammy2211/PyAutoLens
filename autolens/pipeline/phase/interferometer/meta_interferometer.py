from autolens.dataset import dataset as d
from autolens.pipeline.phase.dataset import meta_dataset


class MetaInterferometer(meta_dataset.MetaDataset):
    def __init__(
        self,
        model,
        real_space_mask,
        sub_size=2,
        is_hyper_phase=False,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
        primary_beam_shape_2d=None,
        bin_up_factor=None,
    ):
        super().__init__(
            model=model,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )
        self.real_space_mask = real_space_mask
        self.primary_beam_shape_2d = primary_beam_shape_2d
        self.bin_up_factor = bin_up_factor

    def masked_dataset_from(
        self, dataset, mask, positions, results, modified_visibilities
    ):

        real_space_mask = self.mask_with_phase_sub_size_from_mask(
            mask=self.real_space_mask
        )

        self.check_positions(positions=positions)

        preload_sparse_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        masked_interferometer = d.MaskedInterferometer(
            interferometer=dataset.modified_visibilities_from_visibilities(
                modified_visibilities
            ),
            visibilities_mask=mask,
            real_space_mask=real_space_mask,
            primary_beam_shape_2d=self.primary_beam_shape_2d,
            positions=positions,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return masked_interferometer
