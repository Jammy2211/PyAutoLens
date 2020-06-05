from autogalaxy.pipeline.phase.imaging import meta_imaging as ag_meta_imaging
from autolens.dataset import imaging
from autolens.pipeline.phase.dataset import meta_dataset


class MetaImaging(ag_meta_imaging.MetaImaging, meta_dataset.MetaLens):
    def __init__(self, settings, model, is_hyper_phase=False):

        super().__init__(settings=settings, model=model, is_hyper_phase=is_hyper_phase)

        meta_dataset.MetaLens.__init__(
            self=self, settings=settings, is_hyper_phase=is_hyper_phase
        )

    def masked_dataset_from(self, dataset, mask, results):

        mask = self.mask_with_phase_sub_size_from_mask(mask=mask)

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

        if self.settings.bin_up_factor is not None:

            dataset = dataset.binned_from_bin_up_factor(
                bin_up_factor=self.settings.bin_up_factor
            )

            mask = mask.binned_mask_from_bin_up_factor(
                bin_up_factor=self.settings.bin_up_factor
            )

        if self.settings.signal_to_noise_limit is not None:
            dataset = dataset.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.settings.signal_to_noise_limit
            )

        masked_imaging = imaging.MaskedImaging(
            imaging=dataset,
            mask=mask,
            grid_class=self.settings.grid_class,
            grid_inversion_class=self.settings.grid_inversion_class,
            fractional_accuracy=self.settings.fractional_accuracy,
            sub_steps=self.settings.sub_steps,
            psf_shape_2d=self.settings.psf_shape_2d,
            pixel_scales_interp=self.settings.pixel_scales_interp,
            inversion_pixel_limit=self.settings.inversion_pixel_limit,
            positions_threshold=self.settings.positions_threshold,
            inversion_uses_border=self.settings.inversion_uses_border,
            inversion_stochastic=self.settings.inversion_stochastic,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return masked_imaging
