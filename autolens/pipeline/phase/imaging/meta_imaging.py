from autolens.dataset import imaging
from autogalaxy.pipeline.phase.imaging import meta_imaging as ag_meta_imaging
from autolens.pipeline.phase.dataset import meta_dataset


class MetaImaging(ag_meta_imaging.MetaImaging, meta_dataset.MetaLens):
    def __init__(
        self,
        model,
        sub_size=2,
        is_hyper_phase=False,
        signal_to_noise_limit=None,
        auto_positions_factor=None,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
        bin_up_factor=None,
    ):

        super().__init__(
            model=model,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            signal_to_noise_limit=signal_to_noise_limit,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        meta_dataset.MetaLens.__init__(
            self=self,
            auto_positions_factor=auto_positions_factor,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
        )

        self.psf_shape_2d = psf_shape_2d
        self.bin_up_factor = bin_up_factor

    def masked_dataset_from(self, dataset, mask, results):

        mask = self.mask_with_phase_sub_size_from_mask(mask=mask)

        positions = self.updated_positions_from_positions_and_results(
            positions=dataset.positions, results=results
        )

        self.positions_threshold = self.updated_positions_threshold_from_positions(
            positions=positions, results=results
        )

        self.check_positions(positions=positions)

        dataset.positions = positions

        preload_sparse_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        if self.bin_up_factor is not None:

            dataset = dataset.binned_from_bin_up_factor(
                bin_up_factor=self.bin_up_factor
            )

            mask = mask.mapping.binned_mask_from_bin_up_factor(
                bin_up_factor=self.bin_up_factor
            )

        if self.signal_to_noise_limit is not None:
            dataset = dataset.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        masked_imaging = imaging.MaskedImaging(
            imaging=dataset,
            mask=mask,
            psf_shape_2d=self.psf_shape_2d,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return masked_imaging
