from autolens.masked import masked_dataset
from autolens.pipeline.phase import dataset


class MetaImagingFit(dataset.MetaDatasetFit):
    def __init__(
        self,
        model,
        sub_size=2,
        is_hyper_phase=False,
        signal_to_noise_limit=None,
        positions_threshold=None,
        mask_function=None,
        inner_mask_radii=None,
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
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )
        self.psf_shape_2d = psf_shape_2d
        self.bin_up_factor = bin_up_factor

    def masked_dataset_from(self, dataset, mask, positions, results, modified_image):
        mask = self.setup_phase_mask(
            shape_2d=dataset.shape_2d, pixel_scales=dataset.pixel_scales, mask=mask
        )

        self.check_positions(positions=positions)

        preload_sparse_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        masked_imaging = masked_dataset.MaskedImaging(
            imaging=dataset.modified_image_from_image(modified_image),
            mask=mask,
            psf_shape_2d=self.psf_shape_2d,
            positions=positions,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        if self.signal_to_noise_limit is not None:
            masked_imaging = masked_imaging.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        if self.bin_up_factor is not None:
            masked_imaging = masked_imaging.binned_from_bin_up_factor(
                bin_up_factor=self.bin_up_factor
            )

        return masked_imaging
