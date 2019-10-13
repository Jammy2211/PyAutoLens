from autolens.lens import lens_data as ld
from autolens.pipeline.phase import data


class MetaImagingFit(data.MetaDataFit):
    def __init__(
            self,
            variable,
            sub_size=2,
            is_hyper_phase=False,
            signal_to_noise_limit=None,
            positions_threshold=None,
            mask_function=None,
            inner_mask_radii=None,
            pixel_scale_interpolation_grid=None,
            pixel_scale_binned_cluster_grid=None,
            inversion_uses_border=True,
            inversion_pixel_limit=None,
            psf_shape=None,
            bin_up_factor=None
    ):
        super().__init__(
            variable=variable,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            signal_to_noise_limit=signal_to_noise_limit,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )
        self.psf_shape = psf_shape
        self.bin_up_factor = bin_up_factor

    def data_fit_from(
            self,
            data,
            mask,
            positions,
            results,
            modified_image
    ):
        mask = self.setup_phase_mask(
            data=data,
            mask=mask
        )

        self.check_positions(positions=positions)

        if self.uses_cluster_inversion:
            pixel_scale_binned_grid = self.pixel_scale_binned_grid_from_mask(mask=mask)
        else:
            pixel_scale_binned_grid = None

        preload_pixelization_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        lens_imaging_data = ld.LensImagingData(
            imaging_data=data.modified_image_data_from_image(
                modified_image
            ),
            mask=mask,
            trimmed_psf_shape=self.psf_shape,
            positions=positions,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=pixel_scale_binned_grid,
            hyper_noise_map_max=self.hyper_noise_map_max,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        if self.signal_to_noise_limit is not None:
            lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        if self.bin_up_factor is not None:
            lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_binned_up_imaging_data_and_mask(
                bin_up_factor=self.bin_up_factor
            )

        return lens_imaging_data