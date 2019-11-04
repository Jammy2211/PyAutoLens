from autolens.masked import masked_dataset
from autolens.pipeline.phase import dataset


class MetaInterferometerFit(dataset.MetaDatasetFit):
    def __init__(
        self,
        variable,
        sub_size=2,
        is_hyper_phase=False,
        positions_threshold=None,
        mask_function=None,
        inner_mask_radii=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
        primary_beam_shape_2d=None,
        bin_up_factor=None,
    ):
        super().__init__(
            variable=variable,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )
        self.primary_beam_shape_2d = primary_beam_shape_2d
        self.bin_up_factor = bin_up_factor

    def data_fit_from(self, data, mask, positions, results, modified_image):
        mask = self.setup_phase_mask(data=data, mask=mask)

        self.check_positions(positions=positions)

        preload_sparse_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        masked_interferometer = masked_dataset.MaskedInterferometer(
            interferometer=data.modified_image_from_image(modified_image),
            mask=mask,
            psf_shape_2d=self.primary_beam_shape_2d,
            positions=positions,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return masked_interferometer
