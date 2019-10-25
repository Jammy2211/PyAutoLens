import autoarray as aa
from autoarray.operators import convolution, fourier_transform
import autolens as al
import numpy as np
import pytest

class TestMaskedImaging(object):

    def test__masked_data_via_autoarray(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = aa.masked_imaging.manual(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
        )

        assert (
                masked_imaging_7x7.image.in_1d
                == np.ones(9, )
        ).all()

        assert (
                masked_imaging_7x7.image.in_2d
                == np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (
                masked_imaging_7x7.noise_map.in_1d
                == 2.0 * np.ones(9, )
        ).all()
        assert (
                masked_imaging_7x7.noise_map.in_2d
                == 2.0 * np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.psf.in_1d == np.ones(9, )).all()
        assert (masked_imaging_7x7.psf.in_2d == np.ones((3, 3))).all()
        assert masked_imaging_7x7.trimmed_psf_shape_2d == (3, 3)

        assert type(masked_imaging_7x7.convolver) == convolution.Convolver

    def test__inheritance_from_autoarray(
        self, imaging_7x7, sub_mask_7x7, blurring_grid_7x7
    ):

        masked_imaging_7x7 = al.MaskedImaging(imaging=imaging_7x7,
            mask=sub_mask_7x7, pixel_scale_interpolation_grid=1.0, trimmed_psf_shape_2d=(3,3),
            inversion_pixel_limit=20.0, inversion_uses_border=False, preload_pixelization_grids_of_planes=1
        )

        assert masked_imaging_7x7.inversion_pixel_limit == 20.0
        assert masked_imaging_7x7.inversion_uses_border == False
        assert masked_imaging_7x7.preload_pixelization_grids_of_planes == 1

        grid = aa.masked_grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        assert (masked_imaging_7x7.grid == new_grid).all()
        assert (
            masked_imaging_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx
        ).all()
        assert (
            masked_imaging_7x7.grid.interpolator.wts == new_grid.interpolator.wts
        ).all()

        blurring_grid = grid.blurring_grid_from_kernel_shape(
            kernel_shape=(3, 3)
        )
        new_blurring_grid = blurring_grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (masked_imaging_7x7.blurring_grid.in_1d == blurring_grid_7x7).all()
        assert (masked_imaging_7x7.blurring_grid == new_blurring_grid).all()
        assert (
            masked_imaging_7x7.blurring_grid.interpolator.vtx
            == new_blurring_grid.interpolator.vtx
        ).all()
        assert (
            masked_imaging_7x7.blurring_grid.interpolator.wts
            == new_blurring_grid.interpolator.wts
        ).all()

    def test__masked_imaging_6x6_with_binned_up_imaging(
        self, imaging_6x6, mask_6x6
    ):
        
        masked_imaging_6x6 = al.MaskedImaging(imaging=imaging_6x6, mask=mask_6x6)

        binned_mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        masked_imaging_3x3 = masked_imaging_6x6.binned_from_bin_up_factor(
            bin_up_factor=2
        )

        assert (
                masked_imaging_3x3.image.in_2d
                == np.ones((3, 3)) * np.invert(binned_mask)
        ).all()
        assert (masked_imaging_3x3.psf.in_2d == np.ones((3,3))).all()
        assert (
                masked_imaging_3x3.noise_map.in_2d
                == np.ones((3, 3)) * np.invert(binned_mask)
        ).all()

        assert (
                masked_imaging_3x3.mask
                == binned_mask
        ).all()

        assert (masked_imaging_3x3.image.in_1d == np.ones((1))).all()
        assert (masked_imaging_3x3.noise_map.in_1d == np.ones((1))).all()

    def test__masked_imaging_7x7_with_signal_to_noise_limit(
        self, imaging_7x7, sub_mask_7x7,
    ):

        masked_imaging_7x7 = al.MaskedImaging(imaging=imaging_7x7,
            mask=sub_mask_7x7,
        )

        masked_imaging_snr_limit = masked_imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.25
        )

        assert (masked_imaging_snr_limit.psf.in_2d == np.ones((3, 3))).all()
        assert masked_imaging_snr_limit.trimmed_psf_shape_2d == (3, 3)

        assert (masked_imaging_snr_limit.image.in_1d == np.ones(9)).all()
        assert (masked_imaging_snr_limit.noise_map.in_1d == 4.0 * np.ones(9)).all()

        assert (
            masked_imaging_snr_limit.noise_map.in_2d
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__methods_for_new_data_pass_lensing_only_attributes(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = al.MaskedImaging(
            imaging=imaging_7x7, mask=sub_mask_7x7, positions=[1], positions_threshold=2, preload_pixelization_grids_of_planes=3)

        masked_imaging_new = masked_imaging_7x7.binned_from_bin_up_factor(bin_up_factor=2,
        )

        assert masked_imaging_new.positions == [1]
        assert masked_imaging_new.positions_threshold == 2
        assert masked_imaging_new.preload_pixelization_grids_of_planes == 3

        masked_imaging_new = masked_imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.25
        )

        assert masked_imaging_new.positions == [1]
        assert masked_imaging_new.positions_threshold == 2
        assert masked_imaging_new.preload_pixelization_grids_of_planes == 3


class TestMaskedInterferometer(object):

    def test__masked_data_via_autoarray(self, interferometer_7, sub_mask_7x7):

        masked_interferometer_7 = al.MaskedInterferometer(
            interferometer=interferometer_7,
            real_space_mask=sub_mask_7x7,
        )

        assert (
                masked_interferometer_7.visibilities == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer_7.visibilities == np.ones((7, 2))).all()

        assert (masked_interferometer_7.noise_map == 2.0 * np.ones((7, 2))).all()

        assert (
                masked_interferometer_7.visibilities_mask
                == np.full(fill_value=False, shape=(7, 2))
        ).all()

        assert (masked_interferometer_7.primary_beam.in_2d == np.ones((3, 3))).all()
        assert masked_interferometer_7.trimmed_primary_beam_shape_2d == (3, 3)

        assert (
                masked_interferometer_7.interferometer.uv_wavelengths
                == interferometer_7.uv_wavelengths
        ).all()
        assert masked_interferometer_7.interferometer.uv_wavelengths[0, 0] == -55636.4609375

        assert type(masked_interferometer_7.transformer) == fourier_transform.Transformer

    def test__inheritance_via_autoarray(self, interferometer_7, sub_mask_7x7, grid_7x7, sub_grid_7x7):

        masked_interferometer_7 = al.MaskedInterferometer(
            interferometer=interferometer_7,
            real_space_mask=sub_mask_7x7, pixel_scale_interpolation_grid=1.0, trimmed_primary_beam_shape_2d=(3, 3),
            inversion_pixel_limit=20.0, inversion_uses_border=False, preload_pixelization_grids_of_planes=1
        )

        assert (masked_interferometer_7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_interferometer_7.grid == sub_grid_7x7).all()

        assert masked_interferometer_7.inversion_pixel_limit == 20.0
        assert masked_interferometer_7.inversion_uses_border == False
        assert masked_interferometer_7.preload_pixelization_grids_of_planes == 1

        grid = aa.masked_grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        assert (masked_interferometer_7.grid == new_grid).all()
        assert (
            masked_interferometer_7.grid.interpolator.vtx == new_grid.interpolator.vtx
        ).all()
        assert (
            masked_interferometer_7.grid.interpolator.wts == new_grid.interpolator.wts
        ).all()

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self
    ):
        interferometer = aa.interferometer.manual(
            real_space_shape_2d=(2, 2),
            visibilities=np.ones((19, 2)),
            real_space_pixel_scales=3.0,
            primary_beam=aa.kernel.ones(shape_2d=(7, 7), pixel_scales=1.0),
            noise_map=2.0 * np.ones((19,)),
            uv_wavelengths=3.0 * np.ones((19, 2)),
        )
        mask = aa.mask.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        mask[9, 9] = False

        masked_interferometer = al.MaskedInterferometer(
            interferometer=interferometer,
            real_space_mask=mask,
            trimmed_primary_beam_shape_2d=(5, 5),
            positions=[aa.irregular_grid.manual_1d([[1.0, 1.0]])],
            positions_threshold=1.0,
        )

        assert (masked_interferometer.visibilities == np.ones((19, 2))).all()
        assert (masked_interferometer.noise_map == 2.0 * np.ones((19,2))).all()
        assert (
            masked_interferometer.interferometer.uv_wavelengths == 3.0 * np.ones((19, 2))
        ).all()
        assert (masked_interferometer.primary_beam.in_2d == np.ones((5, 5))).all()
        assert masked_interferometer.trimmed_primary_beam_shape_2d == (5, 5)

        assert (masked_interferometer.positions[0] == np.array([[1.0, 1.0]])).all()
        assert masked_interferometer.positions_threshold == 1.0