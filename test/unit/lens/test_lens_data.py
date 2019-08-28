import autolens as al
import numpy as np
import pytest


@pytest.fixture(name="lens_data_7x7")
def make_lens_data_7x7(ccd_data_7x7, mask_7x7):
    return al.LensData(ccd_data=ccd_data_7x7, mask=mask_7x7)


@pytest.fixture(name="lens_data_6x6")
def make_lens_data_6x6(ccd_data_6x6, mask_6x6):
    return al.LensData(ccd_data=ccd_data_6x6, mask=mask_6x6)


class TestLensData(object):
    def test__attributes(self, ccd_data_7x7, lens_data_7x7):
        assert lens_data_7x7.pixel_scale == ccd_data_7x7.pixel_scale
        assert lens_data_7x7.pixel_scale == 1.0

        assert (lens_data_7x7.unmasked_image == ccd_data_7x7.image).all()
        assert (lens_data_7x7.unmasked_image == np.ones((7, 7))).all()

        assert (lens_data_7x7.unmasked_noise_map == ccd_data_7x7.noise_map).all()
        assert (lens_data_7x7.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (lens_data_7x7.psf == ccd_data_7x7.psf).all()
        assert (lens_data_7x7.psf == np.ones((3, 3))).all()

        assert lens_data_7x7.trimmed_psf_shape == (3, 3)

    def test__masking(self, lens_data_7x7):

        assert (lens_data_7x7.mask_1d == np.full(fill_value=False, shape=(9))).all()
        assert (lens_data_7x7.image_1d == np.ones(9)).all()
        assert (lens_data_7x7.noise_map_1d == 2.0 * np.ones(9)).all()

        assert (
            lens_data_7x7.mask_2d
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            lens_data_7x7.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            lens_data_7x7.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__grid_stack(
        self, lens_data_7x7, grid_7x7, sub_grid_7x7, blurring_grid_7x7
    ):

        assert (lens_data_7x7.grid.unlensed_unsubbed_1d == grid_7x7).all()
        assert (lens_data_7x7.grid == sub_grid_7x7).all()
        assert (lens_data_7x7.preload_blurring_grid == blurring_grid_7x7).all()

    def test__pixel_scale_interpolation_grid_input__grids_nclude_interpolators(
        self, ccd_data_7x7, mask_7x7
    ):

        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, pixel_scale_interpolation_grid=1.0
        )

        grid = al.Grid.from_mask_and_sub_grid_size(mask=mask_7x7, sub_grid_size=2)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        blurring_grid = al.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=mask_7x7, psf_shape=(3, 3)
        )
        new_blurring_grid = blurring_grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (lens_data_7x7.grid == new_grid).all()
        assert (lens_data_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx).all()
        assert (lens_data_7x7.grid.interpolator.wts == new_grid.interpolator.wts).all()

        assert (lens_data_7x7.preload_blurring_grid == new_blurring_grid).all()
        assert (
            lens_data_7x7.preload_blurring_grid.interpolator.vtx
            == new_blurring_grid.interpolator.vtx
        ).all()
        assert (
            lens_data_7x7.preload_blurring_grid.interpolator.wts
            == new_blurring_grid.interpolator.wts
        ).all()

    def test__pixel_scale_binned_grid_is_input__correct_binned_up_grid_calculated(
        self, ccd_data_7x7, mask_7x7, grid_7x7
    ):
        ccd_data_7x7.pixel_scale = 1.0
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert lens_data_7x7.grid.binned.bin_up_factor == 1
        assert (lens_data_7x7.mask_2d == lens_data_7x7.grid.binned.mask).all()
        assert (lens_data_7x7.grid.binned == grid_7x7).all()
        assert (
            lens_data_7x7.grid.binned.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        ccd_data_7x7.pixel_scale = 1.0
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, pixel_scale_binned_grid=1.9
        )

        assert lens_data_7x7.grid.binned.bin_up_factor == 1
        assert (lens_data_7x7.mask_2d == lens_data_7x7.grid.binned.mask).all()
        assert (
            lens_data_7x7.grid.binned.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        ccd_data_7x7.pixel_scale = 1.0
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, pixel_scale_binned_grid=2.0
        )
        assert lens_data_7x7.grid.binned.bin_up_factor == 2
        assert (
            lens_data_7x7.grid.binned.mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()
        assert (
            lens_data_7x7.grid.binned
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            lens_data_7x7.grid.binned.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0, -1, -1, -1], [1, 2, -1, -1], [3, 6, -1, -1], [4, 5, 7, 8]])
        ).all()

        ccd_data_7x7.pixel_scale = 2.0
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert lens_data_7x7.grid.binned.bin_up_factor == 1

        ccd_data_7x7.pixel_scale = 1.0
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, pixel_scale_binned_grid=None
        )

        assert lens_data_7x7.grid.binned == None

    def test__convolvers(self, lens_data_7x7):
        assert type(lens_data_7x7.convolver) == al.Convolver

    def test__inversion_pixel_limit(self, ccd_data_7x7, mask_7x7):
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, inversion_pixel_limit=2
        )

        assert lens_data_7x7.inversion_pixel_limit == 2

        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, inversion_pixel_limit=5
        )

        assert lens_data_7x7.inversion_pixel_limit == 5

    def test__hyper_noise_map_max(self, ccd_data_7x7, mask_7x7):
        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, hyper_noise_map_max=10.0
        )

        assert lens_data_7x7.hyper_noise_map_max == 10.0

        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data_7x7, mask=mask_7x7, hyper_noise_map_max=20.0
        )

        assert lens_data_7x7.hyper_noise_map_max == 20.0

    def test__different_ccd_data_without_mock_objects__customize_constructor_inputs(
        self
    ):

        psf = al.PSF(np.ones((7, 7)), 1)
        ccd_data = al.CCDData(
            np.ones((19, 19)),
            pixel_scale=3.0,
            psf=psf,
            noise_map=2.0 * np.ones((19, 19)),
        )
        mask = al.Mask.unmasked_for_shape_and_pixel_scale(
            shape=(19, 19), pixel_scale=1.0, invert=True
        )
        mask[9, 9] = False

        lens_data_7x7 = al.LensData(
            ccd_data=ccd_data,
            mask=mask,
            sub_grid_size=8,
            trimmed_psf_shape=(7, 7),
            positions=[np.array([[1.0, 1.0]])],
            positions_threshold=1.0,
        )

        assert (lens_data_7x7.unmasked_image == np.ones((19, 19))).all()
        assert (lens_data_7x7.unmasked_noise_map == 2.0 * np.ones((19, 19))).all()
        assert (lens_data_7x7.psf == np.ones((7, 7))).all()

        assert lens_data_7x7.sub_grid_size == 8
        assert lens_data_7x7.convolver.psf.shape == (7, 7)
        assert (lens_data_7x7.positions[0] == np.array([[1.0, 1.0]])).all()
        assert lens_data_7x7.positions_threshold == 1.0

        assert lens_data_7x7.trimmed_psf_shape == (7, 7)

    def test__lens_data_7x7_with_modified_image(self, lens_data_7x7):

        lens_data_7x7 = lens_data_7x7.new_lens_data_with_modified_image(
            modified_image=8.0 * np.ones((7, 7))
        )

        assert (lens_data_7x7.unmasked_image == 8.0 * np.ones((7, 7))).all()

        assert (lens_data_7x7.image_1d == 8.0 * np.ones(9)).all()

        assert (
            lens_data_7x7.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 8.0, 8.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 8.0, 8.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 8.0, 8.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__lens_data_6x6_with_binned_up_ccd_data(self, lens_data_6x6):
        binned_up_psf = lens_data_6x6.ccd_data.psf.new_psf_with_rescaled_odd_dimensioned_array(
            rescale_factor=0.5
        )

        lens_data_6x6 = lens_data_6x6.new_lens_data_with_binned_up_ccd_data_and_mask(
            bin_up_factor=2
        )

        assert (lens_data_6x6.unmasked_image == np.ones((3, 3))).all()
        assert (lens_data_6x6.psf == binned_up_psf).all()
        assert (lens_data_6x6.unmasked_noise_map == np.ones((3, 3))).all()
        assert (
            lens_data_6x6.ccd_data.background_noise_map == 1.5 * np.ones((3, 3))
        ).all()
        assert (lens_data_6x6.ccd_data.poisson_noise_map == 2.0 * np.ones((3, 3))).all()
        assert (
            lens_data_6x6.ccd_data.exposure_time_map == 20.0 * np.ones((3, 3))
        ).all()
        assert (
            lens_data_6x6.ccd_data.background_sky_map == 6.0 * np.ones((3, 3))
        ).all()

        assert (
            lens_data_6x6.mask_2d
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        assert (lens_data_6x6.image_1d == np.ones((1))).all()
        assert (lens_data_6x6.noise_map_1d == np.ones((1))).all()

    def test__lens_data_7x7_with_signal_to_noise_limit(
        self, ccd_data_7x7, lens_data_7x7
    ):

        lens_data_snr_limit = lens_data_7x7.new_lens_data_with_signal_to_noise_limit(
            signal_to_noise_limit=0.25
        )

        assert lens_data_snr_limit.pixel_scale == ccd_data_7x7.pixel_scale
        assert lens_data_snr_limit.pixel_scale == 1.0

        assert (lens_data_snr_limit.unmasked_image == ccd_data_7x7.image).all()
        assert (lens_data_snr_limit.unmasked_image == np.ones((7, 7))).all()

        assert (lens_data_snr_limit.unmasked_noise_map == 4.0 * np.ones((7, 7))).all()

        assert (lens_data_snr_limit.psf == ccd_data_7x7.psf).all()
        assert (lens_data_snr_limit.psf == np.ones((3, 3))).all()

        assert lens_data_snr_limit.trimmed_psf_shape == (3, 3)

        assert (lens_data_snr_limit.image_1d == np.ones(9)).all()
        assert (lens_data_snr_limit.noise_map_1d == 4.0 * np.ones(9)).all()

        assert (
            lens_data_snr_limit.noise_map(return_in_2d=True)
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
