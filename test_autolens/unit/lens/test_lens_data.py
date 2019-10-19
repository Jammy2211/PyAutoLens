import autolens as al
import numpy as np
import pytest


@pytest.fixture(name="lens_imaging_7x7")
def make_lens_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return al.LensImagingData(imaging=imaging_7x7, mask=sub_mask_7x7)


@pytest.fixture(name="lens_imaging_6x6")
def make_lens_imaging_6x6(imaging_6x6, mask_6x6):
    return al.LensImagingData(imaging=imaging_6x6, mask=mask_6x6)


@pytest.fixture(name="lens_interferometer_7")
def make_lens_interferometer_7(interferometer_7, sub_mask_7x7):
    return al.LensUVPlaneData(interferometer=interferometer_7, mask=sub_mask_7x7)


class TestAbstractLensData(object):
    def test__pixel_scale_interpolation_grid_input__grids_nclude_interpolators(
        self, sub_mask_7x7
    ):

        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, pixel_scale_interpolation_grid=1.0
        )

        grid = al.Grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        assert (lens_imaging_7x7.grid == new_grid).all()
        assert (
            lens_imaging_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx
        ).all()
        assert (
            lens_imaging_7x7.grid.interpolator.wts == new_grid.interpolator.wts
        ).all()

    def test__pixel_scale_binned_grid_is_input__correct_binned_up_grid_calculated(
        self, sub_mask_7x7, grid_7x7
    ):
        sub_mask_7x7.pixel_scales = 1.0
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert lens_imaging_7x7.grid.binned.bin_up_factor == 1
        assert (
            lens_imaging_7x7.mask == lens_imaging_7x7.grid.binned.mask
        ).all()
        assert (lens_imaging_7x7.grid.binned == grid_7x7).all()
        assert (
            lens_imaging_7x7.grid.binned.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        sub_mask_7x7.pixel_scales = 1.0
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, pixel_scale_binned_grid=1.9
        )

        assert lens_imaging_7x7.grid.binned.bin_up_factor == 1
        assert (
            lens_imaging_7x7.mask == lens_imaging_7x7.grid.binned.mask
        ).all()
        assert (
            lens_imaging_7x7.grid.binned.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        sub_mask_7x7.pixel_scales = 1.0
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, pixel_scale_binned_grid=2.0
        )
        assert lens_imaging_7x7.grid.binned.bin_up_factor == 2
        assert (
            lens_imaging_7x7.grid.binned.mask
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
            lens_imaging_7x7.grid.binned
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            lens_imaging_7x7.grid.binned.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0, -1, -1, -1], [1, 2, -1, -1], [3, 6, -1, -1], [4, 5, 7, 8]])
        ).all()

        sub_mask_7x7.pixel_scales = 2.0
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert lens_imaging_7x7.grid.binned.bin_up_factor == 1

        sub_mask_7x7.pixel_scales = 1.0
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, pixel_scale_binned_grid=None
        )

        assert lens_imaging_7x7.grid.binned == None

    def test__inversion_pixel_limit(self, sub_mask_7x7):
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, inversion_pixel_limit=2
        )

        assert lens_imaging_7x7.inversion_pixel_limit == 2

        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, inversion_pixel_limit=5
        )

        assert lens_imaging_7x7.inversion_pixel_limit == 5

    def test__hyper_noise_map_max(self, sub_mask_7x7):
        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, hyper_noise_map_max=10.0
        )

        assert lens_imaging_7x7.hyper_noise_map_max == 10.0

        lens_imaging_7x7 = al.AbstractLensData(
            mask=sub_mask_7x7, hyper_noise_map_max=20.0
        )

        assert lens_imaging_7x7.hyper_noise_map_max == 20.0


class TestLensImagingData(object):
    def test__attributes(self, imaging_7x7, lens_imaging_7x7):
        assert lens_imaging_7x7.pixel_scales == imaging_7x7.pixel_scales
        assert lens_imaging_7x7.pixel_scales == (1.0, 1.0)

        assert (
            lens_imaging_7x7.image(return_in_2d=True, return_masked=False)
            == imaging_7x7.image
        ).all()
        assert (
            lens_imaging_7x7.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()

        assert (
            lens_imaging_7x7.noise_map(return_in_2d=True, return_masked=False)
            == imaging_7x7.noise_map
        ).all()
        assert (
            lens_imaging_7x7.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (lens_imaging_7x7.psf == imaging_7x7.psf).all()
        assert (lens_imaging_7x7.psf == np.ones((3, 3))).all()

        assert lens_imaging_7x7.trimmed_psf_shape == (3, 3)

    def test__masking(self, lens_imaging_7x7):

        assert (
            lens_imaging_7x7._mask_1d == np.full(fill_value=False, shape=(9))
        ).all()
        assert (lens_imaging_7x7.image.in_1d == np.ones(9)).all()
        assert (lens_imaging_7x7.noise_map.in_1d == 2.0 * np.ones(9)).all()

        assert (
            lens_imaging_7x7.mask
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
            lens_imaging_7x7.image(return_in_2d=True, return_masked=True)
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
            lens_imaging_7x7.noise_map(return_in_2d=True, return_masked=True)
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

    def test__grids(
        self, lens_imaging_7x7, grid_7x7, sub_grid_7x7, blurring_grid_7x7
    ):

        assert (lens_imaging_7x7.grid.unlensed_unsubbed_1d == grid_7x7).all()
        assert (lens_imaging_7x7.grid == sub_grid_7x7).all()
        assert (lens_imaging_7x7.blurring_grid == blurring_grid_7x7).all()

    def test__pixel_scale_interpolation_grid_input__grids_include_interpolator_on_blurring_grid(
        self, imaging_7x7, sub_mask_7x7
    ):

        lens_imaging_7x7 = al.LensImagingData(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            pixel_scale_interpolation_grid=1.0,
        )

        grid = al.Grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        blurring_grid = al.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=sub_mask_7x7, psf_shape=(3, 3)
        )
        new_blurring_grid = blurring_grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (lens_imaging_7x7.grid == new_grid).all()
        assert (
            lens_imaging_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx
        ).all()
        assert (
            lens_imaging_7x7.grid.interpolator.wts == new_grid.interpolator.wts
        ).all()

        assert (lens_imaging_7x7.blurring_grid == new_blurring_grid).all()
        assert (
            lens_imaging_7x7.blurring_grid.interpolator.vtx
            == new_blurring_grid.interpolator.vtx
        ).all()
        assert (
            lens_imaging_7x7.blurring_grid.interpolator.wts
            == new_blurring_grid.interpolator.wts
        ).all()

    def test__convolvers(self, lens_imaging_7x7):
        assert type(lens_imaging_7x7.convolver) == al.Convolver

    def test__different_imaging_without_mock_objects__customize_constructor_inputs(
        self
    ):

        psf = aa.kernel(np.ones((7, 7)), 1)
        imaging = al.ImagingData(
            np.ones((19, 19)),
            pixel_scales=3.0,
            psf=psf,
            noise_map=2.0 * np.ones((19, 19)),
        )
        mask = al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
            shape=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        mask[9, 9] = False

        lens_imaging_7x7 = al.LensImagingData(
            imaging=imaging,
            mask=mask,
            trimmed_psf_shape=(7, 7),
            positions=[np.array([[1.0, 1.0]])],
            positions_threshold=1.0,
        )

        assert (
            lens_imaging_7x7.image(return_in_2d=True, return_masked=False)
            == np.ones((19, 19))
        ).all()
        assert (
            lens_imaging_7x7.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((19, 19))
        ).all()
        assert (lens_imaging_7x7.psf == np.ones((7, 7))).all()

        assert lens_imaging_7x7.sub_size == 8
        assert lens_imaging_7x7.convolver.psf.shape_2d == (7, 7)
        assert (lens_imaging_7x7.positions[0] == np.array([[1.0, 1.0]])).all()
        assert lens_imaging_7x7.positions_threshold == 1.0

        assert lens_imaging_7x7.trimmed_psf_shape == (7, 7)

    def test__lens_imaging_6x6_with_binned_up_imaging(
        self, lens_imaging_6x6
    ):
        binned_up_psf = lens_imaging_6x6.imaging.psf.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5
        )

        lens_imaging_6x6 = lens_imaging_6x6.new_lens_imaging_with_binned_up_imaging_and_mask(
            bin_up_factor=2
        )

        assert (
            lens_imaging_6x6.image(return_in_2d=True, return_masked=False)
            == np.ones((3, 3))
        ).all()
        assert (lens_imaging_6x6.psf == binned_up_psf).all()
        assert (
            lens_imaging_6x6.noise_map(return_in_2d=True, return_masked=False)
            == np.ones((3, 3))
        ).all()
        assert (
            lens_imaging_6x6.imaging.background_noise_map
            == 1.5 * np.ones((3, 3))
        ).all()
        assert (
            lens_imaging_6x6.imaging.poisson_noise_map
            == 2.0 * np.ones((3, 3))
        ).all()
        assert (
            lens_imaging_6x6.imaging.exposure_time_map
            == 20.0 * np.ones((3, 3))
        ).all()
        assert (
            lens_imaging_6x6.imaging.background_sky_map
            == 6.0 * np.ones((3, 3))
        ).all()

        assert (
            lens_imaging_6x6.mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        assert (lens_imaging_6x6.image.in_1d == np.ones((1))).all()
        assert (lens_imaging_6x6.noise_map.in_1d == np.ones((1))).all()

    def test__lens_imaging_7x7_with_signal_to_noise_limit(
        self, imaging_7x7, lens_imaging_7x7
    ):

        lens_data_snr_limit = lens_imaging_7x7.new_lens_imaging_with_signal_to_noise_limit(
            signal_to_noise_limit=0.25
        )

        assert lens_data_snr_limit.pixel_scales == imaging_7x7.pixel_scales
        assert lens_data_snr_limit.pixel_scales == (1.0, 1.0)

        assert (
            lens_data_snr_limit.image(return_in_2d=True, return_masked=False)
            == imaging_7x7.image
        ).all()
        assert (
            lens_data_snr_limit.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()

        assert (
            lens_data_snr_limit.noise_map(return_in_2d=True, return_masked=False)
            == 4.0 * np.ones((7, 7))
        ).all()

        assert (lens_data_snr_limit.psf == imaging_7x7.psf).all()
        assert (lens_data_snr_limit.psf == np.ones((3, 3))).all()

        assert lens_data_snr_limit.trimmed_psf_shape == (3, 3)

        assert (lens_data_snr_limit.image.in_1d == np.ones(9)).all()
        assert (lens_data_snr_limit.noise_map.in_1d == 4.0 * np.ones(9)).all()

        assert (
            lens_data_snr_limit.noise_map(return_in_2d=True, return_masked=True)
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


class TestLensUVPlaneData(object):
    def test__attributes(self, interferometer_7, lens_interferometer_7, sub_mask_7x7):

        assert lens_interferometer_7.pixel_scales == interferometer_7.pixel_scales
        assert lens_interferometer_7.pixel_scales == (1.0, 1.0)

        assert (
            lens_interferometer_7.visibilities() == interferometer_7.visibilities
        ).all()
        assert (lens_interferometer_7.visibilities() == np.ones((7, 2))).all()

        assert (lens_interferometer_7.noise_map() == interferometer_7.noise_map).all()
        assert (lens_interferometer_7.noise_map() == 2.0 * np.ones((7))).all()
        assert (
            lens_interferometer_7.noise_map(return_x2=True)[:, 0] == 2.0 * np.ones((7))
        ).all()
        assert (
            lens_interferometer_7.noise_map(return_x2=True)[:, 1] == 2.0 * np.ones((7))
        ).all()

        assert (
            lens_interferometer_7.visibilities_mask
            == np.full(fill_value=False, shape=(7, 2))
        ).all()

        assert (lens_interferometer_7.primary_beam == interferometer_7.primary_beam).all()
        assert (lens_interferometer_7.primary_beam == np.ones((3, 3))).all()
        assert lens_interferometer_7.trimmed_primary_beam_shape == (3, 3)

        assert (
            lens_interferometer_7.interferometer.uv_wavelengths
            == interferometer_7.uv_wavelengths
        ).all()
        assert lens_interferometer_7.interferometer.uv_wavelengths[0, 0] == -55636.4609375

    def test__grids(self, lens_interferometer_7, grid_7x7, sub_grid_7x7):
        assert (lens_interferometer_7.grid.unlensed_unsubbed_1d == grid_7x7).all()
        assert (lens_interferometer_7.grid == sub_grid_7x7).all()

    def test__transformer(self, lens_interferometer_7):
        assert type(lens_interferometer_7.transformer) == al.Transformer

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self
    ):
        primary_beam = al.PrimaryBeam(np.ones((7, 7)), 1)
        interferometer = al.UVPlaneData(
            shape_2d=(2, 2),
            visibilities=np.ones((19, 2)),
            pixel_scales=3.0,
            primary_beam=primary_beam,
            noise_map=2.0 * np.ones((19,)),
            uv_wavelengths=3.0 * np.ones((19, 2)),
        )
        mask = al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
            shape=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        mask[9, 9] = False

        lens_interferometer_7 = al.LensUVPlaneData(
            interferometer=interferometer,
            mask=mask,
            trimmed_primary_beam_shape=(7, 7),
            positions=[np.array([[1.0, 1.0]])],
            positions_threshold=1.0,
        )

        assert (lens_interferometer_7.visibilities() == np.ones((19, 2))).all()
        assert (lens_interferometer_7.noise_map() == 2.0 * np.ones((19,))).all()
        assert (
            lens_interferometer_7.interferometer.uv_wavelengths == 3.0 * np.ones((19, 2))
        ).all()
        assert (lens_interferometer_7.primary_beam == np.ones((7, 7))).all()

        assert lens_interferometer_7.sub_size == 8
        assert (lens_interferometer_7.positions[0] == np.array([[1.0, 1.0]])).all()
        assert lens_interferometer_7.positions_threshold == 1.0

        assert lens_interferometer_7.trimmed_primary_beam_shape == (7, 7)

    def test__lens_interferometer_7_with_modified_visibilities(
        self, lens_interferometer_7
    ):
        lens_interferometer_7 = lens_interferometer_7.new_lens_imaging_with_modified_visibilities(
            modified_visibilities=8.0 * np.ones((7, 2))
        )

        assert (lens_interferometer_7.visibilities() == 8.0 * np.ones((7, 2))).all()
