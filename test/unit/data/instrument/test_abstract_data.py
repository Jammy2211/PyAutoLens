import os
import shutil
import logging

import numpy as np
import pytest

from astropy.modeling import functional_models
from astropy import units
from astropy.coordinates import Angle

from autolens import exc
from autolens.data.array.util import grid_util, mapping_util
from autolens.data.instrument import abstract_data

logger = logging.getLogger(__name__)

test_data_dir = "{}/../../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)

test_positions_dir = "{}/../../test_files/positions/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        noise = np.array([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(
            image=array, pixel_scale=1.0, psf=None, noise_map=noise
        )

        assert (data.signal_to_noise_map == np.array([[0.1, 0.2], [0.1, 1.0]])).all()
        assert data.signal_to_noise_max == 1.0

    def test__same_as_above__but_image_has_negative_values__replaced_with_zeros(self):
        array = np.array([[-1.0, 2.0], [3.0, -4.0]])

        noise = np.array([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(
            image=array, pixel_scale=1.0, psf=None, noise_map=noise
        )

        assert (data.signal_to_noise_map == np.array([[0.0, 0.2], [0.1, 0.0]])).all()
        assert data.signal_to_noise_max == 0.2


class TestAbsoluteSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = np.array([[-1.0, 2.0], [3.0, -4.0]])

        noise = np.array([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(
            image=array, pixel_scale=1.0, psf=None, noise_map=noise
        )

        assert (
            data.absolute_signal_to_noise_map == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert data.absolute_signal_to_noise_max == 1.0


class TestPotentialChiSquaredMap:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = np.array([[-1.0, 2.0], [3.0, -4.0]])

        noise = np.array([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(
            image=array, pixel_scale=1.0, psf=None, noise_map=noise
        )

        assert (
            data.potential_chi_squared_map
            == np.array([[0.1 ** 2.0, 0.2 ** 2.0], [0.1 ** 2.0, 1.0 ** 2.0]])
        ).all()
        assert data.potential_chi_squared_max == 1.0


class TestAbstractNoiseMap(object):
    class TestFromWeightMap:
        def test__weight_map_no_zeros__uses_1_over_sqrt_value(self):

            weight_map = np.array([[1.0, 4.0, 16.0], [1.0, 4.0, 16.0]])

            noise_map = abstract_data.AbstractNoiseMap.from_weight_map(
                weight_map=weight_map, pixel_scale=1.0
            )

            assert (noise_map == np.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]])).all()
            assert noise_map.origin == (0.0, 0.0)

        def test__weight_map_no_zeros__zeros_set_to_10000000(self):

            weight_map = np.array([[1.0, 4.0, 0.0], [1.0, 4.0, 16.0]])

            noise_map = abstract_data.AbstractNoiseMap.from_weight_map(
                weight_map=weight_map, pixel_scale=1.0
            )

            assert (noise_map == np.array([[1.0, 0.5, 1.0e8], [1.0, 0.5, 0.25]])).all()
            assert noise_map.origin == (0.0, 0.0)

    class TestFromInverseAbstractNoiseMap:
        def test__inverse_noise_map_no_zeros__uses_1_over_value(self):

            inverse_noise_map = np.array([[1.0, 4.0, 16.0], [1.0, 4.0, 16.0]])

            noise_map = abstract_data.AbstractNoiseMap.from_inverse_noise_map(
                inverse_noise_map=inverse_noise_map, pixel_scale=1.0
            )

            assert (
                noise_map == np.array([[1.0, 0.25, 0.0625], [1.0, 0.25, 0.0625]])
            ).all()
            assert noise_map.origin == (0.0, 0.0)


class TestPSF(object):
    class TestConstructors(object):
        def test__init__input_psf__all_attributes_correct_including_data_inheritance(
            self
        ):
            psf = abstract_data.PSF(
                array=np.ones((3, 3)), pixel_scale=1.0, renormalize=False
            )

            assert psf.shape == (3, 3)
            assert psf.pixel_scale == 1.0
            assert (psf == np.ones((3, 3))).all()
            assert psf.origin == (0.0, 0.0)

            psf = abstract_data.PSF(
                array=np.ones((4, 3)), pixel_scale=1.0, renormalize=False
            )

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.shape == (4, 3)
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_psf_3x3__all_attributes_correct_including_data_inheritance(
            self
        ):
            psf = abstract_data.PSF.from_fits_with_scale(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scale=1.0
            )

            assert (psf == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

            psf = abstract_data.PSF.from_fits_with_scale(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scale=1.0
            )

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

    class TestRenormalize(object):
        def test__input_is_already_normalized__no_change(self):
            psf_data = np.ones((3, 3)) / 9.0

            psf = abstract_data.PSF(array=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            psf_data = np.ones((3, 3))

            psf = abstract_data.PSF(array=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__same_as_above__renomalized_false_does_not_renormalize(self):
            psf_data = np.ones((3, 3))

            psf = abstract_data.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            assert psf == pytest.approx(np.ones((3, 3)), 1e-3)

    class TestBinnedUp(object):
        def test__psf_is_even_x_even__rescaled_to_odd_x_odd__no_use_of_dimension_trimming(
            self
        ):

            array_2d = np.ones((6, 6))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((9, 9))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.333333333333333, renormalize=True
            )
            assert psf.pixel_scale == 3.0
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((18, 6))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 27.0) * np.ones((9, 3))

            array_2d = np.ones((6, 18))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 27.0) * np.ones((3, 9))

        def test__psf_is_even_x_even_after_binning_up__resized_to_odd_x_odd_with_shape_plus_one(
            self
        ):

            array_2d = np.ones((2, 2))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == 0.4
            assert psf == (1.0 / 25.0) * np.ones((5, 5))

            array_2d = np.ones((40, 40))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.1, renormalize=True
            )
            assert psf.pixel_scale == 8.0
            assert psf == (1.0 / 25.0) * np.ones((5, 5))

            array_2d = np.ones((2, 4))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0 / 45.0) * np.ones((5, 9))

            array_2d = np.ones((4, 2))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0 / 45.0) * np.ones((9, 5))

        def test__psf_is_odd_and_even_after_binning_up__resized_to_odd_and_odd_with_shape_plus_one(
            self
        ):

            array_2d = np.ones((6, 4))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((9, 12))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.33333333333, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((3, 5))

            array_2d = np.ones((4, 6))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((12, 9))
            psf = abstract_data.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.33333333333, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((5, 3))

    class TestNewRenormalizedPsf(object):
        def test__input_is_already_normalized__no_change(self):

            psf_data = np.ones((3, 3)) / 9.0

            psf = abstract_data.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            psf_new = psf.new_psf_with_renormalized_array()

            assert psf_new == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            psf_data = np.ones((3, 3))

            psf = abstract_data.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            psf_new = psf.new_psf_with_renormalized_array()

            assert psf_new == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    class TestConvolve(object):
        def test__kernel_is_not_odd_x_odd__raises_error(self):
            kernel = np.array([[0.0, 1.0], [1.0, 2.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            with pytest.raises(exc.KernelException):
                psf.convolve(np.ones((5, 5)))

        def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == kernel).all()

        def test__image_is_4x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ).all()

        def test__image_is_4x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
                )
            ).all()

        def test__image_is_3x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                )
            ).all()

        def test__image_is_4x4_has_two_central_values__kernel_is_asymmetric__blurred_image_follows_convolution(
            self
        ):
            image = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 0.0],
                        [2.0, 3.0, 2.0, 1.0],
                        [1.0, 5.0, 5.0, 1.0],
                        [0.0, 1.0, 3.0, 3.0],
                    ]
                )
            ).all()

        def test__image_is_4x4_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
            self
        ):
            image = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [2.0, 1.0, 1.0, 1.0],
                        [3.0, 3.0, 2.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                    ]
                )
            ).all()

        def test__image_is_4x4_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
            self
        ):
            image = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

            psf = abstract_data.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [2.0, 1.0, 0.0, 0.0],
                        [3.0, 3.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 2.0, 2.0],
                    ]
                )
            ).all()

    class TestFromKernelNoBlurring(object):
        def test__correct_kernel(self):

            psf = abstract_data.PSF.from_no_blurring_kernel(pixel_scale=1.0)

            assert (
                psf == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            ).all()
            assert psf.pixel_scale == 1.0

            psf = abstract_data.PSF.from_no_blurring_kernel(pixel_scale=2.0)

            assert (
                psf == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            ).all()
            assert psf.pixel_scale == 2.0

    class TestFromGaussian(object):
        def test__identical_to_gaussian_light_profile(self):

            from autolens.model.profiles import light_profiles as lp

            grid = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
                mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=1
            )

            gaussian = lp.EllipticalGaussian(
                centre=(0.1, 0.1), axis_ratio=0.9, phi=45.0, intensity=1.0, sigma=1.0
            )
            profile_gaussian_1d = gaussian.intensities_from_grid(grid)
            profile_gaussian_2d = mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_grid_size(
                sub_array_1d=profile_gaussian_1d,
                mask=np.full(fill_value=False, shape=(3, 3)),
                sub_grid_size=1,
            )
            profile_psf = abstract_data.PSF(
                array=profile_gaussian_2d, pixel_scale=1.0, renormalize=True
            )

            imaging_psf = abstract_data.PSF.from_gaussian(
                shape=(3, 3),
                pixel_scale=1.0,
                centre=(0.1, 0.1),
                axis_ratio=0.9,
                phi=45.0,
                sigma=1.0,
            )

            assert profile_psf == pytest.approx(imaging_psf, 1e-4)

    class TestFromAlmaGaussian(object):
        def test__identical_to_astropy_gaussian_model__circular_no_rotation(self):

            pixel_scale = 0.1

            x_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            y_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )

            gaussian_astropy = functional_models.Gaussian2D(
                amplitude=1.0,
                x_mean=2.0,
                y_mean=2.0,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=0.0,
            )

            shape = (5, 5)
            y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = abstract_data.PSF.from_as_gaussian_via_alma_fits_header_parameters(
                shape=shape,
                pixel_scale=pixel_scale,
                y_stddev=2.0e-5,
                x_stddev=2.0e-5,
                theta=0.0,
            )

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__circular_no_rotation_different_pixel_scale(
            self
        ):

            pixel_scale = 0.02

            x_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            y_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )

            gaussian_astropy = functional_models.Gaussian2D(
                amplitude=1.0,
                x_mean=2.0,
                y_mean=2.0,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=0.0,
            )

            shape = (5, 5)
            y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = abstract_data.PSF.from_as_gaussian_via_alma_fits_header_parameters(
                shape=shape,
                pixel_scale=pixel_scale,
                y_stddev=2.0e-5,
                x_stddev=2.0e-5,
                theta=0.0,
            )

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_ellipticity_from_x_and_y_stddev(
            self
        ):

            pixel_scale = 0.1

            x_stddev = (
                1.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            y_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )

            theta_deg = 0.0
            theta = Angle(theta_deg, "deg").radian

            gaussian_astropy = functional_models.Gaussian2D(
                amplitude=1.0,
                x_mean=2.0,
                y_mean=2.0,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
            )

            shape = (5, 5)
            y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = abstract_data.PSF.from_as_gaussian_via_alma_fits_header_parameters(
                shape=shape,
                pixel_scale=pixel_scale,
                y_stddev=2.0e-5,
                x_stddev=1.0e-5,
                theta=theta_deg,
            )

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_different_ellipticity_from_x_and_y_stddev(
            self
        ):

            pixel_scale = 0.1

            x_stddev = (
                3.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            y_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )

            theta_deg = 0.0
            theta = Angle(theta_deg, "deg").radian

            gaussian_astropy = functional_models.Gaussian2D(
                amplitude=1.0,
                x_mean=2.0,
                y_mean=2.0,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
            )

            shape = (5, 5)
            y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = abstract_data.PSF.from_as_gaussian_via_alma_fits_header_parameters(
                shape=shape,
                pixel_scale=pixel_scale,
                y_stddev=2.0e-5,
                x_stddev=3.0e-5,
                theta=theta_deg,
            )

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_rotation_angle_30(self):

            pixel_scale = 0.1

            x_stddev = (
                1.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            y_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )

            theta_deg = 30.0
            theta = Angle(theta_deg, "deg").radian

            gaussian_astropy = functional_models.Gaussian2D(
                amplitude=1.0,
                x_mean=1.0,
                y_mean=1.0,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
            )

            shape = (3, 3)
            y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = abstract_data.PSF.from_as_gaussian_via_alma_fits_header_parameters(
                shape=shape,
                pixel_scale=pixel_scale,
                y_stddev=2.0e-5,
                x_stddev=1.0e-5,
                theta=theta_deg,
            )

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_rotation_angle_230(self):

            pixel_scale = 0.1

            x_stddev = (
                1.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            y_stddev = (
                2.0e-5
                * (units.deg).to(units.arcsec)
                / pixel_scale
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )

            theta_deg = 230.0
            theta = Angle(theta_deg, "deg").radian

            gaussian_astropy = functional_models.Gaussian2D(
                amplitude=1.0,
                x_mean=1.0,
                y_mean=1.0,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
            )

            shape = (3, 3)
            y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = abstract_data.PSF.from_as_gaussian_via_alma_fits_header_parameters(
                shape=shape,
                pixel_scale=pixel_scale,
                y_stddev=2.0e-5,
                x_stddev=1.0e-5,
                theta=theta_deg,
            )

            assert psf_astropy == pytest.approx(psf, 1e-4)


class TestExposureTimeMap(object):
    class TestFromExposureTimeAndBackgroundNoiseMap:
        def test__from_background_noise_map__covnerts_to_exposure_times(self):

            background_noise_map = np.array([[1.0, 4.0, 8.0], [1.0, 4.0, 8.0]])

            exposure_time_map = abstract_data.ExposureTimeMap.from_exposure_time_and_inverse_noise_map(
                pixel_scale=0.1,
                exposure_time=1.0,
                inverse_noise_map=background_noise_map,
            )

            assert (
                exposure_time_map == np.array([[0.125, 0.5, 1.0], [0.125, 0.5, 1.0]])
            ).all()
            assert exposure_time_map.origin == (0.0, 0.0)

            exposure_time_map = abstract_data.ExposureTimeMap.from_exposure_time_and_inverse_noise_map(
                pixel_scale=0.1,
                exposure_time=3.0,
                inverse_noise_map=background_noise_map,
            )

            assert (
                exposure_time_map == np.array([[0.375, 1.5, 3.0], [0.375, 1.5, 3.0]])
            ).all()
            assert exposure_time_map.origin == (0.0, 0.0)


class TestDataFromFits(object):
    def test__psf_renormalized_true__renormalized_psf(self):

        psf = abstract_data.load_psf(
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            psf_hdu=0,
            renormalize=True,
        )

        assert psf == pytest.approx((1.0 / 9.0) * np.ones((3, 3)), 1e-2)


class TestPositionsToFile(object):
    def test__load_positions__retains_list_structure(self):

        positions = abstract_data.load_positions(
            positions_path=test_positions_dir + "positions_test.dat"
        )

        assert positions == [
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0], [5.0, 6.0]],
        ]

    def test__output_positions(self):

        positions = [[[4.0, 4.0], [5.0, 5.0]], [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]]

        output_data_dir = "{}/../test_files/positions/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        abstract_data.output_positions(
            positions=positions, positions_path=output_data_dir + "positions_test.dat"
        )

        positions = abstract_data.load_positions(
            positions_path=output_data_dir + "positions_test.dat"
        )

        assert positions == [
            [[4.0, 4.0], [5.0, 5.0]],
            [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
        ]
