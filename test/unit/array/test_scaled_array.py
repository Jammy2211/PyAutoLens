import os

import numpy as np
import pytest

import autolens as al
from autolens import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)

class TestConstructorMethods:

    def test__square_pixel_array__input_scaled_array__centre_is_origin(self):

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(
            array_2d=np.ones((3, 3)), pixel_scale=1.0
        )

        assert (scaled_array.in_1d == np.ones((9,))).all()
        assert (scaled_array.in_2d == np.ones((3,3))).all()
        assert scaled_array.geometry.pixel_scale == 1.0
        assert scaled_array.geometry.shape == (3, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((3.0, 3.0))
        assert scaled_array.geometry.arc_second_maxima == (1.5, 1.5)
        assert scaled_array.geometry.arc_second_minima == (-1.5, -1.5)

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(
            array_2d=np.ones((3, 4)), pixel_scale=0.1
        )

        assert (scaled_array.in_1d == np.ones((12,))).all()
        assert (scaled_array.in_2d == np.ones((3, 4))).all()
        assert scaled_array.geometry.pixel_scale == 0.1
        assert scaled_array.geometry.shape == (3, 4)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.5)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((0.3, 0.4))
        assert scaled_array.geometry.arc_second_maxima == pytest.approx((0.15, 0.2), 1e-4)
        assert scaled_array.geometry.arc_second_minima == pytest.approx((-0.15, -0.2), 1e-4)

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(
            array_2d=np.ones((4, 3)), pixel_scale=0.1, origin=(1.0, 1.0)
        )

        assert (scaled_array.in_1d == np.ones((12,))).all()
        assert (scaled_array.in_2d == np.ones((4, 3))).all()
        assert scaled_array.geometry.pixel_scale == 0.1
        assert scaled_array.geometry.shape == (4, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.5, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((0.4, 0.3))
        assert scaled_array.geometry.arc_second_maxima == pytest.approx((1.2, 1.15), 1e-4)
        assert scaled_array.geometry.arc_second_minima == pytest.approx((0.8, 0.85), 1e-4)

    def test__rectangular_pixel_array__input_scaled_array(self):

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scales(
            array_2d=np.ones((3, 3)), pixel_scales=(2.0, 1.0)
        )

        assert scaled_array.in_1d == pytest.approx(np.ones((9, )), 1e-4)
        assert scaled_array.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
        assert scaled_array.geometry.pixel_scales == (2.0, 1.0)
        assert scaled_array.geometry.shape == (3, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
        assert scaled_array.geometry.arc_second_maxima == pytest.approx((3.0, 1.5), 1e-4)
        assert scaled_array.geometry.arc_second_minima == pytest.approx((-3.0, -1.5), 1e-4)

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scales(
            array_2d=np.ones((4, 3)), pixel_scales=(0.2, 0.1)
        )

        assert scaled_array.in_1d == pytest.approx(np.ones((12,)), 1e-4)
        assert scaled_array.in_2d == pytest.approx(np.ones((4, 3)), 1e-4)
        assert scaled_array.geometry.pixel_scales == (0.2, 0.1)
        assert scaled_array.geometry.shape == (4, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.5, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((0.8, 0.3), 1e-3)
        assert scaled_array.geometry.arc_second_maxima == pytest.approx((0.4, 0.15), 1e-4)
        assert scaled_array.geometry.arc_second_minima == pytest.approx((-0.4, -0.15), 1e-4)

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scales(
            array_2d=np.ones((3, 4)), pixel_scales=(0.1, 0.2)
        )

        assert scaled_array.in_1d == pytest.approx(np.ones((12, )), 1e-4)
        assert scaled_array.in_2d == pytest.approx(np.ones((3, 4)), 1e-4)
        assert scaled_array.geometry.pixel_scales == (0.1, 0.2)
        assert scaled_array.geometry.shape == (3, 4)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.5)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((0.3, 0.8), 1e-3)
        assert scaled_array.geometry.arc_second_maxima == pytest.approx((0.15, 0.4), 1e-4)
        assert scaled_array.geometry.arc_second_minima == pytest.approx((-0.15, -0.4), 1e-4)

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scales(
            array_2d=np.ones((3, 3)), pixel_scales=(2.0, 1.0), origin=(-1.0, -2.0)
        )

        assert scaled_array.in_1d == pytest.approx(np.ones((9, )), 1e-4)
        assert scaled_array.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
        assert scaled_array.geometry.pixel_scales == (2.0, 1.0)
        assert scaled_array.geometry.shape == (3, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
        assert scaled_array.geometry.origin == (-1.0, -2.0)
        assert scaled_array.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert scaled_array.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

    def test__init__input_scaled_array_single_value__all_attributes_correct_including_data_inheritance(
        self
    ):
        scaled_array = al.ScaledArray.from_single_value_shape_and_pixel_scale(
            value=5.0, shape=(3, 3), pixel_scale=1.0, origin=(1.0, 1.0)
        )

        assert (scaled_array.in_2d == 5.0 * np.ones((3, 3))).all()
        assert scaled_array.geometry.pixel_scale == 1.0
        assert scaled_array.geometry.shape == (3, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((3.0, 3.0))
        assert scaled_array.geometry.origin == (1.0, 1.0)

    def test__from_fits__all_attributes_correct_including_data_inheritance(
        self
    ):
        scaled_array = al.ScaledArray.from_fits_with_pixel_scale(
            file_path=test_data_dir + "3x3_ones.fits",
            hdu=0,
            pixel_scale=1.0,
            origin=(1.0, 1.0),
        )

        assert (scaled_array.in_2d == np.ones((3, 3))).all()
        assert scaled_array.geometry.pixel_scale == 1.0
        assert scaled_array.geometry.shape == (3, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((3.0, 3.0))
        assert scaled_array.geometry.origin == (1.0, 1.0)

        scaled_array = al.ScaledArray.from_fits_with_pixel_scale(
            file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scale=0.1
        )

        assert (scaled_array.in_2d == np.ones((4, 3))).all()
        assert scaled_array.geometry.pixel_scale == 0.1
        assert scaled_array.geometry.shape == (4, 3)
        assert scaled_array.geometry.central_pixel_coordinates == (1.5, 1.0)
        assert scaled_array.geometry.shape_arcsec == pytest.approx((0.4, 0.3))


class TestNewScaledArrayResized:
    def test__pad__compare_to_array_util(self):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=array_2d, pixel_scale=1.0)

        scaled_array = scaled_array.new_scaled_array_resized_from_new_shape(
            new_shape=(7, 7), new_centre_pixels=(1, 1)
        )

        scaled_array_resized_manual = np.array([[0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 1., 1., 1., 1., 1.],
                                                [0. ,0., 1., 1., 1., 1., 1.],
                                                [0., 0., 1., 1., 2., 1., 1.],
                                                [0., 0., 1., 1., 1., 1., 1.],
                                                [0., 0., 1., 1., 1., 1., 1.]])

        assert type(scaled_array) == al.ScaledArray
        assert (scaled_array.in_2d == scaled_array_resized_manual).all()
        assert scaled_array.geometry.pixel_scale == 1.0

    def test__trim__compare_to_array_util(self):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=array_2d, pixel_scale=1.0)

        scaled_array = scaled_array.new_scaled_array_resized_from_new_shape(
            new_shape=(3, 3), new_centre_pixels=(4, 4)
        )

        scaled_array_resized_manual = np.array([[1., 1., 0.],
                                                [1., 1., 0.],
                                                [0., 0., 0.]])


        assert type(scaled_array) == al.ScaledArray
        assert (scaled_array.in_2d == scaled_array_resized_manual).all()
        assert scaled_array.geometry.pixel_scale == 1.0

    def test__new_centre_is_in_arcsec(self):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        array = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=array_2d, pixel_scale=3.0)

        scaled_array = array.new_scaled_array_resized_from_new_shape(
            new_shape=(3, 3), new_centre_arcsec=(6.0, 6.0)
        )
        scaled_array_util = al.array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=array_2d, resized_shape=(3, 3), origin=(0, 4)
        )
        assert (scaled_array.in_2d == scaled_array_util).all()

        scaled_array = array.new_scaled_array_resized_from_new_shape(
            new_shape=(3, 3), new_centre_arcsec=(7.49, 4.51)
        )
        scaled_array_util = al.array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=array_2d, resized_shape=(3, 3), origin=(0, 4)
        )
        assert (scaled_array.in_2d == scaled_array_util).all()

        scaled_array = array.new_scaled_array_resized_from_new_shape(
            new_shape=(3, 3), new_centre_arcsec=(7.49, 7.49)
        )
        scaled_array_util = al.array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=array_2d, resized_shape=(3, 3), origin=(0, 4)
        )
        assert (scaled_array.in_2d == scaled_array_util).all()

        scaled_array = array.new_scaled_array_resized_from_new_shape(
            new_shape=(3, 3), new_centre_arcsec=(4.51, 4.51)
        )
        scaled_array_util = al.array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=array_2d, resized_shape=(3, 3), origin=(0, 4)
        )
        assert (scaled_array.in_2d == scaled_array_util).all()

        scaled_array = array.new_scaled_array_resized_from_new_shape(
            new_shape=(3, 3), new_centre_arcsec=(4.51, 7.49)
        )
        scaled_array_util = al.array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=array_2d, resized_shape=(3, 3), origin=(0, 4)
        )
        assert (scaled_array.in_2d == scaled_array_util).all()


class TestNewScaledArrayZoomed:
    def test__2d_array_zoomed__uses_the_limits_of_the_mask(self):
        array_2d = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=array_2d, pixel_scale=1.0)

        mask = al.Mask(
            array_2d=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        scaled_array_zoomed = scaled_array.new_scaled_array_zoomed_from_mask(mask=mask, buffer=0)
        assert (scaled_array_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

        mask = al.Mask(
            array_2d=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, False],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        scaled_array_zoomed = scaled_array.new_scaled_array_zoomed_from_mask(mask=mask, buffer=0)
        assert (
            scaled_array_zoomed.in_2d == np.array([[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]])
        ).all()

        mask = al.Mask(
            array_2d=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, False, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        scaled_array_zoomed = scaled_array.new_scaled_array_zoomed_from_mask(mask=mask, buffer=0)
        assert (
            scaled_array_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]])
        ).all()

        mask = al.Mask(
            array_2d=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [False, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        scaled_array_zoomed = scaled_array.new_scaled_array_zoomed_from_mask(mask=mask, buffer=0)
        assert (
            scaled_array_zoomed.in_2d == np.array([[5.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
        ).all()

        mask = al.Mask(
            array_2d=np.array(
                [
                    [True, False, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        scaled_array_zoomed = scaled_array.new_scaled_array_zoomed_from_mask(mask=mask, buffer=0)
        assert (
            scaled_array_zoomed.in_2d == np.array([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]])
        ).all()

        mask = al.Mask(
            array_2d=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        scaled_array_zoomed = scaled_array.new_scaled_array_zoomed_from_mask(mask=mask, buffer=1)
        assert (
            scaled_array_zoomed.in_2d
            == np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            )
        ).all()


class TestNewScaledArrayBinnedUp:
    def test__compare_all_extract_methods_to_array_util(self):
        array_2d = np.array(
            [
                [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=array_2d, pixel_scale=0.1)

        scaled_array_binned_util = al.binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
            array_2d=array_2d, bin_up_factor=4
        )
        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=4, method="mean"
        )
        assert (scaled_array_binned.in_2d == scaled_array_binned_util).all()

        scaled_array_binned_util = al.binning_util.binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(
            array_2d=array_2d, bin_up_factor=4
        )
        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=4, method="quadrature"
        )
        assert (scaled_array_binned.in_2d == scaled_array_binned_util).all()

        scaled_array_binned_util = al.binning_util.binned_array_2d_using_sum_from_array_2d_and_bin_up_factor(
            array_2d=array_2d, bin_up_factor=4
        )
        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=4, method="sum"
        )
        assert (scaled_array_binned.in_2d == scaled_array_binned_util).all()

    def test__pixel_scale_of_scaled_arrays_are_updated(self):
        scaled_array = np.array(
            [
                [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )

        scaled_array = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=scaled_array, pixel_scale=0.1)

        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=4, method="mean"
        )
        assert scaled_array_binned.geometry.pixel_scale == pytest.approx(0.4, 1.0e-4)
        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=6, method="mean"
        )
        assert scaled_array_binned.geometry.pixel_scale == pytest.approx(0.6, 1.0e-4)

        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=4, method="quadrature"
        )
        assert scaled_array_binned.geometry.pixel_scale == pytest.approx(0.4, 1.0e-4)
        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=6, method="quadrature"
        )
        assert scaled_array_binned.geometry.pixel_scale == pytest.approx(0.6, 1.0e-4)

        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=4, method="sum"
        )
        assert scaled_array_binned.geometry.pixel_scale == pytest.approx(0.4, 1.0e-4)
        scaled_array_binned = scaled_array.new_scaled_array_binned_from_bin_up_factor(
            bin_up_factor=6, method="sum"
        )
        assert scaled_array_binned.geometry.pixel_scale == pytest.approx(0.6, 1.0e-4)

    def test__invalid_method__raises_exception(self):
        array_2d = np.array(
            [
                [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )

        array_2d = al.ScaledArray.from_array_2d_and_pixel_scale(array_2d=array_2d, pixel_scale=0.1)
        with pytest.raises(exc.DataException):
            array_2d.new_scaled_array_binned_from_bin_up_factor(bin_up_factor=4, method="wrong")
