import os
import shutil
import numpy as np
import pytest

from autolens.data.array.util import array_util

test_data_path = "{}/../../../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(name="memoizer")
def make_memoizer():
    return array_util.Memoizer()


class TestMemoizer(object):
    def test_storing(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(2)
        func(1)

        assert memoizer.results == {"('arg', 1)": "result for 1", "('arg', 2)": "result for 2"}
        assert memoizer.calls == 2

    def test_multiple_arguments(self, memoizer):
        @memoizer
        def func(arg1, arg2):
            return arg1 * arg2

        func(1, 2)
        func(2, 1)
        func(1, 2)

        assert memoizer.results == {"('arg1', 1), ('arg2', 2)": 2, "('arg1', 2), ('arg2', 1)": 2}
        assert memoizer.calls == 2

    def test_key_word_arguments(self, memoizer):
        @memoizer
        def func(arg1=0, arg2=0):
            return arg1 * arg2

        func(arg1=1)
        func(arg2=1)
        func(arg1=1)
        func(arg1=1, arg2=1)

        assert memoizer.results == {"('arg1', 1)": 0, "('arg2', 1)": 0, "('arg1', 1), ('arg2', 1)": 1}
        assert memoizer.calls == 3

    def test_key_word_for_positional(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(arg=2)
        func(arg=1)

        assert memoizer.calls == 2

    def test_methods(self, memoizer):
        class Class(object):
            def __init__(self, value):
                self.value = value

            @memoizer
            def method(self):
                return self.value

        one = Class(1)
        two = Class(2)

        assert one.method() == 1
        assert two.method() == 2


class TestResize:

    def test__trim__from_7x7_to_3x3(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3))

        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__trim__from_7x7_to_4x4(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(4, 4))


        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0]])).all()

    def test__trim__from_6x6_to_4x4(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(4, 4))

        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                      [1.0, 2.0, 2.0, 1.0],
                                      [1.0, 2.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0]])).all()

    def test__trim__from_6x6_to_3x3(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3))

        assert (modified == np.array([[2.0, 2.0, 1.0],
                                      [2.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__trim__from_5x4_to_3x2(self):
        array = np.ones((5, 4))
        array[2, 1:3] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 2))

        assert (modified == np.array([[1.0, 1.0],
                                      [2.0, 2.0],
                                      [1.0, 1.0]])).all()

    def test__trim__from_4x5_to_2x3(self):
        array = np.ones((4, 5))
        array[1:3, 2] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(2, 3))

        assert (modified == np.array([[1.0, 2.0, 1.0],
                                      [1.0, 2.0, 1.0]])).all()

    def test__trim_with_new_origin_as_input(self):

        array = np.ones((7, 7))
        array[4, 4] = 2.0
        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(4, 4))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((6, 6))
        array[3, 4] = 2.0
        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(3, 4))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((9, 8))
        array[4, 3] = 2.0
        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(4, 3))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((8, 9))
        array[3, 5] = 2.0
        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(3, 3), origin=(3, 5))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__pad__from_3x3_to_5x5(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(5, 5))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_3x3_to_4x4(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(4, 4))


        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0],
                                      [0.0, 1.0, 2.0, 1.0],
                                      [0.0, 1.0, 1.0, 1.0]])).all()

    def test__pad__from_4x4_to_6x6(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(6, 6))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],])).all()

    def test__pad__from_4x4_to_5x5(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(5, 5))

        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 0.0],
                                      [1.0, 2.0, 2.0, 1.0, 0.0],
                                      [1.0, 2.0, 2.0, 1.0, 0.0],
                                      [1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_3x2_to_5x4(self):
        array = np.ones((3, 2))
        array[1, 0:2] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(5, 4))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [0.0, 2.0, 2.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_2x3_to_4x5(self):
        array = np.ones((2, 3))
        array[0:2, 1] = 2.0

        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(4, 5))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],])).all()

    def test__pad__with_input_new_origin(self):

        array = np.ones((3, 3))
        array[2, 2] = 2.0
        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(5, 5), origin=(2, 2))

        assert (modified == np.array([[1.0, 1.0, 1.0, 0.0, 0.0],
                                      [1.0, 1.0, 1.0, 0.0, 0.0],
                                      [1.0, 1.0, 2.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        array = np.ones((2, 3))
        array[0, 0] = 2.0
        modified = array_util.resized_array_2d_from_array_2d_and_resized_shape(array_2d=array, resized_shape=(4, 5), origin=(0, 1))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 2.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0]])).all()


class TestFits:

    def test__numpy_array_from_fits__3x3_all_ones(self):
        arr = array_util.numpy_array_2d_from_fits(file_path=test_data_path + '3x3_ones.fits', hdu=0)

        assert (arr == np.ones((3, 3))).all()

    def test__numpy_array_from_fits__4x3_all_ones(self):
        arr = array_util.numpy_array_2d_from_fits(file_path=test_data_path + '4x3_ones.fits', hdu=0)

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_to_fits__output_and_load(self):
        if os.path.exists(test_data_path + 'test.fits'):
            os.remove(test_data_path + 'test.fits')

        arr = np.array([[10., 30., 40.],
                        [92., 19., 20.]])

        array_util.numpy_array_2d_to_fits(arr, file_path=test_data_path + 'test.fits')

        array_load = array_util.numpy_array_2d_from_fits(file_path=test_data_path + 'test.fits', hdu=0)

        assert (arr == array_load).all()


class TestReplaceNegativeNoise:

    def test__2x2_array__no_negative_values__no_change(self):

        image_2d = np.ones(shape=(2,2))

        noise_map_2d = np.array([[1.0, 2.0],
                                 [3.0, 4.0]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0)

        assert (noise_map_2d == noise_map_2d).all()

    def test__2x2_array__negative_values__do_not_produce_absolute_signal_to_noise_values_above_target__no_change(self):

        image_2d = -1.0*np.ones(shape=(2,2))

        noise_map_2d = np.array([[1.0, 0.5],
                                 [0.25, 0.125]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=10.0)

        assert (noise_map_2d == noise_map_2d).all()

    def test__2x2_array__negative_values__values_give_absolute_signal_to_noise_below_target__replaces_their_noise(self):

        image_2d = -1.0*np.ones(shape=(2,2))

        noise_map_2d = np.array([[1.0, 0.5],
                                 [0.25, 0.125]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=4.0)

        assert (noise_map_2d == np.array([[1.0, 0.5],
                                          [0.25, 0.25]])).all()

        noise_map_2d = np.array([[1.0, 0.5],
                                 [0.25, 0.125]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0)

        assert (noise_map_2d == np.array([[1.0, 0.5],
                                          [0.5, 0.5]])).all()


        noise_map_2d = np.array([[1.0, 0.5],
                                 [0.25, 0.125]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0)

        assert (noise_map_2d == np.array([[1.0, 1.0],
                                          [1.0, 1.0]])).all()


        noise_map_2d = np.array([[1.0, 0.5],
                                 [0.25, 0.125]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=0.5)

        assert (noise_map_2d == np.array([[2.0, 2.0],
                                          [2.0, 2.0]])).all()

    def test__same_as_above__image_not_all_negative_ones(self):

        image_2d = np.array([[1.0, -2.0],
                             [5.0, -4.0]])

        noise_map_2d = np.array([[3.0, 1.0],
                                 [4.0, 8.0]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0)

        assert (noise_map_2d == np.array([[3.0, 2.0],
                                          [4.0, 8.0]])).all()

        image_2d = np.array([[-10.0, -20.0],
                             [100.0, -30.0]])

        noise_map_2d = np.array([[1.0, 2.0],
                                 [40.0, 3.0]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=5.0)

        assert (noise_map_2d == np.array([[2.0, 4.0],
                                          [40.0, 6.0]])).all()

    def test__rectangular_2x3_and_3x2_arrays(self):

        image_2d = -1.0*np.ones(shape=(2,3))

        noise_map_2d = np.array([[1.0, 0.5, 0.25],
                                 [0.25, 0.125, 2.0]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0)

        assert (noise_map_2d == np.array([[1.0, 0.5, 0.5],
                                          [0.5, 0.5, 2.0]])).all()

        image_2d = -1.0*np.ones(shape=(3,2))

        noise_map_2d = np.array([[1.0, 0.5],
                                 [0.25, 0.125],
                                 [0.25, 2.0]])

        noise_map_2d = array_util.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0)

        assert (noise_map_2d == np.array([[1.0, 0.5],
                                          [0.5, 0.5],
                                          [0.5, 2.0]])).all()


class TestBinUpPadding:

    def test__bin_up_factor_is_1__array_2d_does_not_change_shape(self):

        array_2d = np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]])

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=1)

        assert (array_2d_padded == array_2d).all()

    def test__bin_up_factor_gives_no_remainder__array_2d_does_not_change_shape(self):

        array_2d = np.ones(shape=(6, 6))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (6, 6)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=3)
        assert array_2d_padded.shape == (6, 6)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=6)
        assert array_2d_padded.shape == (6, 6)

        array_2d = np.ones(shape=(8, 8))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (8, 8)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=4)
        assert array_2d_padded.shape == (8, 8)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=8)
        assert array_2d_padded.shape == (8, 8)

        array_2d = np.ones(shape=(9, 9))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=3)
        assert array_2d_padded.shape == (9, 9)

        array_2d = np.ones(shape=(16, 16))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (16, 16)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=4)
        assert array_2d_padded.shape == (16, 16)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=8)
        assert array_2d_padded.shape == (16, 16)

        array_2d = np.ones(shape=(12, 16))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (12, 16)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=4)
        assert array_2d_padded.shape == (12, 16)

        array_2d = np.ones(shape=(16, 12))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (16, 12)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=4)
        assert array_2d_padded.shape == (16, 12)

    def test__bin_up_factor_gives_remainder__array_2d_padded_to_give_no_remainder(self):

        array_2d = np.ones(shape=(6, 6))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=4)
        assert array_2d_padded.shape == (8, 8)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=5)
        assert array_2d_padded.shape == (10, 10)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=7)
        assert array_2d_padded.shape == (7, 7)

        array_2d = np.ones(shape=(10, 10))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=3)
        assert array_2d_padded.shape == (12, 12)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=4)
        assert array_2d_padded.shape == (12, 12)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=6)
        assert array_2d_padded.shape == (12, 12)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=7)
        assert array_2d_padded.shape == (14, 14)

        array_2d = np.ones(shape=(7, 10))

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=3)
        assert array_2d_padded.shape == (9, 12)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=5)
        assert array_2d_padded.shape == (10, 10)

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=7)
        assert array_2d_padded.shape == (7, 14)

    def test__padding_using_arrays_and_not_shapes(self):

        array_2d = np.ones(shape=(4, 4))
        array_2d[1,1] = 2.0

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=3)
        assert array_2d_padded.shape == (6, 6)
        assert (array_2d_padded == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                             [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
                                             [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                             [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=5)
        assert array_2d_padded.shape == (5, 5)
        assert (array_2d_padded == np.array([[1.0, 1.0, 1.0, 1.0, 0.0],
                                             [1.0, 2.0, 1.0, 1.0, 0.0],
                                             [1.0, 1.0, 1.0, 1.0, 0.0],
                                             [1.0, 1.0, 1.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        array_2d = np.ones(shape=(2, 3))
        array_2d[1,1] = 2.0
        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (2, 4)
        assert (array_2d_padded == np.array([[0.0, 1.0, 1.0, 1.0],
                                             [0.0, 1.0, 2.0, 1.0]])).all()

        array_2d = np.ones(shape=(3, 2))
        array_2d[1,1] = 2.0
        array_2d_padded = array_util.pad_2d_array_for_binning_up_with_bin_up_factor(array_2d=array_2d, bin_up_factor=2)
        assert array_2d_padded.shape == (4, 2)
        assert (array_2d_padded == np.array([[0.0, 0.0],
                                             [1.0, 1.0],
                                             [1.0, 2.0],
                                             [1.0, 1.0]])).all()


class TestBinUpArrays2d:

    def test__bin_using_mean__array_4x4_to_2x2__uses_mean_correctly(self):

        array_2d = np.array([[1.0, 1.0, 2.0, 2.0],
                             [1.0, 1.0, 2.0, 2.0],
                             [3.0, 3.0, 4.0, 4.0],
                             [3.0, 3.0, 4.0, 4.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=2)

        assert (binned_array_2d == np.array([[1.0, 2.0],
                                             [3.0, 4.0]])).all()

        array_2d = np.array([[1.0, 2.0, 2.0, 2.0],
                             [1.0, 6.0, 2.0, 10.0],
                             [9.0, 3.0, 4.0, 0.0],
                             [3.0, 3.0, 4.0, 4.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=2)

        assert (binned_array_2d == np.array([[2.5, 4.0],
                                             [4.5, 3.0]])).all()

    def test__bin_using_mean__array_6x3_to_2x1_and_3x6_to_1x2__uses_mean_correctly(self):

        array_2d = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=3)

        assert (binned_array_2d == np.array([[1.0],
                                             [2.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0],
                             [1.0, 10.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [2.0, 11.0, 2.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[2.0],
                                             [3.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[1.0, 2.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 10.0, 1.0, 11.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[2.0, 3.0]])).all()

    def test__bin_using_mean__bin_includes_padding_image_with_zeros(self):

        # Padded array:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        array_2d = np.ones(shape=(4, 4))
        array_2d[1,1] = 2.0
        binned_array_2d = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[(5.0 / 9.0), (4.0 / 9.0)],
                                             [(4.0 / 9.0), (4.0 / 9.0)]])).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        array_2d = np.ones(shape=(2, 3))
        array_2d[1,1] = 2.0
        binned_2d_array = array_util.bin_up_array_2d_using_mean(array_2d=array_2d, bin_up_factor=2)
        assert (binned_2d_array == np.array([[0.5, 1.25]])).all()
        
    def test__bin_using_quadrature__array_4x4_to_2x2__uses_quadrature_correctly(self):

        array_2d = np.array([[1.0, 1.0, 2.0, 2.0],
                             [1.0, 1.0, 2.0, 2.0],
                             [3.0, 3.0, 4.0, 4.0],
                             [3.0, 3.0, 4.0, 4.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=2)

        assert (binned_array_2d == np.array([[np.sqrt(4.0) / 4.0, np.sqrt(16.0) / 4.0],
                                             [np.sqrt(36.0) / 4.0, np.sqrt(64.0) / 4.0]])).all()

        array_2d = np.array([[1.0, 2.0, 2.0, 2.0],
                             [1.0, 6.0, 2.0, 10.0],
                             [9.0, 3.0, 4.0, 0.0],
                             [3.0, 3.0, 4.0, 4.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=2)

        assert (binned_array_2d == np.array([[np.sqrt(42.0) / 4.0, np.sqrt(112.0) / 4.0],
                                             [np.sqrt(108.0) / 4.0, np.sqrt(48.0) / 4.0]])).all()

    def test__bin_using_quadrature__array_6x3_to_2x1_and_3x6_to_1x2__uses_quadrature_correctly(self):

        array_2d = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=3)

        assert (binned_array_2d == np.array([[np.sqrt(9.0) / 9.0],
                                             [np.sqrt(36.0) / 9.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0],
                             [1.0, 10.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [2.0, 4.0, 2.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[np.sqrt(108.0) / 9.0],
                                             [np.sqrt(48.0) / 9.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[np.sqrt(9.0) / 9.0, np.sqrt(36.0) / 9.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 10.0, 1.0, 4.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[np.sqrt(108.0) / 9.0, np.sqrt(48.0) / 9.0]])).all()

    def test__bin_using_quadrature__bin_includes_padding_image_with_zeros(self):

        # Padded array:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        array_2d = np.ones(shape=(4, 4))
        array_2d[1,1] = 2.0
        binned_array_2d = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[np.sqrt(7.0) / 9.0, np.sqrt(4.0) / 9.0],
                                             [np.sqrt(4.0) / 9.0, np.sqrt(4.0) / 9.0]])).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        array_2d = np.ones(shape=(2, 3))
        array_2d[1,1] = 2.0
        binned_2d_array = array_util.bin_up_array_2d_using_quadrature(array_2d=array_2d, bin_up_factor=2)
        assert (binned_2d_array == np.array([[np.sqrt(2.0) / 4.0, np.sqrt(7.0) / 4.0]])).all()
        
    def test__bin_using_sum__array_4x4_to_2x2__uses_sum_correctly(self):

        array_2d = np.array([[1.0, 1.0, 2.0, 2.0],
                             [1.0, 1.0, 2.0, 2.0],
                             [3.0, 3.0, 4.0, 4.0],
                             [3.0, 3.0, 4.0, 4.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=2)

        assert (binned_array_2d == np.array([[4.0, 8.0],
                                             [12.0, 16.0]])).all()

        array_2d = np.array([[1.0, 2.0, 2.0, 2.0],
                             [1.0, 6.0, 2.0, 10.0],
                             [9.0, 3.0, 4.0, 0.0],
                             [3.0, 3.0, 4.0, 4.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=2)

        assert (binned_array_2d == np.array([[10.0, 16.0],
                                             [18.0, 12.0]])).all()

    def test__bin_using_sum__array_6x3_to_2x1_and_3x6_to_1x2__uses_sum_correctly(self):

        array_2d = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=3)

        assert (binned_array_2d == np.array([[9.0],
                                             [18.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0],
                             [1.0, 10.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [2.0, 11.0, 2.0],
                             [2.0, 2.0, 2.0],
                             [2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[18.0],
                                             [27.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[9.0, 18.0]])).all()

        array_2d = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                             [1.0, 10.0, 1.0, 11.0, 2.0, 2.0],
                             [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[18.0, 27.0]])).all()

    def test__bin_using_sum__bin_includes_padding_image_with_zeros(self):

        # Padded array:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        array_2d = np.ones(shape=(4, 4))
        array_2d[1,1] = 2.0
        binned_array_2d = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=3)
        assert (binned_array_2d == np.array([[5.0, 4.0],
                                             [4.0, 4.0]])).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        array_2d = np.ones(shape=(2, 3))
        array_2d[1,1] = 2.0
        binned_2d_array = array_util.bin_up_array_2d_using_sum(array_2d=array_2d, bin_up_factor=2)
        assert (binned_2d_array == np.array([[2.0, 5.0]])).all()
