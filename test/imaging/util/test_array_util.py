import os
import numpy as np
import pytest

from autolens.imaging.util import array_util

test_data_dir = "{}/../../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))

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

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 3))

        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__trim__from_7x7_to_4x4(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(4, 4))


        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0]])).all()

    def test__trim__from_6x6_to_4x4(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(4, 4))

        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                      [1.0, 2.0, 2.0, 1.0],
                                      [1.0, 2.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0]])).all()

    def test__trim__from_6x6_to_3x3(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 3))

        assert (modified == np.array([[2.0, 2.0, 1.0],
                                      [2.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__trim__from_5x4_to_3x2(self):
        array = np.ones((5, 4))
        array[2, 1:3] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 2))

        assert (modified == np.array([[1.0, 1.0],
                                      [2.0, 2.0],
                                      [1.0, 1.0]])).all()

    def test__trim__from_4x5_to_2x3(self):
        array = np.ones((4, 5))
        array[1:3, 2] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(2, 3))

        assert (modified == np.array([[1.0, 2.0, 1.0],
                                      [1.0, 2.0, 1.0]])).all()

    def test__trim_with_new_origin_as_input(self):

        array = np.ones((7, 7))
        array[4, 4] = 2.0
        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 3), origin=(4, 4))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((6, 6))
        array[3, 4] = 2.0
        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 3), origin=(3, 4))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((9, 8))
        array[4, 3] = 2.0
        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 3), origin=(4, 3))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((8, 9))
        array[3, 5] = 2.0
        modified = array_util.resize_array_2d(array_2d=array, new_shape=(3, 3), origin=(3, 5))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__pad__from_3x3_to_5x5(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(5, 5))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_3x3_to_4x4(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(4, 4))


        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0],
                                      [0.0, 1.0, 2.0, 1.0],
                                      [0.0, 1.0, 1.0, 1.0]])).all()

    def test__pad__from_4x4_to_6x6(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(6, 6))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],])).all()

    def test__pad__from_4x4_to_5x5(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(5, 5))

        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 0.0],
                                      [1.0, 2.0, 2.0, 1.0, 0.0],
                                      [1.0, 2.0, 2.0, 1.0, 0.0],
                                      [1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_3x2_to_5x4(self):
        array = np.ones((3, 2))
        array[1, 0:2] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(5, 4))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [0.0, 2.0, 2.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_2x3_to_4x5(self):
        array = np.ones((2, 3))
        array[0:2, 1] = 2.0

        modified = array_util.resize_array_2d(array_2d=array, new_shape=(4, 5))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],])).all()

    def test__pad__with_input_new_origin(self):

        array = np.ones((3, 3))
        array[2, 2] = 2.0
        modified = array_util.resize_array_2d(array_2d=array, new_shape=(5, 5), origin=(2, 2))

        assert (modified == np.array([[1.0, 1.0, 1.0, 0.0, 0.0],
                                      [1.0, 1.0, 1.0, 0.0, 0.0],
                                      [1.0, 1.0, 2.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        array = np.ones((2, 3))
        array[0, 0] = 2.0
        modified = array_util.resize_array_2d(array_2d=array, new_shape=(4, 5), origin=(0, 1))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 2.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0]])).all()


class TestFits:

    def test__numpy_array_from_fits__3x3_all_ones(self):
        arr = array_util.numpy_array_from_fits(path=test_data_dir + '3x3_ones.fits', hdu=0)

        assert (arr == np.ones((3, 3))).all()

    def test__numpy_array_from_fits__4x3_all_ones(self):
        arr = array_util.numpy_array_from_fits(path=test_data_dir + '4x3_ones.fits', hdu=0)

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_to_fits__output_and_load(self):
        if os.path.exists(test_data_dir + 'test.fits'):
            os.remove(test_data_dir + 'test.fits')

        arr = np.array([[10., 30., 40.],
                        [92., 19., 20.]])

        array_util.numpy_array_to_fits(arr, path=test_data_dir + 'test.fits')

        array_load = array_util.numpy_array_from_fits(path=test_data_dir + 'test.fits', hdu=0)

        assert (arr == array_load).all()


class TestVariancesFromNoise:

    def test__noise_all_1s__variances_all_1s(self):
        noise = np.array([[1.0, 1.0],
                          [1.0, 1.0]])

        assert (array_util.compute_variances_from_noise(noise) == np.array([[1.0, 1.0],
                                                                      [1.0, 1.0]])).all()

    def test__noise_all_2s__variances_all_4s(self):
        noise = np.array([[2.0, 2.0],
                          [2.0, 2.0]])

        assert (array_util.compute_variances_from_noise(noise) == np.array([[4.0, 4.0],
                                                                      [4.0, 4.0]])).all()

    def test__noise_all_05s__variances_all_025s(self):
        noise = np.array([[0.5, 0.5],
                          [0.5, 0.5]])

        assert (array_util.compute_variances_from_noise(noise) == np.array([[0.25, 0.25],
                                                                      [0.25, 0.25]])).all()
