from autolens.imaging import array_util
import numpy as np
import os

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFits:

    def test__numpy_array_from_fits__3x3_all_ones(self):
        arr = array_util.numpy_array_from_fits(file_path=test_data_dir + '3x3_ones', hdu=0)

        assert (arr == np.ones((3, 3))).all()

    def test__numpy_array_from_fits__4x3_all_ones(self):
        arr = array_util.numpy_array_from_fits(file_path=test_data_dir + '4x3_ones', hdu=0)

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_to_fits__output_and_load(self):
        if os.path.exists(test_data_dir + 'test.fits'):
            os.remove(test_data_dir + 'test.fits')

        arr = np.array([[10., 30., 40.],
                        [92., 19., 20.]])

        array_util.numpy_array_to_fits(arr, file_path=test_data_dir + 'test')

        array_load = array_util.numpy_array_from_fits(file_path=test_data_dir + 'test', hdu=0)

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
