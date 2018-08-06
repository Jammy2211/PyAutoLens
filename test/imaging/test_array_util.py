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


class TestComputeResiduals:

    def test__model_matches_data__residuals_are_all_zero(self):
        data = np.array([[10, 10],
                         [10, 10]])

        model = np.array([[10, 10],
                          [10, 10]])

        result = array_util.compute_residuals(data, model)

        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[1, 0] == 0
        assert result[1, 1] == 0

    def test__model_does_not_match_data__residuals_are_non_zero(self):
        test_image = np.array([[10, 5],
                               [-2, -4.5]])

        model = np.array([[10, 10],
                          [10, -5]])

        result = array_util.compute_residuals(test_image, model)

        assert result[0, 0] == 0  # (10 - 10 = 0)
        assert result[0, 1] == -5  # (5 - 10 = -5)
        assert result[1, 0] == -12  # (-2 - 10 = -12)
        assert result[1, 1] == 0.5  # (-4.5 - (-5)) = 0.5


class TestComputeChiSquared:

    def test__model_matches_data__chi_squareds_all_zeros(self):
        data = np.array([[10, 10],
                         [10, 10]])

        model = np.array([[10, 10],
                          [10, 10]])

        noise = np.array([[1, 1],
                          [1, 1]])

        result = array_util.compute_chi_sq_image(data, model, noise)

        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[1, 0] == 0
        assert result[1, 1] == 0

    def test__model_does_not_match_data__chi_squareds_are_non_zero(self):
        test_image = np.array([[10, 5],
                               [-2, -4.5]])

        model = np.array([[10, 10],
                          [10, -5]])

        noise = np.array([[1, 5],
                          [-1, -2]])

        result = array_util.compute_chi_sq_image(test_image, model, noise)

        assert result[0, 0] == 0  # ( (10 - 10)/1 )^2 = 0
        assert result[0, 1] == 1  # ( (5 - 10)/5 )^2 = ((-)2.5)^2
        assert result[1, 0] == 144  # ( (-2 - 10)/(-1))^2 = 12^2 = 144
        assert result[1, 1] == 0.0625  # ( (-4.5 - (-5))/-2))^2 = (0.5/-3)^2 = (1/6)^2 = 1/16


class TestComputeLikelihood:

    def test__model_matches_data__likelihood_is_zero(self):
        data = np.array([[10, 10],
                         [10, 10]])
        model = np.array([[10, 10],
                          [10, 10]])
        noise = np.array([[1, 1],
                          [1, 1]])

        result = array_util.compute_likelihood(data, model, noise)

        assert result == 0

    def test___model_does_not_match_data__likelihood_computed_correctly(self):
        test_image = np.array([[10, 5],
                               [-2, -4.5]])

        model = np.array([[10, 10],
                          [10, -5]])

        noise = np.array([[1, 5],
                          [-1, -2]])

        result = array_util.compute_likelihood(test_image, model, noise)

        assert result == -72.53125  # -0.5*(0 + 1 + 144 + 0.0625)
