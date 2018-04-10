from auto_lens.tools import image
import numpy as np
import pytest

class TestComputeResiduals:

    def test__model_matches_data__residuals_are_all_zero(self):

        data = np.array([[10, 10],
                         [10, 10]])

        model = np.array([[10, 10],
                          [10, 10]])

        result = image.compute_residuals(data, model)

        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[1, 0] == 0
        assert result[1, 1] == 0

    def test__model_does_not_match_data__residuals_are_non_zero(self):

        test_image = np.array([[10, 5],
                               [-2, -4.5]])

        model = np.array([[10, 10],
                          [10, -5]])

        result = image.compute_residuals(test_image, model)

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

        result = image.compute_chi_sq_image(data, model, noise)

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

        result = image.compute_chi_sq_image(test_image, model, noise)

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
        mask = np.array([[False, False],
                        [False, False]])

        result = image.compute_likelihood(data, model, noise, mask)

        assert result == 0

    def test___model_does_not_match_data__likelihood_computed_correctly(self):

        test_image = np.array([[10, 5],
                               [-2, -4.5]])

        model = np.array([[10, 10],
                          [10, -5]])

        noise = np.array([[1, 5],
                          [-1, -2]])

        mask = np.array([[False, False],
                        [False, False]])

        result = image.compute_likelihood(test_image, model, noise, mask)

        assert result == -72.53125  # -0.5*(0 + 1 + 144 + 0.0625)


class TestEstimatePoissonNoiseFromImage:

    def test__image_and_exposure_times_all_1s__noise_is_all_1s(self):

        test_image = np.ones((3, 3))

        exposure_time_map = np.ones((3, 3))

        poisson_noise_estimate = image.estimate_poisson_noise_from_image(test_image, exposure_time_map)

        assert (poisson_noise_estimate == np.ones((3, 3))).all()

    def test__image_all_4s__exposure_time_all_1s__noise_is_all_2s(self):

        test_image = 4.0 * np.ones((4, 2))

        exposure_time_map = np.ones((4, 2))

        poisson_noise_estimate = image.estimate_poisson_noise_from_image(test_image, exposure_time_map)

        assert (poisson_noise_estimate == 2.0 * np.ones((4, 2))).all()

    def test__image_all_1s__exposure_time_all_4s__noise_is_all_2_divided_4_so_halves(self):

        test_image = np.ones((1, 5))

        exposure_time_map = 4.0 * np.ones((1, 5))

        poisson_noise_estimate = image.estimate_poisson_noise_from_image(test_image, exposure_time_map)

        assert (poisson_noise_estimate == 0.5 * np.ones((1, 5))).all()

    def test__image_and_exposure_times_range_of_values__noises_estimates_correct(self):

        test_image = np.array([[5.0, 3.0],
                               [10.0, 20.0]])

        exposure_time_map = np.array([[1.0, 2.0],
                                      [3.0, 4.0]])

        poisson_noise_estimate = image.estimate_poisson_noise_from_image(test_image, exposure_time_map)

        assert (poisson_noise_estimate == np.array([[np.sqrt(5.0),     np.sqrt(6.0)/2.0],
                                                   [np.sqrt(30.0)/3.0, np.sqrt(80.0)/4.0]])).all()


class TestEstimatePoissonNoiseFromImageAndBackground:

    def test__image_and_exposure_times_all_1s__no_background__noise_is_all_1s(self):

        test_image = np.ones((3, 3))

        exposure_time_map = np.ones((3, 3))

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=0.0, exposure_time_mean=1.0)

        assert (poisson_noise_estimate == np.ones((3, 3))).all()

    def test__image_and_exposure_times_all_1s__background_is_sqrt_3__noise_is_all_2s(self):

        # noise is going to be sqrt(3**2.0 + 1) in every pixel, so 2.0

        test_image = np.ones((3, 3))

        exposure_time_map = np.ones((3, 3))

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                sigma_background=3.0 ** 0.5, exposure_time_mean=1.0)

        assert poisson_noise_estimate == pytest.approx(2.0 * np.ones((3, 3)), 1e-2)

    def test__image_and_exposure_times_all_1s__background_sigma_is_5_for_1_second_exposure__noise_all_correct(self):

        test_image = np.ones((2, 3))

        exposure_time_map = np.ones((2, 3))

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=5.0, exposure_time_mean=1.0)

        assert poisson_noise_estimate == \
               pytest.approx(np.array([[np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0)],
                                       [np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0)]]), 1e-2)

    def test__image_different_values__exposure_times_all_1s__background_is_1_for_5_seconds__noise_all_correct(self):

        test_image = np.ones((2, 3))

        exposure_time_map = np.ones((2, 3))

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=1.0, exposure_time_mean=5.0)

        assert poisson_noise_estimate == \
               pytest.approx(np.array([[np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0)],
                                       [np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0), np.sqrt(25.0 + 1.0)]]), 1e-2)

    def test__same_as_above_but_different_image_values_in_each_pixel_and_new_background_values(self):

        test_image = np.array([[1.0, 2.0],
                               [3.0, 4.0],
                               [5.0, 6.0]])

        exposure_time_map = np.ones((3, 2))

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=4.0, exposure_time_mean=3.0)

        assert poisson_noise_estimate == \
               pytest.approx(np.array([[np.sqrt(144.0 + 1.0), np.sqrt(144.0 + 2.0)],
                                       [np.sqrt(144.0 + 3.0), np.sqrt(144.0 + 4.0)],
                                       [np.sqrt(144.0 + 5.0), np.sqrt(144.0 + 6.0)]]), 1e-2)

    def test__same_as_above_but_image_values_all_1s_exposure_times_change_instead__noise_is_in_electrons_per_sec(self):

        test_image = np.ones((3, 2))

        exposure_time_map =  np.array([[1.0, 2.0],
                                       [3.0, 4.0],
                                       [5.0, 6.0]])

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=4.0, exposure_time_mean=3.0)

        assert poisson_noise_estimate == \
               pytest.approx(np.array([[np.sqrt(144.0 + 1.0)/1.0, np.sqrt(144.0 + 2.0)/2.0],
                                       [np.sqrt(144.0 + 3.0)/3.0, np.sqrt(144.0 + 4.0)/4.0],
                                       [np.sqrt(144.0 + 5.0)/5.0, np.sqrt(144.0 + 6.0)/6.0]]), 1e-2)

    def test__image_and_exposure_times_range_of_values__no_bacground__noise_estimates_correct(self):

        test_image = np.array([[5.0, 3.0],
                               [10.0, 20.0]])

        exposure_time_map = np.array([[1.0, 2.0],
                                      [3.0, 4.0]])

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=0.0, exposure_time_mean=4.0)

        assert (poisson_noise_estimate == np.array([[np.sqrt(5.0),     np.sqrt(6.0)/2.0],
                                                   [np.sqrt(30.0)/3.0, np.sqrt(80.0)/4.0]])).all()

    def test__image_and_exposure_times_range_of_values__background_has_value___noise_estimates_correct(self):

        test_image = np.array([[5.0, 3.0],
                               [10.0, 20.0]])

        exposure_time_map = np.array([[1.0, 2.0],
                                      [3.0, 4.0]])

        poisson_noise_estimate = image.estimate_noise_from_image_and_background(test_image, exposure_time_map,
                                                                    sigma_background=3.0, exposure_time_mean=3.0)

        assert (poisson_noise_estimate == np.array([[np.sqrt(81.0 + 5.0),     np.sqrt(81.0 + 6.0)/2.0],
                                                   [np.sqrt(81.0 + 30.0)/3.0, np.sqrt(81.0 + 80.0)/4.0]])).all()


class TestGenerateGaussianNoiseMap:

    def test__input_mean_is_0__mean_of_image_values_consistent_with_0(self):

        gaussian_noise_map = image.generate_gaussian_noise_map(dimensions=(5, 5), mean=0.0, sigma=1.0,
                                                               seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert -0.1 <= np.mean(gaussian_noise_map) <= 0.1

    def test__input_mean_is_10__mean_of_image_values_consistent_with_10(self):
        gaussian_noise_map = image.generate_gaussian_noise_map(dimensions=(5, 5), mean=10.0,
                                                               sigma=1.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 9.9 <= np.mean(gaussian_noise_map) <= 10.1

    def test__input_sigma_is_1__standard_deviation_of_image_values_consistent_with_1(self):
        gaussian_noise_map = image.generate_gaussian_noise_map(dimensions=(5, 5), mean=10.0,
                                                               sigma=1.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 0.9 <= np.std(gaussian_noise_map) <= 1.1

    def test__input_sigma_is_10__standard_deviation_of_image_values_consistent_with_10(self):
        gaussian_noise_map = image.generate_gaussian_noise_map(dimensions=(5, 5), mean=100.0,
                                                               sigma=10.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 9.0 <= np.std(gaussian_noise_map) <= 11.0

    def test__known_noise_map_for_mean_0_sigma_1_seed_1(self):
        gaussian_noise_map = image.generate_gaussian_noise_map(dimensions=(5, 5), mean=0.0, sigma=1.0,
                                                               seed=1)

        assert gaussian_noise_map == pytest.approx(
            np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
                      [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
                      [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944],
                      [-1.09989127, -0.17242821, -0.87785842, 0.04221375, 0.58281521],
                      [-1.10061918, 1.14472371, 0.90159072, 0.50249434, 0.90085595]]), 1e-2)


class TestGeneratePoissonNoiseMap:

    def test__input_image_all_0s__exposure_times_all_1s__all_noise_values_are_zeros(self):

        test_image = np.zeros((2,2))

        exposure_time_map = np.ones((2, 2))

        poisson_noise_map = image.generate_poisson_noise_map(test_image, exposure_time_map, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.zeros((2, 2))).all()

    def test__input_image_has_10s__exposure_times_all_1s__gives_noise_values_near_1_to_5(self):

        test_image = np.array([[10., 0.],
                               [0., 10.]])

        exposure_time_map = np.ones((2, 2))

        poisson_noise_map = image.generate_poisson_noise_map(test_image, exposure_time_map, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[1, 0],  # Use known noise map for given seed.
                                               [0, 4]])).all()

    def test__input_image_has_1000000s__exposure_times_all_1s__these_give_positive_noise_values_near_1000(self):

        test_image = np.array([[10000000., 0.],
                               [0., 10000000.]])

        exposure_time_map = np.ones((2, 2))

        poisson_noise_map = image.generate_poisson_noise_map(test_image, exposure_time_map, seed=2)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[571, 0],  # Use known noise map for given seed.
                                               [0, -441]])).all()

    def test__two_images__same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(self):

        test_image_0 = np.array([[10., 0.],
                                 [0., 10.]])

        exposure_time_map_0 = np.ones((2, 2))

        test_image_1 = np.array([[5., 0.],
                                 [0., 5.]])

        exposure_time_map_1 = 2.0 * np.ones((2, 2))

        poisson_noise_map_0 = image.generate_poisson_noise_map(test_image_0, exposure_time_map_0, seed=1)
        poisson_noise_map_1 = image.generate_poisson_noise_map(test_image_1, exposure_time_map_1, seed=1)

        assert (poisson_noise_map_0/2.0 == poisson_noise_map_1 ).all()

    def test__same_as_above_but_range_of_image_values_and_exposure_times(self):

        test_image_0 = np.array([[10., 20.],
                                 [30., 40.]])

        exposure_time_map_0 =  np.array([[2., 2.],
                                         [3., 4.]])

        test_image_1 = np.array([[20., 20.],
                                 [45., 20.]])

        exposure_time_map_1 =  np.array([[1., 2.],
                                         [2., 8.]])

        poisson_noise_map_0 = image.generate_poisson_noise_map(test_image_0, exposure_time_map_0, seed=1)
        poisson_noise_map_1 = image.generate_poisson_noise_map(test_image_1, exposure_time_map_1, seed=1)

        assert (poisson_noise_map_0[0,0] == poisson_noise_map_1[0,0]/2.0 ).all()
        assert (poisson_noise_map_0[0,1] == poisson_noise_map_1[0,1]).all()
        assert (poisson_noise_map_0[1,0]*1.5 == pytest.approx(poisson_noise_map_1[1,0], 1e-2) ).all()
        assert (poisson_noise_map_0[1,1]/2.0 == poisson_noise_map_1[1,1] ).all()


class TestConvolveImageWithKernal:

    def test__image_is_central_value_of_one__kernel_is_cross__both_3x3(self):
        test_image = np.array([[0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 1.0, 0.0],
                           [1.0, 2.0, 1.0],
                           [0.0, 1.0, 0.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == kernel).all()

    def test__image_is_central_value_of_one__kernel_is_not_odd_x_odd__raises_error(self):
        test_image = np.array([[0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 1.0],
                           [1.0, 2.0]])

        with pytest.raises(image.KernelException):
            image.convolve_image_with_kernel(test_image, kernel)

    def test__image_is_central_value_of_one__kernel_is_cross__image_4x4_kernel_3x3(self):
        test_image = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 1.0, 0.0],
                           [1.0, 2.0, 1.0],
                           [0.0, 1.0, 0.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == np.array([[0.0, 1.0, 0.0, 0.0],
                                           [1.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

    def test__image_is_central_value_of_one__kernel_is_cross__image_4x3_kernel_3x3(self):
        test_image = np.array([[0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 1.0, 0.0],
                           [1.0, 2.0, 1.0],
                           [0.0, 1.0, 0.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == np.array([[0.0, 1.0, 0.0],
                                           [1.0, 2.0, 1.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0]])).all()

    def test__image_is_central_value_of_one__kernel_is_cross__image_3x4_kernel_3x3(self):
        test_image = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[0.0, 1.0, 0.0],
                           [1.0, 2.0, 1.0],
                           [0.0, 1.0, 0.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == np.array([[0.0, 1.0, 0.0, 0.0],
                                           [1.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0]])).all()

    def test__image_has_two_central_values__kernel_is_asymmetric__image_follows_convolution(self):
        test_image = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[1.0, 1.0, 1.0],
                           [2.0, 2.0, 1.0],
                           [1.0, 3.0, 3.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == np.array([[1.0, 1.0, 1.0, 0.0],
                                           [2.0, 3.0, 2.0, 1.0],
                                           [1.0, 5.0, 5.0, 1.0],
                                           [0.0, 1.0, 3.0, 3.0]])).all()

    def test__image_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(self):
        test_image = np.array([[0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0, 0.0]])

        kernel = np.array([[1.0, 1.0, 1.0],
                           [2.0, 2.0, 1.0],
                           [1.0, 3.0, 3.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == np.array([[1.0, 1.0, 0.0, 0.0],
                                           [2.0, 1.0, 1.0, 1.0],
                                           [3.0, 3.0, 2.0, 2.0],
                                           [0.0, 0.0, 1.0, 3.0]])).all()

    def test__image_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(self):
        test_image = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

        kernel = np.array([[1.0, 1.0, 1.0],
                           [2.0, 2.0, 1.0],
                           [1.0, 3.0, 3.0]])

        blurred_test_image = image.convolve_image_with_kernel(test_image, kernel)

        assert (blurred_test_image == np.array([[2.0, 1.0, 0.0, 0.0],
                                           [3.0, 3.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0],
                                           [0.0, 0.0, 2.0, 2.0]])).all()
