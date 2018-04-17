import numpy as np
import pytest

from auto_lens.imaging import imaging, simulate


class TestConstructor(object):

    def test__setup_with_all_features_off(self):

        image = np.array(([0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]))

        sim_image = simulate.SimulateImage(data=image, pixel_scale=0.1)

        assert (sim_image.data_original == np.array(([0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0]))).all()

        assert (sim_image.data == np.array(([0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]))).all()

        assert sim_image.pixel_scale == 0.1

        assert sim_image.background_sky_level == None
        assert sim_image.psf == None
        assert sim_image.exposure_time_map == None

    def test__setup_with_background_sky_on__sky_is_added_to_image(self):

        image = np.array(([0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]))

        sim_image = simulate.SimulateImage(data=image, pixel_scale=0.1, background_sky_level=3.0)

        assert (sim_image.data_original == np.array(([0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0]))).all()

        assert (sim_image.data == np.array(([3.0, 3.0, 3.0],
                                            [3.0, 4.0, 3.0],
                                            [3.0, 3.0, 3.0]))).all()

        assert (sim_image.background_sky_map == np.array(([3.0, 3.0, 3.0],
                                                          [3.0, 3.0, 3.0],
                                                          [3.0, 3.0, 3.0]))).all()

        assert (sim_image.pixel_scale == 0.1)

    def test__setup_with_psf_blurring_on(self):

        image = np.array(([0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]))

        psf = np.array(([0.0, 1.0, 0.0],
                        [1.0, 2.0, 1.0],
                        [0.0, 1.0, 0.0]))

        sim_image = simulate.SimulateImage(data=image, pixel_scale=0.1,
                                           psf=imaging.PSF(psf, pixel_scale=0.1))

        assert (sim_image.data_original == np.array(([0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0]))).all()

        assert (sim_image.data == np.array(([0.0, 1.0, 0.0],
                                            [1.0, 2.0, 1.0],
                                            [0.0, 1.0, 0.0]))).all()

        assert (sim_image.psf.data == np.array(([0.0, 1.0, 0.0],
                                                [1.0, 2.0, 1.0],
                                                [0.0, 1.0, 0.0]))).all()

        assert (sim_image.pixel_scale == 0.1)

    def test__setup_with_exposure_time_map__adds_poisson_noise(self):

        image = np.array(([0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]))

        exposure_time_map = 20.0*np.ones((3,3))

        sim_image = simulate.SimulateImage(data=image, pixel_scale=0.1, exposure_time_map=exposure_time_map,
                                           noise_seed=1)

        assert (sim_image.data_original == np.array(([0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0]))).all()

        assert sim_image.data == pytest.approx(np.array(([0.0, 0.0, 0.0],
                                                         [0.0, 1.05, 0.0],
                                                         [0.0, 0.0, 0.0])), 1e-2)

        assert sim_image.pixel_scale == 0.1

        assert (sim_image.exposure_time_map == 20.0*np.ones((3,3))).all()

        assert sim_image.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                      [0.0, 0.05, 0.0],
                                                                      [0.0, 0.0, 0.0]]), 1e-2)



class TestGeneratePoissonNoiseMap:

    def test__input_image_all_0s__exposure_times_all_1s__all_noise_values_are_zeros(self):

        image = np.zeros((2, 2))

        exposure_time_map = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time_map, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.zeros((2, 2))).all()

    def test__input_image_has_10s__exposure_times_all_1s__gives_noise_values_near_1_to_5(self):

        image = np.array([[10., 0.],
                          [0., 10.]])

        exposure_time_map = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time_map, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[1, 0],  # Use known noise map for given seed.
                                               [0, 4]])).all()

    def test__input_image_has_1000000s__exposure_times_all_1s__these_give_positive_noise_values_near_1000(self):

        image = np.array([[10000000., 0.],
                          [0., 10000000.]])

        exposure_time_map = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time_map, seed=2)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[571, 0],  # Use known noise map for given seed.
                                               [0, -441]])).all()

    def test__two_images__same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(self):
        image_0 = np.array([[10., 0.],
                            [0., 10.]])

        exposure_time_map_0 = np.ones((2, 2))

        image_1 = np.array([[5., 0.],
                            [0., 5.]])

        exposure_time_map_1 = 2.0 * np.ones((2, 2))

        poisson_noise_map_0 = simulate.generate_poisson_noise_map(image_0, exposure_time_map_0, seed=1)
        poisson_noise_map_1 = simulate.generate_poisson_noise_map(image_1, exposure_time_map_1, seed=1)

        assert (poisson_noise_map_0 / 2.0 == poisson_noise_map_1).all()

    def test__same_as_above_but_range_of_image_values_and_exposure_times(self):
        image_0 = np.array([[10., 20.],
                            [30., 40.]])

        exposure_time_map_0 = np.array([[2., 2.],
                                        [3., 4.]])

        image_1 = np.array([[20., 20.],
                            [45., 20.]])

        exposure_time_map_1 = np.array([[1., 2.],
                                        [2., 8.]])

        poisson_noise_map_0 = simulate.generate_poisson_noise_map(image_0, exposure_time_map_0, seed=1)
        poisson_noise_map_1 = simulate.generate_poisson_noise_map(image_1, exposure_time_map_1, seed=1)

        assert (poisson_noise_map_0[0, 0] == poisson_noise_map_1[0, 0] / 2.0).all()
        assert (poisson_noise_map_0[0, 1] == poisson_noise_map_1[0, 1]).all()
        assert (poisson_noise_map_0[1, 0] * 1.5 == pytest.approx(poisson_noise_map_1[1, 0], 1e-2)).all()
        assert (poisson_noise_map_0[1, 1] / 2.0 == poisson_noise_map_1[1, 1]).all()


class TestGenerateGaussianNoiseMap:

    def test__input_mean_is_0__mean_of_image_values_consistent_with_0(self):

        gaussian_noise_map = simulate.generate_gaussian_noise_map(dimensions=(5, 5), mean=0.0, sigma=1.0,
                                                               seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert -0.1 <= np.mean(gaussian_noise_map) <= 0.1

    def test__input_mean_is_10__mean_of_image_values_consistent_with_10(self):
        gaussian_noise_map = simulate.generate_gaussian_noise_map(dimensions=(5, 5), mean=10.0,
                                                               sigma=1.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 9.9 <= np.mean(gaussian_noise_map) <= 10.1

    def test__input_sigma_is_1__standard_deviation_of_image_values_consistent_with_1(self):
        gaussian_noise_map = simulate.generate_gaussian_noise_map(dimensions=(5, 5), mean=10.0,
                                                               sigma=1.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 0.9 <= np.std(gaussian_noise_map) <= 1.1

    def test__input_sigma_is_10__standard_deviation_of_image_values_consistent_with_10(self):
        gaussian_noise_map = simulate.generate_gaussian_noise_map(dimensions=(5, 5), mean=100.0,
                                                               sigma=10.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 9.0 <= np.std(gaussian_noise_map) <= 11.0

    def test__known_noise_map_for_mean_0_sigma_1_seed_1(self):
        gaussian_noise_map = simulate.generate_gaussian_noise_map(dimensions=(5, 5), mean=0.0, sigma=1.0,
                                                               seed=1)

        assert gaussian_noise_map == pytest.approx(
            np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
                      [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
                      [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944],
                      [-1.09989127, -0.17242821, -0.87785842, 0.04221375, 0.58281521],
                      [-1.10061918, 1.14472371, 0.90159072, 0.50249434, 0.90085595]]), 1e-2)