import numpy as np
import pytest
import os

from auto_lens.imaging import imaging, simulate


test_data_dir = "{}/../../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))

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

        assert sim_image.sky_level == 0.0
        assert sim_image.psf == None
        assert sim_image.exposure_time == None

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

    def test__setup_with_exposure_time__adds_poisson_noise(self):

        image = np.array(([0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]))

        exposure_time = 20.0*np.ones((3,3))

        sim_image = simulate.SimulateImage(data=image, pixel_scale=0.1, exposure_time=exposure_time,
                                           noise_seed=1)

        assert (sim_image.data_original == np.array(([0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0]))).all()

        assert sim_image.data == pytest.approx(np.array(([0.0, 0.0, 0.0],
                                                         [0.0, 1.05, 0.0],
                                                         [0.0, 0.0, 0.0])), 1e-2)

        assert sim_image.pixel_scale == 0.1

        assert (sim_image.exposure_time == 20.0*np.ones((3,3))).all()

        assert sim_image.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                      [0.0, 0.05, 0.0],
                                                                      [0.0, 0.0, 0.0]]), 1e-2)

    def test__setup_with_exposure_time_and_sky_level__adds_poisson_noise(self):

        image = np.array(([-20.0, -20.0, -20.0],
                          [-20.0,   0.0, -20.0],
                          [-20.0, -20.0, -20.0]))

        exposure_time = np.ones((3,3))

        sim_image = simulate.SimulateImage(data=image, pixel_scale=0.1, sky_level=20.0,
                                           exposure_time=exposure_time, noise_seed=1)

        assert (sim_image.data_original == np.array(([-20.0, -20.0, -20.0],
                                                     [-20.0, 0.0, -20.0],
                                                     [-20.0, -20.0, -20.0]))).all()

        assert sim_image.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                      [0.0, 20*0.05, 0.0],
                                                                      [0.0, 0.0, 0.0]]), 1e-2)

    def test__setup_all_features_off__from_fits(self):

        sim_image = simulate.SimulateImage.from_fits(path=test_data_dir, filename='3x3_ones.fits', hdu=0,
                                                     pixel_scale=0.1)

        assert (sim_image.data_original == np.array(([1.0, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0]))).all()

        assert (sim_image.data == np.array(([1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0]))).all()

        assert sim_image.pixel_scale == 0.1

        assert sim_image.sky_level == 0.0
        assert sim_image.psf == None
        assert sim_image.exposure_time == None


class TestGeneratePoissonNoiseMap:

    def test__input_image_all_0s__exposure_times_is_float_1__all_noise_values_are_zeros(self):

        image = np.zeros((2, 2))

        exposure_time = 1.0

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.zeros((2, 2))).all()

    def test__input_image_all_0s__exposure_time_is_ndarray_of_all_1s__all_noise_values_are_zeros(self):

        image = np.zeros((2, 2))

        exposure_time = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.zeros((2, 2))).all()

    def test__input_image_includes_10s__exposure_time_is_float_of_1__gives_noise_values_near_1_to_5(self):

        image = np.array([[10., 0.],
                          [0., 10.]])

        exposure_time = 1.0

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[1, 0],  # Use known noise map for given seed.
                                               [0, 4]])).all()

    def test__input_image_includes_10s__exposure_time_is_ndarray_of_1s__gives_same_noise_values_as_above(self):

        image = np.array([[10., 0.],
                          [0., 10.]])

        exposure_time = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[1, 0],  # Use known noise map for given seed.
                                               [0, 4]])).all()

    def test__input_image_is_all_10s__exposure_time_is_float_of_1s__gives_noise_values_near_1_to_5(self):

        image = np.array([[10., 10.],
                          [10., 10.]])

        exposure_time = 1.0

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[1, 4],  # Use known noise map for given seed.
                                               [3, 1]])).all()

    def test__input_image_is_all_10s__exposure_time_is_ndarray_of_1s__gives_noise_values_near_1_to_5(self):

        image = np.array([[10., 10.],
                          [10., 10.]])

        exposure_time = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=1)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[1, 4],  # Use known noise map for given seed.
                                               [3, 1]])).all()

    def test__input_image_has_1000000s__exposure_times_all_1s__these_give_positive_noise_values_near_1000(self):

        image = np.array([[10000000., 0.],
                          [0., 10000000.]])

        exposure_time = np.ones((2, 2))

        poisson_noise_map = simulate.generate_poisson_noise_map(image, exposure_time, seed=2)

        assert poisson_noise_map.shape == (2, 2)
        assert (poisson_noise_map == np.array([[571, 0],  # Use known noise map for given seed.
                                               [0, -441]])).all()

    def test__two_images_same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(self):

        image_0 = np.array([[10., 0.],
                            [0., 10.]])

        exposure_time_0 = 1.0

        image_1 = np.array([[5., 0.],
                            [0., 5.]])

        exposure_time_1 = 2.0

        poisson_noise_map_0 = simulate.generate_poisson_noise_map(image_0, exposure_time_0, seed=1)
        poisson_noise_map_1 = simulate.generate_poisson_noise_map(image_1, exposure_time_1, seed=1)

        assert (poisson_noise_map_0 / 2.0 == poisson_noise_map_1).all()

    def test__same_as_above_but_exposure_times_now_ndarray(self):

        image_0 = np.array([[10., 0.],
                            [0., 10.]])

        exposure_time_0 = np.ones((2, 2))

        image_1 = np.array([[5., 0.],
                            [0., 5.]])

        exposure_time_1 = 2.0 * np.ones((2, 2))

        poisson_noise_map_0 = simulate.generate_poisson_noise_map(image_0, exposure_time_0, seed=1)
        poisson_noise_map_1 = simulate.generate_poisson_noise_map(image_1, exposure_time_1, seed=1)

        assert (poisson_noise_map_0 / 2.0 == poisson_noise_map_1).all()

    def test__same_as_above_but_range_of_image_values_and_exposure_times(self):

        image_0 = np.array([[10., 20.],
                            [30., 40.]])

        exposure_time_0 = np.array([[2., 2.],
                                        [3., 4.]])

        image_1 = np.array([[20., 20.],
                            [45., 20.]])

        exposure_time_1 = np.array([[1., 2.],
                                        [2., 8.]])

        poisson_noise_map_0 = simulate.generate_poisson_noise_map(image_0, exposure_time_0, seed=1)
        poisson_noise_map_1 = simulate.generate_poisson_noise_map(image_1, exposure_time_1, seed=1)

        assert (poisson_noise_map_0[0, 0] == poisson_noise_map_1[0, 0] / 2.0).all()
        assert (poisson_noise_map_0[0, 1] == poisson_noise_map_1[0, 1]).all()
        assert (poisson_noise_map_0[1, 0] * 1.5 == pytest.approx(poisson_noise_map_1[1, 0], 1e-2)).all()
        assert (poisson_noise_map_0[1, 1] / 2.0 == poisson_noise_map_1[1, 1]).all()


class TestGenerateGaussianNoiseMap:

    def test__input_background_noise_is_float_1__mean_of_image_values_consistent_with_0(self):

        gaussian_noise_map = simulate.generate_background_noise_map(dimensions=(5, 5), background_noise=1.0,
                                                                    seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert -0.1 <= np.mean(gaussian_noise_map) <= 0.1

    def test__input_background_noise_is_float_10__standard_deviation_of_image_values_consistent_with_10(self):

        gaussian_noise_map = simulate.generate_background_noise_map(dimensions=(5, 5),
                                                                    background_noise=10.0, seed=1)

        assert gaussian_noise_map.shape == (5, 5)
        assert 9.0 <= np.std(gaussian_noise_map) <= 11.0

    def test__compare_background_noise_as_float_and_same_ndarray__same_noise_maps(self):

        gaussian_noise_map_1 = simulate.generate_background_noise_map(dimensions=(5, 5),
                                                                    background_noise=10.0, seed=1)

        background_noise = 10.0*np.ones((5,5))

        gaussian_noise_map_2 = simulate.generate_background_noise_map(dimensions=(5, 5),
                                                                    background_noise=background_noise, seed=1)

        assert (gaussian_noise_map_1 == gaussian_noise_map_2).all()

    def test__input_background_noise_is_ndarray_with_four_increasing_values__noise_values_increase(self):

        background_noise = np.array([[1.0, 100.0],
                                     [10000.0, 1000000.0]])

        gaussian_noise_map = simulate.generate_background_noise_map(dimensions=(2, 2),
                                                                    background_noise=background_noise, seed=1)

        gaussian_noise_map = np.abs(gaussian_noise_map)

        assert gaussian_noise_map[0,0] < gaussian_noise_map[0,1]
        assert gaussian_noise_map[0,1] < gaussian_noise_map[1,0]
        assert gaussian_noise_map[1,0] < gaussian_noise_map[1,1]

    def test__known_noise_map__sigma_1_seed_1(self):

        gaussian_noise_map = simulate.generate_background_noise_map(dimensions=(5, 5), background_noise=1.0, seed=1)

        assert gaussian_noise_map == pytest.approx(
            np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
                      [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
                      [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944],
                      [-1.09989127, -0.17242821, -0.87785842, 0.04221375, 0.58281521],
                      [-1.10061918, 1.14472371, 0.90159072, 0.50249434, 0.90085595]]), 1e-2)