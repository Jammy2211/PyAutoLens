import numpy as np
import pytest
import os

from auto_lens.imaging import image

test_data_dir = "{}/../../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestSimulateImage(object):
    class TestConstructor(object):

        def test__setup_with_all_features_off(self):
            img = np.array(([0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]))

            exposure_time = image.DataGrid.single_value(value=1.0, pixel_scale=0.1,
                                                        shape=img.shape)

            sim_img = image.Image.simulate(array=img, effective_exposure_time=exposure_time, pixel_scale=0.1)

            assert (sim_img.exposure_time == np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert (sim_img.data_original == np.array(([0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0]))).all()
            assert (sim_img.data == np.array(([0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0],
                                              [0.0, 0.0, 0.0]))).all()

            assert sim_img.sim_optics is None
            assert sim_img.sim_poisson_noise is None
            assert sim_img.sim_background_noise is None

        def test__setup_with_psf_blurring_on(self):
            img = np.array(([0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]))

            psf = image.PSF(array=np.array(([0.0, 1.0, 0.0],
                                            [1.0, 2.0, 1.0],
                                            [0.0, 1.0, 0.0])), pixel_scale=0.1)

            exposure_time = image.DataGrid.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.Image.simulate(array=img, effective_exposure_time=exposure_time.data, pixel_scale=0.1,
                                           psf=psf)

            assert (sim_img.exposure_time == np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert (sim_img.data_original == np.array(([0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0]))).all()
            assert (sim_img.data == np.array(([0.0, 1.0, 0.0],
                                              [1.0, 2.0, 1.0],
                                              [0.0, 1.0, 0.0]))).all()

            assert (sim_img.sim_optics.psf.data == psf).all()

            assert sim_img.sim_poisson_noise is None
            assert sim_img.sim_background_noise is None

        def test__setup_with__poisson_noise_on(self):
            img = np.array(([0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]))

            exposure_time = image.DataGrid.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.Image.simulate(array=img, pixel_scale=0.1, effective_exposure_time=exposure_time,
                                           poisson_noise=image.generate_poisson_noise(img, exposure_time, seed=1))

            assert (sim_img.exposure_time.data == 20.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1
            #
            # assert (sim_img.data_original == np.array(([0.0, 0.0, 0.0],
            #                                              [0.0, 1.0, 0.0],
            #                                              [0.0, 0.0, 0.0]))).all()

            assert sim_img.data == pytest.approx(np.array(([0.0, 0.0, 0.0],
                                                           [0.0, 2.05, 0.0],
                                                           [0.0, 0.0, 0.0])), 1e-2)

            assert sim_img.sim_poisson_noise.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                                          [0.0, 0.05, 0.0],
                                                                                          [0.0, 0.0, 0.0]]), 1e-2)

            assert sim_img.sim_optics is None
            assert sim_img.sim_background_noise is None


class TestSimulatePoissonNoise(object):
    class TestSimulateForImage:

        def test__input_img_all_0s__exposure_time_all_1s__all_noise_values_are_0s(self):
            img = np.zeros((2, 2))

            exposure_time = image.DataGrid.single_value(1.0, img.shape, pixel_scale=0.1)
            sim_poisson_img = img + image.generate_poisson_noise(img, exposure_time.data, seed=1)

            assert sim_poisson_img.shape == (2, 2)
            assert (sim_poisson_img == np.zeros((2, 2))).all()

        def test__input_img_includes_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            img = np.array([[10., 0.],
                            [0., 10.]])

            exposure_time = image.DataGrid.single_value(1.0, img.shape, pixel_scale=0.1)
            poisson_noise_map = image.generate_poisson_noise(img, exposure_time.data, seed=1)
            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise map for given seed.
            assert (poisson_noise_map == np.array([[1, 0],
                                                   [0, 4]])).all()
            assert (sim_poisson_img == np.array([[11, 0],
                                                 [0, 14]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__input_img_is_all_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            img = np.array([[10., 10.],
                            [10., 10.]])

            exposure_time = image.DataGrid.single_value(1.0, img.shape, pixel_scale=0.1)
            poisson_noise_map = image.generate_poisson_noise(img, exposure_time.data, seed=1)
            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise map for given seed.
            assert (poisson_noise_map == np.array([[1, 4],
                                                   [3, 1]])).all()

            assert (sim_poisson_img == np.array([[11, 14],
                                                 [13, 11]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__input_img_has_1000000s__exposure_times_is_1s__these_give_positive_noise_values_near_1000(self):
            img = np.array([[10000000., 0.],
                            [0., 10000000.]])

            exposure_time = image.DataGrid(array=np.ones((2, 2)), pixel_scale=0.1)

            poisson_noise_map = image.generate_poisson_noise(img, exposure_time.data, seed=2)

            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise map for given seed.
            assert (poisson_noise_map == np.array([[571, 0],
                                                   [0, -441]])).all()

            assert (sim_poisson_img == np.array([[10000000.0 + 571, 0.],
                                                 [0., 10000000.0 - 441]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__two_imgs_same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(self):
            img_0 = np.array([[10., 0.],
                              [0., 10.]])

            exposure_time_0 = image.DataGrid(array=np.ones((2, 2)), pixel_scale=0.1)

            img_1 = np.array([[5., 0.],
                              [0., 5.]])

            exposure_time_1 = image.DataGrid(array=2.0 * np.ones((2, 2)), pixel_scale=0.1)

            sim_poisson_img_0 = img_0 + image.generate_poisson_noise(img_0, exposure_time_0.data, seed=1)
            sim_poisson_img_1 = img_1 + image.generate_poisson_noise(img_1, exposure_time_1.data, seed=1)

            assert (sim_poisson_img_0 / 2.0 == sim_poisson_img_1).all()

        def test__same_as_above_but_range_of_img_values_and_exposure_times(self):
            img_0 = np.array([[10., 20.],
                              [30., 40.]])

            exposure_time_0 = image.DataGrid(array=np.array([[2., 2.],
                                                             [3., 4.]]), pixel_scale=0.1)

            img_1 = np.array([[20., 20.],
                              [45., 20.]])

            exposure_time_1 = image.DataGrid(array=np.array([[1., 2.],
                                                             [2., 8.]]), pixel_scale=0.1)

            sim_poisson_img_0 = img_0 + image.generate_poisson_noise(img_0, exposure_time_0.data, seed=1)
            sim_poisson_img_1 = img_1 + image.generate_poisson_noise(img_1, exposure_time_1.data, seed=1)

            assert (sim_poisson_img_0[0, 0] == sim_poisson_img_1[0, 0] / 2.0).all()
            assert (sim_poisson_img_0[0, 1] == sim_poisson_img_1[0, 1]).all()
            assert (sim_poisson_img_0[1, 0] * 1.5 == pytest.approx(sim_poisson_img_1[1, 0], 1e-2)).all()
            assert (sim_poisson_img_0[1, 1] / 2.0 == sim_poisson_img_1[1, 1]).all()


class TestSimulateBackgroundNoise(object):
    def test__background_noise_sigma_0__background_noise_map_all_0__img_is_identical_to_input(self):
        img = np.zeros((3, 3))
        background_noise = image.generate_background_noise(img, sigma=0.0, seed=1)

        assert (background_noise == np.zeros((3, 3))).all()

    def test__background_noise_sigma_1__background_noise_map_all_non_0__img_has_noise_added(self):
        img = np.zeros((3, 3))
        background_noise = image.generate_background_noise(img, sigma=1.0, seed=1)

        # Use seed to give us a known read noise map we'll test for

        assert background_noise == pytest.approx(np.array([[1.62, -0.61, -0.53],
                                                           [-1.07, 0.87, -2.30],
                                                           [1.74, -0.76, 0.32]]), 1e-2)
