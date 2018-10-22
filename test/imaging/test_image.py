import os

import numpy as np
import pytest

from autolens import exc
from autolens.imaging import scaled_array
from autolens.imaging import image
from autolens.imaging import imaging_util

test_data_dir = "{}/../test_files/data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestPrepatoryImage:
    class TestConstructor:

        def test__setup_image__correct_attributes(self):
            array = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            psf = image.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)
            noise = 5.0 * np.ones((3, 3))

            im = image.PreparatoryImage(array=array, pixel_scale=0.1, noise_map=noise, psf=psf,
                                        background_noise_map=7.0 * np.ones((3, 3)),
                                        poisson_noise_map=9.0 * np.ones((3, 3)),
                                        exposure_time=100.0, effective_exposure_map=11.0 * np.ones((3, 3)))

            assert im == pytest.approx(np.array([[1.0, 2.0, 3.0],
                                                 [4.0, 5.0, 6.0],
                                                 [7.0, 8.0, 9.0]]), 1e-2)
            assert (im.psf == 3.0 * np.ones((3, 3))).all()
            assert (im.noise_map == 5.0 * np.ones((3, 3))).all()
            assert (im.background_noise_map == 7.0 * np.ones((3, 3))).all()
            assert (im.poisson_noise_map == 9.0 * np.ones((3, 3))).all()
            assert (im.effective_exposure_map == 11.0 * np.ones((3, 3))).all()
            assert (im.exposure_time == 100.0)

    class TestSimulateImage(object):

        def test__setup_with_all_features_off(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, effective_exposure_map=exposure_map,
                                                                      pixel_scale=0.1, add_noise=False)

            assert (sim_img.effective_exposure_map == np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1
            assert (sim_img == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()

        def test__setup_with_background_sky_on__noise_off__no_noise_in_image(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            background_sky = image.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                                      effective_exposure_map=exposure_map,
                                                                      background_sky_map=background_sky,
                                                                      add_noise=False,
                                                                      seed=1)

            assert (sim_img.effective_exposure_map == 1.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert (sim_img == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()

            assert (sim_img.background_noise_map == 4.0 * np.ones((3, 3))).all()

        def test__setup_with_background_sky_on__noise_on_so_background_adds_noise_to_image(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            background_sky = image.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                                      background_sky_map=background_sky,
                                                                      effective_exposure_map=exposure_map,
                                                                      add_noise=True,
                                                                      seed=1)

            assert (sim_img.effective_exposure_map == 1.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert (sim_img == np.array([[1.0, 5.0, 4.0],
                                         [1.0, 2.0, 1.0],
                                         [5.0, 2.0, 7.0]])).all()

            assert (sim_img.poisson_noise_map == np.array([[np.sqrt(1.0), np.sqrt(5.0), np.sqrt(4.0)],
                                                           [np.sqrt(1.0), np.sqrt(2.0), np.sqrt(1.0)],
                                                           [np.sqrt(5.0), np.sqrt(2.0), np.sqrt(7.0)]])).all()

            assert (sim_img.background_noise_map == 4.0 * np.ones((3, 3))).all()

        def test__setup_with_psf_blurring_on__blurs_image_and_trims_psf_edge_off(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = image.PSF(array=np.array([[0.0, 1.0, 0.0],
                                            [1.0, 2.0, 1.0],
                                            [0.0, 1.0, 0.0]]), pixel_scale=1.0)

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                                      effective_exposure_map=exposure_map, psf=psf,
                                                                      add_noise=False)

            assert (sim_img == np.array([[0.0, 1.0, 0.0],
                                         [1.0, 2.0, 1.0],
                                         [0.0, 1.0, 0.0]])).all()
            assert (sim_img.effective_exposure_map == np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

        def test__setup_with_background_sky_and_psf_on__psf_does_no_blurring__image_and_sky_both_trimmed(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]), pixel_scale=1.0)

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            background_sky = image.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                                      effective_exposure_map=exposure_map,
                                                                      psf=psf, background_sky_map=background_sky,
                                                                      add_noise=False, seed=1)

            assert (sim_img.effective_exposure_map == 1.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert (sim_img == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()

            assert (sim_img.background_noise_map == 4.0 * np.ones((3, 3))).all()

        def test__setup_with_noise(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = image.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                                      effective_exposure_map=exposure_map,
                                                                      add_noise=True, seed=1)

            assert (sim_img.effective_exposure_map == 20.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert sim_img == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 1.05, 0.0],
                                                      [0.0, 0.0, 0.0]]), 1e-2)

            # Because of the image value is 1.05, the estimated Poisson noise_map is:
            # sqrt((1.05 * 20))/20 = 0.2291

            assert sim_img.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                        [0.0, 0.2291, 0.0],
                                                                        [0.0, 0.0, 0.0]]), 1e-2)

            assert sim_img.noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                [0.0, 0.2291, 0.0],
                                                                [0.0, 0.0, 0.0]]), 1e-2)

        def test__setup_with__psf_blurring_and_poisson_noise_on__poisson_noise_added_to_blurred_image(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = image.PSF(array=np.array([[0.0, 1.0, 0.0],
                                            [1.0, 2.0, 1.0],
                                            [0.0, 1.0, 0.0]]), pixel_scale=1.0)

            exposure_map = image.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                                      effective_exposure_map=exposure_map,
                                                                      psf=psf,
                                                                      add_noise=True, seed=1)

            assert (sim_img.effective_exposure_map == 20.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1
            assert sim_img == pytest.approx(np.array([[0.0, 1.05, 0.0],
                                                      [1.3, 2.35, 1.05],
                                                      [0.0, 1.05, 0.0]]), 1e-2)

            # The estimated Poisson noises are:
            # sqrt((2.35 * 20))/20 = 0.3427
            # sqrt((1.3 * 20))/20 = 0.2549
            # sqrt((1.05 * 20))/20 = 0.2291

            assert sim_img.poisson_noise_map == pytest.approx(np.array([[0.0, 0.2291, 0.0],
                                                                        [0.2549, 0.3427, 0.2291],
                                                                        [0.0, 0.2291, 0.0]]), 1e-2)

        def test__simulate_function__turns_exposure_time_and_sky_level_to_arrays(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = image.PSF(array=np.array([[0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0]]), pixel_scale=1.0)

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)
            background_sky = image.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=img.shape)
            sim_img_variable = image.PreparatoryImage.simulate_variable_arrays(array=img,
                                                                               effective_exposure_map=exposure_map,
                                                                               psf=psf,
                                                                               background_sky_map=background_sky,
                                                                               pixel_scale=0.1, add_noise=False, seed=1)

            img = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            sim_img = image.PreparatoryImage.simulate(array=img, pixel_scale=0.1, exposure_time=1.0,
                                                      background_sky_level=16.0, psf=psf, seed=1)

            assert (sim_img_variable.effective_exposure_map == sim_img.effective_exposure_map).all()
            assert sim_img_variable.pixel_scale == sim_img.pixel_scale
            assert sim_img_variable == pytest.approx(sim_img, 1e-4)
            assert (sim_img_variable.background_noise_map == sim_img.background_noise_map).all()

        def test__target_signal_to_noise__no_background_sky(self):
            img = np.array([[0.01, 0.02, 0.01],
                            [0.01, 5.0, 0.01],
                            [0.01, 0.01, 0.01]])

            exposure_time = image.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_to_target_signal_to_noise(array=img, pixel_scale=0.1,
                                                                                target_signal_to_noise=30.0,
                                                                                effective_exposure_map=exposure_time,
                                                                                seed=1)

            assert 29.3 < sim_img.signal_to_noise_max < 30.7

        def test__target_signal_to_noise__background_sky_and_poisson(self):
            img = np.array([[0.01, 0.02, 0.01],
                            [0.01, 5.0, 0.01],
                            [0.01, 0.01, 0.01]])

            exposure_time = image.ScaledSquarePixelArray.single_value(value=2.0, pixel_scale=0.1, shape=img.shape)
            background_sky = image.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.PreparatoryImage.simulate_to_target_signal_to_noise(array=img, pixel_scale=0.1,
                                                                                target_signal_to_noise=30.0,
                                                                                effective_exposure_map=exposure_time,
                                                                                background_sky_map=background_sky,
                                                                                seed=1)

            assert 29.2 < sim_img.signal_to_noise_max < 30.8

    class TestSimulatePoissonNoise(object):

        def test__input_img_all_0s__exposure_time_all_1s__all_noise_values_are_0s(self):
            img = np.zeros((2, 2))

            exposure_time = image.ScaledSquarePixelArray.single_value(1.0, img.shape, pixel_scale=0.1)
            sim_poisson_img = img + image.generate_poisson_noise(img, exposure_time.data, seed=1)

            assert sim_poisson_img.shape == (2, 2)
            assert (sim_poisson_img == np.zeros((2, 2))).all()

        def test__input_img_includes_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            img = np.array([[10., 0.],
                            [0., 10.]])

            exposure_time = image.ScaledSquarePixelArray.single_value(1.0, img.shape, pixel_scale=0.1)
            poisson_noise_map = image.generate_poisson_noise(img, exposure_time.data, seed=1)
            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise_map map for given seed.
            assert (poisson_noise_map == np.array([[1, 0],
                                                   [0, 4]])).all()
            assert (sim_poisson_img == np.array([[11, 0],
                                                 [0, 14]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__input_img_is_all_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            img = np.array([[10., 10.],
                            [10., 10.]])

            exposure_time = image.ScaledSquarePixelArray.single_value(1.0, img.shape, pixel_scale=0.1)
            poisson_noise_map = image.generate_poisson_noise(img, exposure_time.data, seed=1)
            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise_map map for given seed.
            assert (poisson_noise_map == np.array([[1, 4],
                                                   [3, 1]])).all()

            assert (sim_poisson_img == np.array([[11, 14],
                                                 [13, 11]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__input_img_has_1000000s__exposure_times_is_1s__these_give_positive_noise_values_near_1000(self):
            img = np.array([[10000000., 0.],
                            [0., 10000000.]])

            exposure_time = image.ScaledSquarePixelArray(array=np.ones((2, 2)), pixel_scale=0.1)

            poisson_noise_map = image.generate_poisson_noise(img, exposure_time.data, seed=2)

            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise_map map for given seed.
            assert (poisson_noise_map == np.array([[571, 0],
                                                   [0, -441]])).all()

            assert (sim_poisson_img == np.array([[10000000.0 + 571, 0.],
                                                 [0., 10000000.0 - 441]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__two_imgs_same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(self):
            img_0 = np.array([[10., 0.],
                              [0., 10.]])

            exposure_time_0 = image.ScaledSquarePixelArray(array=np.ones((2, 2)), pixel_scale=0.1)

            img_1 = np.array([[5., 0.],
                              [0., 5.]])

            exposure_time_1 = image.ScaledSquarePixelArray(array=2.0 * np.ones((2, 2)), pixel_scale=0.1)

            sim_poisson_img_0 = img_0 + image.generate_poisson_noise(img_0, exposure_time_0.data, seed=1)
            sim_poisson_img_1 = img_1 + image.generate_poisson_noise(img_1, exposure_time_1.data, seed=1)

            assert (sim_poisson_img_0 / 2.0 == sim_poisson_img_1).all()

        def test__same_as_above_but_range_of_img_values_and_exposure_times(self):
            img_0 = np.array([[10., 20.],
                              [30., 40.]])

            exposure_time_0 = image.ScaledSquarePixelArray(array=np.array([[2., 2.],
                                                                           [3., 4.]]), pixel_scale=0.1)

            img_1 = np.array([[20., 20.],
                              [45., 20.]])

            exposure_time_1 = image.ScaledSquarePixelArray(array=np.array([[1., 2.],
                                                                           [2., 8.]]), pixel_scale=0.1)

            sim_poisson_img_0 = img_0 + image.generate_poisson_noise(img_0, exposure_time_0.data, seed=1)
            sim_poisson_img_1 = img_1 + image.generate_poisson_noise(img_1, exposure_time_1.data, seed=1)

            assert (sim_poisson_img_0[0, 0] == sim_poisson_img_1[0, 0] / 2.0).all()
            assert sim_poisson_img_0[0, 1] == sim_poisson_img_1[0, 1]
            assert (sim_poisson_img_0[1, 0] * 1.5 == pytest.approx(sim_poisson_img_1[1, 0], 1e-2)).all()
            assert (sim_poisson_img_0[1, 1] / 2.0 == sim_poisson_img_1[1, 1]).all()

    class TestEstimateNoiseFromImage:

        def test__image_and_exposure_time_all_1s__no_background__noise_is_all_1s(self):
            # Image (eps) = 1.0
            # Background (eps) = 0.0
            # Exposure times = 1.0 s
            # Image (counts) = 1.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(1.0 + 0.0**2) = 1.0
            # Noise (eps) = 1.0 / 1.0

            array = np.ones((3, 3))
            exposure_time = np.ones((3, 3))
            background_noise = np.zeros((3, 3))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert (noise_estimate == np.ones((3, 3))).all()

        def test__image_all_4s__exposure_time_all_1s__no_background__noise_is_all_2s(self):
            # Image (eps) = 4.0
            # Background (eps) = 0.0
            # Exposure times = 1.0 s
            # Image (counts) = 4.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
            # Noise (eps) = 2.0 / 1.0

            array = 4.0 * np.ones((4, 2))

            exposure_time = np.ones((4, 2))
            background_noise = np.zeros((4, 2))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert (noise_estimate == 2.0 * np.ones((4, 2))).all()

        def test__image_all_1s__exposure_time_all_4s__no_background__noise_is_all_2_divided_4_so_halves(self):
            # Image (eps) = 1.0
            # Background (eps) = 0.0
            # Exposure times = 4.0 s
            # Image (counts) = 4.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
            # Noise (eps) = 2.0 / 4.0 = 0.5

            array = np.ones((1, 5))

            exposure_time = 4.0 * np.ones((1, 5))

            background_noise = np.zeros((1, 5))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert (noise_estimate == 0.5 * np.ones((1, 5))).all()

        def test__image_and_exposure_times_range_of_values__no_background__noises_estimates_correct(self):
            array = np.array([[5.0, 3.0],
                              [10.0, 20.0]])

            exposure_time = image.ScaledSquarePixelArray(np.array([[1.0, 2.0],
                                                                   [3.0, 4.0]]), pixel_scale=1.0)

            background_noise = np.zeros((2, 2))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert (noise_estimate == np.array([[np.sqrt(5.0), np.sqrt(6.0) / 2.0],
                                                [np.sqrt(30.0) / 3.0, np.sqrt(80.0) / 4.0]])).all()

        def test__image_and_exposure_times_all_1s__background_is_float_sqrt_3__noise_is_all_2s(self):
            # Image (eps) = 1.0
            # Background (eps) = sqrt(3.0)
            # Exposure times = 1.0 s
            # Image (counts) = 1.0
            # Background (counts) = sqrt(3.0)

            # Noise (counts) = sqrt(1.0 + sqrt(3.0)**2) = sqrt(1.0 + 3.0) = 2.0
            # Noise (eps) = 2.0 / 1.0 = 2.0

            array = np.ones((3, 3))

            exposure_time = np.ones((3, 3))

            background_noise = 3.0 ** 0.5 * np.ones((3, 3))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(2.0 * np.ones((3, 3)), 1e-2)

        def test__image_and_exposure_times_all_1s__background_is_float_5__noise_all_correct(self):
            # Image (eps) = 1.0
            # Background (eps) = 5.0
            # Exposure times = 1.0 s
            # Image (counts) = 1.0
            # Background (counts) = 5.0

            # Noise (counts) = sqrt(1.0 + 5**2)
            # Noise (eps) = sqrt(1.0 + 5**2) / 1.0

            array = np.ones((2, 3))

            exposure_time = np.ones((2, 3))

            background_noise = 5 * np.ones((2, 3))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(
                np.array([[np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
                          [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)]]), 1e-2)

        def test__image_all_1s__exposure_times_all_2s__background_is_float_5__noise_all_correct(self):
            # Image (eps) = 1.0
            # Background (eps) = 5.0
            # Exposure times = 2.0 s
            # Image (counts) = 2.0
            # Background (counts) = 10.0

            # Noise (counts) = sqrt(2.0 + 10**2) = sqrt(2.0 + 100.0)
            # Noise (eps) = sqrt(2.0 + 100.0) / 2.0

            array = np.ones((2, 3))

            exposure_time = 2.0 * np.ones((2, 3))
            background_noise = 5.0 * np.ones((2, 3))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(
                np.array([[np.sqrt(2.0 + 100.0) / 2.0, np.sqrt(2.0 + 100.0) / 2.0, np.sqrt(2.0 + 100.0) / 2.0],
                          [np.sqrt(2.0 + 100.0) / 2.0, np.sqrt(2.0 + 100.0) / 2.0, np.sqrt(2.0 + 100.0) / 2.0]]),
                1e-2)

        def test__same_as_above_but_different_image_values_in_each_pixel_and_new_background_values(self):
            # Can use pattern from previous test for values

            array = np.array([[1.0, 2.0],
                              [3.0, 4.0],
                              [5.0, 6.0]])

            exposure_time = np.ones((3, 2))
            background_noise = 12.0 * np.ones((3, 2))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(np.array([[np.sqrt(1.0 + 144.0), np.sqrt(2.0 + 144.0)],
                                                             [np.sqrt(3.0 + 144.0), np.sqrt(4.0 + 144.0)],
                                                             [np.sqrt(5.0 + 144.0), np.sqrt(6.0 + 144.0)]]), 1e-2)

        def test__image_and_exposure_times_range_of_values__background_has_value_9___noise_estimates_correct(self):
            # Use same pattern as above, noting that here our background values are now being converts to counts using
            # different exposure time and then being squared.

            array = np.array([[5.0, 3.0],
                              [10.0, 20.0]])

            exposure_time = np.array([[1.0, 2.0],
                                      [3.0, 4.0]])
            background_noise = 9.0 * np.ones((2, 2))

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(np.array([[np.sqrt(5.0 + 81.0), np.sqrt(6.0 + 18.0 ** 2.0) / 2.0],
                                                             [np.sqrt(30.0 + 27.0 ** 2.0) / 3.0,
                                                              np.sqrt(80.0 + 36.0 ** 2.0) / 4.0]]),
                                                   1e-2)

        def test__image_and_exposure_times_and_background_are_all_ranges_of_values__noise_estimates_correct(self):
            # Use same pattern as above, noting that we are now also using a variable background signal_to_noise_ratio map.

            array = np.array([[5.0, 3.0],
                              [10.0, 20.0]])

            exposure_time = np.array([[1.0, 2.0],
                                      [3.0, 4.0]])

            background_noise = np.array([[5.0, 6.0],
                                         [7.0, 8.0]])

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                         effective_exposure_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(
                np.array([[np.sqrt(5.0 + 5.0 ** 2.0), np.sqrt(6.0 + 12.0 ** 2.0) / 2.0],
                          [np.sqrt(30.0 + 21.0 ** 2.0) / 3.0,
                           np.sqrt(80.0 + 32.0 ** 2.0) / 4.0]]),
                1e-2)

    class TestEstimateDataGrid(object):

        def test__via_edges__input_all_ones__sky_bg_level_1(self):
            img = image.PreparatoryImage(array=np.ones((3, 3)), noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):
            image_array = np.array([[1, 1, 1],
                                    [1, 100, 1],
                                    [1, 1, 1]])

            img = image.PreparatoryImage(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):
            image_array = np.array([[1, 1, 1],
                                    [1, 100, 1],
                                    [1, 100, 1],
                                    [1, 1, 1]])

            img = image.PreparatoryImage(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image_array = np.array([[1, 1, 1, 1],
                                    [1, 100, 100, 1],
                                    [1, 100, 100, 1],
                                    [1, 1, 1, 1]])

            img = image.PreparatoryImage(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):
            image_array = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 100, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])

            img = image.PreparatoryImage(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=2)

            assert sky_noise == 0.0

        def test__via_edges__6x5_image_two_edges__values(self):
            image_array = np.array([[0, 1, 2, 3, 4],
                                    [5, 6, 7, 8, 9],
                                    [10, 11, 100, 12, 13],
                                    [14, 15, 100, 16, 17],
                                    [18, 19, 20, 21, 22],
                                    [23, 24, 25, 26, 27]])

            img = image.PreparatoryImage(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=2)

            assert sky_noise == np.std(np.arange(28))

        def test__via_edges__7x7_image_three_edges__values(self):
            image_array = np.array([[0, 1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [21, 22, 23, 100, 24, 25, 26],
                                    [27, 28, 29, 30, 31, 32, 33],
                                    [34, 35, 36, 37, 38, 39, 40],
                                    [41, 42, 43, 44, 45, 46, 47]])

            img = image.PreparatoryImage(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                         pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=3)

            assert sky_noise == np.std(np.arange(48))

    class TestSignalToNoise:

        def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):

            array = np.array([[1.0, 2.0],
                              [3.0, 4.0]])

            noise = np.array([[10.0, 10.0],
                              [30.0, 4.0]])

            img = image.PreparatoryImage(array=array, pixel_scale=1.0,
                                         psf=image.PSF(array=np.ones((2, 2)), pixel_scale=1.0), noise_map=noise)

            assert (img.signal_to_noise_map == np.array([[0.1, 0.2],
                                                         [0.1, 1.0]])).all()
            assert img.signal_to_noise_max == 1.0


class TestImage(object):

    class TestConstructor:

        def test__setup_image__correct_attributes(self):
            array = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            psf = image.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)
            noise = 5.0 * np.ones((3, 3))

            im = image.Image(array=array, pixel_scale=0.1, noise_map=noise, psf=psf)

            assert im == pytest.approx(np.array([[1.0, 2.0, 3.0],
                                                 [4.0, 5.0, 6.0],
                                                 [7.0, 8.0, 9.0]]), 1e-2)
            assert (im.psf == 3.0 * np.ones((3, 3))).all()
            assert (im.noise_map == 5.0 * np.ones((3, 3))).all()
            assert (im.background_noise_map == None)


    class TestTrimming:

        def test_trim_around_centre(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[2:4, 2:4] = 2.0

            noise_map_array = scaled_array.Array(np.ones((6, 6)))
            noise_map_array[3,3] = 2.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array)

            im = im.trim_image_and_noise_around_centre(new_shape=(4, 4))

            assert (im == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()

            assert (im.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 2.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0]])).all()

            assert im.background_noise_map == None

        def test_trim_around_centre__include_background_noise(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[2:4, 2:4] = 2.0

            noise_map_array = scaled_array.Array(np.ones((6, 6)))
            noise_map_array[3,3] = 2.0

            bg_noise_map_array = scaled_array.Array(np.ones((6, 6)))
            bg_noise_map_array[2,2] = 2.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=bg_noise_map_array)

            im = im.trim_image_and_noise_around_centre(new_shape=(4, 4))

            assert (im == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()

            assert (im.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 2.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0]])).all()

            assert (im.background_noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                                         [1.0, 2.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0, 1.0]])).all()

        def test_trim_around_region(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[2:4, 2:4] = 2.0

            noise_map_array = scaled_array.Array(np.ones((6, 6)))
            noise_map_array[3,3] = 2.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array)

            im = im.trim_image_and_noise_around_region(x0=1, x1=5, y0=1, y1=5)

            assert (im == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()

            assert (im.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 2.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0]])).all()

            assert im.background_noise_map == None

        def test_trim_around_region__include_background_noise(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[2:4, 2:4] = 2.0

            noise_map_array = scaled_array.Array(np.ones((6, 6)))
            noise_map_array[3,3] = 2.0

            bg_noise_map_array = scaled_array.Array(np.ones((6, 6)))
            bg_noise_map_array[2,2] = 2.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=bg_noise_map_array)

            im = im.trim_image_and_noise_around_region(x0=1, x1=5, y0=1, y1=5)

            assert (im == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 2.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()

            assert (im.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0, 2.0, 1.0],
                                              [1.0, 1.0, 1.0, 1.0]])).all()

            assert (im.background_noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                                         [1.0, 2.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0, 1.0]])).all()


class TestNoiseMap(object):

    class TestFromWeightMap:

        def test__weight_map_no_zeros__uses_1_over_sqrt_value(self):

            weight_map = np.array([[1.0, 4.0, 16.0],
                                   [1.0, 4.0, 16.0]])

            noise_map = image.NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.5, 0.25],
                                           [1.0, 0.5, 0.25]])).all()

        def test__weight_map_no_zeros__zeros_set_to_10000000(self):

            weight_map = np.array([[1.0, 4.0, 0.0],
                                   [1.0, 4.0, 16.0]])

            noise_map = image.NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.5, 1.0e8],
                                           [1.0, 0.5, 0.25]])).all()


class TestPSF(object):
    class TestConstructors(object):

        def test__init__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF(array=np.ones((3, 3)), pixel_scale=1.0, renormalize=False)

            assert psf.shape == (3, 3)
            assert psf.pixel_scale == 1.0
            assert (psf == np.ones((3, 3))).all()

        def test__init__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF(array=np.ones((4, 3)), pixel_scale=1.0, renormalize=False)

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.shape == (4, 3)

        def test__from_fits__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF.from_fits_with_scale(file_path=test_data_dir + '3x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (psf == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0

        def test__from_fits__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF.from_fits_with_scale(file_path=test_data_dir + '4x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0

    class TestRenormalize(object):

        def test__input_is_already_normalized__no_change(self):
            psf_data = np.ones((3, 3)) / 9.0

            psf = image.PSF(array=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):
            psf_data = np.ones((3, 3))

            psf = image.PSF(array=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__same_as_above__renomalized_false_does_not_renormalize(self):
            psf_data = np.ones((3, 3))

            psf = image.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            assert psf == pytest.approx(np.ones((3, 3)), 1e-3)

    class TestConvolve(object):

        def test__kernel_is_not_odd_x_odd__raises_error(self):
            kernel = np.array([[0.0, 1.0],
                               [1.0, 2.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            with pytest.raises(exc.KernelException):
                psf.convolve(np.ones((5, 5)))

        def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == kernel).all()

        def test__image_is_4x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == np.array([[0.0, 1.0, 0.0, 0.0],
                                             [1.0, 2.0, 1.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0]])).all()

        def test__image_is_4x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == np.array([[0.0, 1.0, 0.0],
                                             [1.0, 2.0, 1.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])).all()

        def test__image_is_3x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == np.array([[0.0, 1.0, 0.0, 0.0],
                                             [1.0, 2.0, 1.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0]])).all()

        def test__image_is_4x4_has_two_central_values__kernel_is_asymmetric__blurred_image_follows_convolution(self):
            img = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == np.array([[1.0, 1.0, 1.0, 0.0],
                                             [2.0, 3.0, 2.0, 1.0],
                                             [1.0, 5.0, 5.0, 1.0],
                                             [0.0, 1.0, 3.0, 3.0]])).all()

        def test__image_is_4x4_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
                self):
            img = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == np.array([[1.0, 1.0, 0.0, 0.0],
                                             [2.0, 1.0, 1.0, 1.0],
                                             [3.0, 3.0, 2.0, 2.0],
                                             [0.0, 0.0, 1.0, 3.0]])).all()

        def test__image_is_4x4_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
                self):
            img = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_img = psf.convolve(img)

            assert (blurred_img == np.array([[2.0, 1.0, 0.0, 0.0],
                                             [3.0, 3.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.0, 1.0],
                                             [0.0, 0.0, 2.0, 2.0]])).all()

    class TestSimulateAsGaussian(object):

        def test__identical_to_gaussian_light_profile(self):
            from autolens.profiles import light_profiles as lp

            grid = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=np.full((3, 3), False),
                                                                                pixel_scales=(1.0, 1.0))

            gaussian = lp.EllipticalGaussian(centre=(0.1, 0.1), axis_ratio=0.9, phi=45.0, intensity=1.0, sigma=1.0)
            profile_gaussian_1d = gaussian.intensities_from_grid(grid)
            profile_gaussian_2d = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
                array_1d=profile_gaussian_1d, shape=(3, 3))
            profile_psf = image.PSF(array=profile_gaussian_2d, pixel_scale=1.0, renormalize=True)

            imaging_psf = image.PSF.simulate_as_gaussian(shape=(3, 3), pixel_scale=1.0, centre=(0.1, 0.1),
                                                         axis_ratio=0.9, phi=45.0, sigma=1.0)

            assert profile_psf == pytest.approx(imaging_psf, 1e-4)


class TestLoadImagingFromFits(object):

    def test__no_settings_just_pass_fits(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                          noise_map_path=test_data_dir + '3x3_ones.fits',
                                          psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1)

        assert (im == np.ones((3,3))).all()
        assert (im.noise_map == np.ones((3,3))).all()
        assert (im.psf == np.ones((3,3))).all()
        assert (im.pixel_scale == 0.1)

    def test__pad_shape_of_image_noise_and_psf(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                          noise_map_path=test_data_dir + '3x3_ones.fits',
                                          psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          image_shape=(5,5), psf_shape=(7,7))

        padded_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 1.0, 1.0, 0.0],
                                 [0.0, 1.0, 1.0, 1.0, 0.0],
                                 [0.0, 1.0, 1.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0]])

        assert (im == padded_array).all()
        assert (im.noise_map == padded_array).all()

        psf_padded_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        assert (im.psf == psf_padded_array).all()
        assert (im.pixel_scale == 0.1)

    def test__trim_shape_of_image_noise_and_psf(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                          noise_map_path=test_data_dir + '3x3_ones.fits',
                                          psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          image_shape=(1,1), psf_shape=(1,1))

        padded_array = np.array([[1.0]])

        assert (im == padded_array).all()
        assert (im.noise_map == padded_array).all()

        psf_padded_array = np.array([[1.0]])

        assert (im.psf == psf_padded_array).all()
        assert (im.pixel_scale == 0.1)