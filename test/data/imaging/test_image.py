import os
import shutil

import numpy as np
import pytest
from astropy.modeling import functional_models
from astropy import units
from astropy.coordinates import Angle

from autolens import exc
from autolens.data.array import scaled_array
from autolens.data.imaging import image
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util

test_data_dir = "{}/../../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestImage:

    class TestConstructor:

        def test__setup_image__correct_attributes(self):

            array = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            psf = image.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)
            noise_map = 5.0 * np.ones((3, 3))

            im = image.Image(array=array, pixel_scale=0.1, noise_map=noise_map, psf=psf,
                             background_noise_map=7.0 * np.ones((3, 3)),
                             poisson_noise_map=9.0 * np.ones((3, 3)),
                             exposure_time_map=11.0 * np.ones((3, 3)))

            assert im == pytest.approx(np.array([[1.0, 2.0, 3.0],
                                                 [4.0, 5.0, 6.0],
                                                 [7.0, 8.0, 9.0]]), 1e-2)
            assert (im.psf == 3.0 * np.ones((3, 3))).all()
            assert (im.noise_map == 5.0 * np.ones((3, 3))).all()
            assert (im.background_noise_map == 7.0 * np.ones((3, 3))).all()
            assert (im.poisson_noise_map == 9.0 * np.ones((3, 3))).all()
            assert (im.exposure_time_map == 11.0 * np.ones((3, 3))).all()
            assert im.origin == (0.0, 0.0)

    class TestSimulateImage(object):

        def test__setup_with_all_features_off(self):

            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.Image.simulate_variable_arrays(array=img, exposure_time_map=exposure_map,
                                                           pixel_scale=0.1, add_noise=False)

            assert (sim_img.exposure_time_map == np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1
            assert (sim_img == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()
            assert sim_img.origin == (0.0, 0.0)

        def test__setup_with_background_sky_on__noise_off__no_noise_in_image(self):
            img = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = image.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=img.shape)

            background_sky = image.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.Image.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                           exposure_time_map=exposure_map,
                                                           background_sky_map=background_sky,
                                                           add_noise=False,
                                                           seed=1)

            assert (sim_img.exposure_time_map == 1.0 * np.ones((3, 3))).all()
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

            sim_img = image.Image.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                           background_sky_map=background_sky,
                                                           exposure_time_map=exposure_map,
                                                           add_noise=True,
                                                           seed=1)

            assert (sim_img.exposure_time_map == 1.0 * np.ones((3, 3))).all()
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

            sim_img = image.Image.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                           exposure_time_map=exposure_map, psf=psf,
                                                           add_noise=False)

            assert (sim_img == np.array([[0.0, 1.0, 0.0],
                                         [1.0, 2.0, 1.0],
                                         [0.0, 1.0, 0.0]])).all()
            assert (sim_img.exposure_time_map == np.ones((3, 3))).all()
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

            sim_img = image.Image.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                           exposure_time_map=exposure_map,
                                                           psf=psf, background_sky_map=background_sky,
                                                           add_noise=False, seed=1)

            assert (sim_img.exposure_time_map == 1.0 * np.ones((3, 3))).all()
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

            sim_img = image.Image.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                           exposure_time_map=exposure_map,
                                                           add_noise=True, seed=1)

            assert (sim_img.exposure_time_map == 20.0 * np.ones((3, 3))).all()
            assert sim_img.pixel_scale == 0.1

            assert sim_img == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 1.05, 0.0],
                                                      [0.0, 0.0, 0.0]]), 1e-2)

            # Because of the regular value is 1.05, the estimated Poisson noise_map_ is:
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

            sim_img = image.Image.simulate_variable_arrays(array=img, pixel_scale=0.1,
                                                           exposure_time_map=exposure_map,
                                                           psf=psf,
                                                           add_noise=True, seed=1)

            assert (sim_img.exposure_time_map == 20.0 * np.ones((3, 3))).all()
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
            sim_img_variable = image.Image.simulate_variable_arrays(array=img,
                                                                    exposure_time_map=exposure_map,
                                                                    psf=psf,
                                                                    background_sky_map=background_sky,
                                                                    pixel_scale=0.1, add_noise=False, seed=1)

            img = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            sim_img = image.Image.simulate(array=img, pixel_scale=0.1, exposure_time=1.0,
                                           background_sky_level=16.0, psf=psf, seed=1)

            assert (sim_img_variable.exposure_time_map == sim_img.exposure_time_map).all()
            assert sim_img_variable.pixel_scale == sim_img.pixel_scale
            assert sim_img_variable == pytest.approx(sim_img, 1e-4)
            assert (sim_img_variable.background_noise_map == sim_img.background_noise_map).all()

        def test__target_signal_to_noise__no_background_sky(self):
            img = np.array([[0.01, 0.02, 0.01],
                            [0.01, 5.0, 0.01],
                            [0.01, 0.01, 0.01]])

            exposure_time = image.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.Image.simulate_to_target_signal_to_noise(array=img, pixel_scale=0.1,
                                                                     target_signal_to_noise=30.0,
                                                                     exposure_time_map=exposure_time,
                                                                     seed=1)

            assert 29.3 < sim_img.signal_to_noise_max < 30.7

        def test__target_signal_to_noise__background_sky_and_poisson(self):
            img = np.array([[0.01, 0.02, 0.01],
                            [0.01, 5.0, 0.01],
                            [0.01, 0.01, 0.01]])

            exposure_time = image.ScaledSquarePixelArray.single_value(value=2.0, pixel_scale=0.1, shape=img.shape)
            background_sky = image.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=img.shape)

            sim_img = image.Image.simulate_to_target_signal_to_noise(array=img, pixel_scale=0.1,
                                                                     target_signal_to_noise=30.0,
                                                                     exposure_time_map=exposure_time,
                                                                     background_sky_map=background_sky,
                                                                     seed=1)

            assert 29.2 < sim_img.signal_to_noise_max < 30.8

    class TestSimulatePoissonNoise(object):

        def test__input_img_all_0s__exposure_time_all_1s__all_noise_values_are_0s(self):

            img = np.zeros((2, 2))

            exposure_time = image.ScaledSquarePixelArray.single_value(1.0, img.shape, pixel_scale=0.1)
            sim_poisson_img = img + image.generate_poisson_noise(img, exposure_time, seed=1)

            assert sim_poisson_img.shape == (2, 2)
            assert (sim_poisson_img == np.zeros((2, 2))).all()


        def test__input_img_includes_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            img = np.array([[10., 0.],
                            [0., 10.]])

            exposure_time = image.ScaledSquarePixelArray.single_value(1.0, img.shape, pixel_scale=0.1)
            poisson_noise_map = image.generate_poisson_noise(img, exposure_time, seed=1)
            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise_map_ map for given seed.
            assert (poisson_noise_map == np.array([[(10.0 - 9.0), 0],
                                                   [0, (10.0 - 6.0)]])).all()
            assert (sim_poisson_img == np.array([[11, 0],
                                                 [0, 14]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()


        def test__input_img_is_all_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            img = np.array([[10., 10.],
                            [10., 10.]])

            exposure_time = image.ScaledSquarePixelArray.single_value(1.0, img.shape, pixel_scale=0.1)
            poisson_noise_map = image.generate_poisson_noise(img, exposure_time, seed=1)
            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise_map_ map for given seed.
            assert (poisson_noise_map == np.array([[1, 4],
                                                   [3, 1]])).all()

            assert (sim_poisson_img == np.array([[11, 14],
                                                 [13, 11]])).all()

            assert (sim_poisson_img - poisson_noise_map == img).all()

        def test__input_img_has_1000000s__exposure_times_is_1s__these_give_positive_noise_values_near_1000(self):
            img = np.array([[10000000., 0.],
                            [0., 10000000.]])

            exposure_time = image.ScaledSquarePixelArray(array=np.ones((2, 2)), pixel_scale=0.1)

            poisson_noise_map = image.generate_poisson_noise(img, exposure_time, seed=2)

            sim_poisson_img = img + poisson_noise_map

            assert sim_poisson_img.shape == (2, 2)

            # Use known noise_map_ map for given seed.
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

            sim_poisson_img_0 = img_0 + image.generate_poisson_noise(img_0, exposure_time_0, seed=1)
            sim_poisson_img_1 = img_1 + image.generate_poisson_noise(img_1, exposure_time_1, seed=1)

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

            sim_poisson_img_0 = img_0 + image.generate_poisson_noise(img_0, exposure_time_0, seed=1)
            sim_poisson_img_1 = img_1 + image.generate_poisson_noise(img_1, exposure_time_1, seed=1)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert (noise_estimate == 0.5 * np.ones((1, 5))).all()

        def test__image_and_exposure_times_range_of_values__no_background__noises_estimates_correct(self):
            array = np.array([[5.0, 3.0],
                              [10.0, 20.0]])

            exposure_time = image.ScaledSquarePixelArray(np.array([[1.0, 2.0],
                                                                   [3.0, 4.0]]), pixel_scale=1.0)

            background_noise = np.zeros((2, 2))

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

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

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                              exposure_time_map=exposure_time, background_noise_map=background_noise)

            noise_estimate = img.estimated_noise_map

            assert noise_estimate == pytest.approx(
                np.array([[np.sqrt(5.0 + 5.0 ** 2.0), np.sqrt(6.0 + 12.0 ** 2.0) / 2.0],
                          [np.sqrt(30.0 + 21.0 ** 2.0) / 3.0,
                           np.sqrt(80.0 + 32.0 ** 2.0) / 4.0]]),
                1e-2)

    class TestEstimateDataGrid(object):

        def test__via_edges__input_all_ones__sky_bg_level_1(self):
            img = image.Image(array=np.ones((3, 3)), noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                              pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):
            image_array = np.array([[1, 1, 1],
                                    [1, 100, 1],
                                    [1, 1, 1]])

            img = image.Image(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                              pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):
            image_array = np.array([[1, 1, 1],
                                    [1, 100, 1],
                                    [1, 100, 1],
                                    [1, 1, 1]])

            img = image.Image(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                              pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image_array = np.array([[1, 1, 1, 1],
                                    [1, 100, 100, 1],
                                    [1, 100, 100, 1],
                                    [1, 1, 1, 1]])

            img = image.Image(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                              pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):
            image_array = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 100, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])

            img = image.Image(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
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

            img = image.Image(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
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

            img = image.Image(array=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                              pixel_scale=0.1)
            sky_noise = img.background_noise_from_edges(no_edges=3)

            assert sky_noise == np.std(np.arange(48))

    class TestSignalToNoise:

        def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):

            array = np.array([[1.0, 2.0],
                              [3.0, 4.0]])

            noise = np.array([[10.0, 10.0],
                              [30.0, 4.0]])

            img = image.Image(array=array, pixel_scale=1.0,
                              psf=image.PSF(array=np.ones((2, 2)), pixel_scale=1.0), noise_map=noise)

            assert (img.signal_to_noise_map == np.array([[0.1, 0.2],
                                                         [0.1, 1.0]])).all()
            assert img.signal_to_noise_max == 1.0

    class TestNewImageResize:

        def test__all_components_resized__psf_is_not(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3, 3] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            noise_map_array[3,3] = 3.0

            background_noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_noise_map_array[3,3] = 4.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            exposure_time_map_array[3,3] = 5.0

            background_sky_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_sky_map_array[3,3] = 6.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                             exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            im = im.new_image_with_resized_arrays(new_shape=(4, 4))

            assert (im == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert (im.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (im.background_noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 4.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (im.exposure_time_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 5.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (im.background_sky_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 6.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()

            assert (im.poisson_noise_map == None)

            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()
            assert im.origin == (0.0, 0.0)

        def test__resize_psf(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0))

            im = im.new_image_with_resized_psf(new_shape=(1, 1))

            assert (im == np.ones((6,6))).all()
            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((1,1))).all()
            assert im.origin == (0.0, 0.0)

        def test__input_new_centre_pixels__arrays_use_new_centre__psf_does_not(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3,3] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            noise_map_array[3,3] = 3.0

            background_noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_noise_map_array[3,3] = 4.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            exposure_time_map_array[3,3] = 5.0

            background_sky_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_sky_map_array[3,3] = 6.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                             exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            im = im.new_image_with_resized_arrays(new_shape=(3, 3), new_centre_pixels=(3, 3))

            assert (im == np.array([[1.0, 1.0, 1.0],
                                    [1.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0]])).all()
            assert (im.noise_map ==  np.array([[1.0, 1.0, 1.0],
                                               [1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()
            assert (im.background_noise_map ==  np.array([[1.0, 1.0, 1.0],
                                                          [1.0, 4.0, 1.0],
                                                          [1.0, 1.0, 1.0]])).all()
            assert (im.exposure_time_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 5.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()
            assert (im.background_sky_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 6.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()

            assert (im.poisson_noise_map == None)

            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()
            assert im.origin == (0.0, 0.0)

        def test__input_new_centre_arc_seconds__arrays_use_new_centre__psf_does_not(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3,3] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            noise_map_array[3,3] = 3.0

            background_noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_noise_map_array[3,3] = 4.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            exposure_time_map_array[3,3] = 5.0

            background_sky_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_sky_map_array[3,3] = 6.0

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                             exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            im = im.new_image_with_resized_arrays(new_shape=(3, 3), new_centre_arc_seconds=(-0.5, 0.5))

            assert (im == np.array([[1.0, 1.0, 1.0],
                                    [1.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0]])).all()
            assert (im.noise_map ==  np.array([[1.0, 1.0, 1.0],
                                               [1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()
            assert (im.background_noise_map ==  np.array([[1.0, 1.0, 1.0],
                                                          [1.0, 4.0, 1.0],
                                                          [1.0, 1.0, 1.0]])).all()
            assert (im.exposure_time_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 5.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()
            assert (im.background_sky_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 6.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()

            assert (im.poisson_noise_map == None)

            assert im.pixel_scale == 1.0
            assert (im.psf == np.zeros((3,3))).all()
            assert im.origin == (0.0, 0.0)

        def test__input_both_centres__raises_error(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0))

            with pytest.raises(exc.ImagingException):
                im.new_image_with_resized_arrays(new_shape=(3, 3), new_centre_pixels=(3, 3),
                                                 new_centre_arc_seconds=(-0.5, 0.5))

    class TestNewImageConvertedFrom:

        def test__counts__all_arrays_in_units_of_flux_are_converted(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((3, 3)), pixel_scale=1.0)
            noise_map_array = scaled_array.ScaledSquarePixelArray(2.0 * np.ones((3, 3)), pixel_scale=1.0)
            background_noise_map_array = scaled_array.ScaledSquarePixelArray(3.0 * np.ones((3, 3)), pixel_scale=1.0)
            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(0.5 * np.ones((3, 3)), pixel_scale=1.0)
            background_sky_map_array = scaled_array.ScaledSquarePixelArray(6.0 * np.ones((3, 3)), pixel_scale=1.0)

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                             poisson_noise_map=None, exposure_time_map=exposure_time_map_array,
                             background_sky_map=background_sky_map_array)

            im = im.new_image_converted_from_electrons()

            assert (im == 2.0*np.ones((3,3))).all()
            assert (im.noise_map == 4.0*np.ones((3,3))).all()
            assert (im.background_noise_map == 6.0*np.ones((3,3))).all()
            assert im.poisson_noise_map == None
            assert (im.background_sky_map == 12.0*np.ones((3,3))).all()
            assert im.origin == (0.0, 0.0)

        def test__adus__all_arrays_in_units_of_flux_are_converted(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((3, 3)), pixel_scale=1.0)
            noise_map_array = scaled_array.ScaledSquarePixelArray(2.0 * np.ones((3, 3)), pixel_scale=1.0)
            background_noise_map_array = scaled_array.ScaledSquarePixelArray(3.0 * np.ones((3, 3)), pixel_scale=1.0)
            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(0.5 * np.ones((3, 3)), pixel_scale=1.0)
            background_sky_map_array = scaled_array.ScaledSquarePixelArray(6.0 * np.ones((3, 3)), pixel_scale=1.0)

            im = image.Image(array=image_array, pixel_scale=1.0, psf=image.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                             noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                             poisson_noise_map=None, exposure_time_map=exposure_time_map_array,
                             background_sky_map=background_sky_map_array)

            im = im.new_image_converted_from_adus(gain=2.0)

            assert (im == 2.0*2.0*np.ones((3,3))).all()
            assert (im.noise_map == 2.0*4.0*np.ones((3,3))).all()
            assert (im.background_noise_map == 2.0*6.0*np.ones((3,3))).all()
            assert im.poisson_noise_map == None
            assert (im.background_sky_map == 2.0*12.0*np.ones((3,3))).all()
            assert im.origin == (0.0, 0.0)

    class TestNewImageWithPoissonNoiseAdded:

        def test__mock_image_all_1s__poisson_noise_is_added_correct(self):

            psf = image.PSF(array=np.ones((3, 3)), pixel_scale=3.0, renormalize=False)
            im = image.Image(np.ones((4, 4)), pixel_scale=3., psf=psf, noise_map=np.ones((4, 4)),
                             exposure_time_map=3.0 * np.ones((4, 4)), background_sky_map=4.0 * np.ones((4, 4)))

            mock_image = np.ones((4, 4))
            mock_image_with_sky = mock_image + 4.0 * np.ones((4, 4))
            mock_image_with_sky_and_noise = mock_image_with_sky + image.generate_poisson_noise(image=mock_image_with_sky,
                                                                                               exposure_time_map=3.0 * np.ones(
                                                                                                (4, 4)), seed=1)

            mock_image_with_noise = mock_image_with_sky_and_noise - 4.0 * np.ones((4, 4))

            im_with_noise = im.new_image_with_poisson_noise_added(seed=1)

            assert (im_with_noise == mock_image_with_noise).all()


class TestNoiseMap(object):

    class TestFromWeightMap:

        def test__weight_map_no_zeros__uses_1_over_sqrt_value(self):

            weight_map = np.array([[1.0, 4.0, 16.0],
                                   [1.0, 4.0, 16.0]])

            noise_map = image.NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.5, 0.25],
                                           [1.0, 0.5, 0.25]])).all()
            assert noise_map.origin == (0.0, 0.0)

        def test__weight_map_no_zeros__zeros_set_to_10000000(self):

            weight_map = np.array([[1.0, 4.0, 0.0],
                                   [1.0, 4.0, 16.0]])

            noise_map = image.NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.5, 1.0e8],
                                           [1.0, 0.5, 0.25]])).all()
            assert noise_map.origin == (0.0, 0.0)

    class TestFromInverseNoiseMap:

        def test__inverse_noise_map_no_zeros__uses_1_over_value(self):

            inverse_noise_map = np.array([[1.0, 4.0, 16.0],
                                          [1.0, 4.0, 16.0]])

            noise_map = image.NoiseMap.from_inverse_noise_map(inverse_noise_map=inverse_noise_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.25, 0.0625],
                                           [1.0, 0.25, 0.0625]])).all()
            assert noise_map.origin == (0.0, 0.0)

    class TestFromImageAndBackgroundNoiseMap:

        def test__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                                           background_noise_map=background_noise_map,
                                                                           gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(2.)], [np.sqrt(2.), np.sqrt(2.)]])).all()

        def test__image_all_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_3s(self):

            im = np.array([[2.0, 2.0], [2.0, 2.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                                           background_noise_map=background_noise_map,
                                                                           gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(3.), np.sqrt(3.)], [np.sqrt(3.), np.sqrt(3.)]])).all()

        def test__image_all_1s__bg_noise_all_2s__exposure_time_all_1s__noise_map_all_sqrt_5s(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[2.0, 2.0], [2.0, 2.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                                           background_noise_map=background_noise_map,
                                                                           gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(5.), np.sqrt(5.)], [np.sqrt(5.), np.sqrt(5.)]])).all()

        def test__image_all_1s__bg_noise_all_1s__exposure_time_all_2s__noise_map_all_sqrt_6s_over_2(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[2.0, 2.0], [2.0, 2.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                                           background_noise_map=background_noise_map,
                                                                           gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(6.) / 2.0, np.sqrt(6.) / 2.0],
                                           [np.sqrt(6.) / 2.0, np.sqrt(6.) / 2.0]])).all()

        def test__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_1s(self):

            im = np.array([[-2.0, -2.0], [-2.0, -2.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                                           background_noise_map=background_noise_map,
                                                                           gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__same_as_above__use_different_values_in_different_array_elemets(self):

            im = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[4.0, 3.0], [2.0, 1.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                                           background_noise_map=background_noise_map,
                                                                           gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(20.) / 4.0, np.sqrt(15.) / 3.0],
                                           [np.sqrt(20.) / 2.0, np.sqrt(12.)]])).all()

        def test__convert_from_electrons__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                        background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                        gain=2.0, convert_from_electrons=True)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(2.)], [np.sqrt(2.), np.sqrt(2.)]])).all()

        def test__convert_from_electrons__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_10s__noise_map_all_1s(self):

            im = np.array([[-2.0, -2.0], [-2.0, -2.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[10.0, 10.0], [10.0, 10.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                gain=1.0, convert_from_electrons=True)

            assert (noise_map == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__convert_from_electrons__same_as_above__use_different_values_in_different_array_elemets(self):

            im = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[10.0, 11.0], [12.0, 13.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                      background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                    gain=4.0, convert_from_electrons=True)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(3.)], [np.sqrt(6.), np.sqrt(12.)]])).all()

        def test__convert_from_adus__same_as_above__gain_is_1__same_values(self):

            im = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[10.0, 11.0], [12.0, 13.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                      background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                    gain=1.0, convert_from_adus=True)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(3.)], [np.sqrt(6.), np.sqrt(12.)]])).all()

        def test__convert_from_adus__same_as_above__gain_is_2__values_change(self):

            im = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[10.0, 11.0], [12.0, 13.0]])

            noise_map = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                      background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                    gain=2.0, convert_from_adus=True)

            assert (noise_map == np.array([[np.sqrt(6.) / 2.0, np.sqrt(8.) / 2.0],
                                           [np.sqrt(20.) / 2.0, np.sqrt(42.) / 2.0]])).all()


class TestPoissonNoiseMap(object):

    class TestFromImageAndExposureTimeMap:

        def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=im,
                                             exposure_time_map=exposure_time_map, gain=1.0)

            assert (poisson_noise_map == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__image_all_2s_and_3s__exposure_time_all_1s__noise_map_all_sqrt_2s_and_3s(self):

            im = np.array([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=im,
                                          exposure_time_map=exposure_time_map, gain=1.0)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]])).all()

        def test__image_all_1s__exposure_time_all__2s_and_3s__noise_map_all_sqrt_2s_and_3s(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[2.0, 2.0], [3.0, 3.0]])

            poisson_noise_map = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=im,
                                        exposure_time_map=exposure_time_map, gain=1.0)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
                                                   [np.sqrt(3.0) / 3.0, np.sqrt(3.0) / 3.0]])).all()

        def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s__gain_is_2__ignores_gain(self):

            im = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=im,
                                             exposure_time_map=exposure_time_map, gain=2.0)

            assert (poisson_noise_map == np.array([[np.sqrt(1.0), np.sqrt(1.0)],
                                                   [np.sqrt(1.0), np.sqrt(1.0)]])).all()

        def test__convert_from_electrons_is_true__image_already_in_counts_so_exposure_time_ignored(self):

            im = np.array([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = np.array([[10.0, 10.0], [10.0, 10.0]])

            poisson_noise_map = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=im,
                                             exposure_time_map=exposure_time_map, gain=4.0,
                                             convert_from_electrons=True)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]])).all()

        def test__same_as_above__convert_from_adus__includes_gain_multiplication(self):

            im = np.array([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = np.array([[10.0, 10.0], [10.0, 10.0]])

            poisson_noise_map = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=im,
                                             exposure_time_map=exposure_time_map, gain=2.0,
                                             convert_from_adus=True)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0*2.0) / 2.0, np.sqrt(2.0*2.0) / 2.0],
                                                   [np.sqrt(2.0*3.0) / 2.0, np.sqrt(2.0*3.0) / 2.0]])).all()


class TestPSF(object):

    class TestConstructors(object):

        def test__init__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF(array=np.ones((3, 3)), pixel_scale=1.0, renormalize=False)

            assert psf.shape == (3, 3)
            assert psf.pixel_scale == 1.0
            assert (psf == np.ones((3, 3))).all()
            assert psf.origin == (0.0, 0.0)

        def test__init__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF(array=np.ones((4, 3)), pixel_scale=1.0, renormalize=False)

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.shape == (4, 3)
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF.from_fits_with_scale(file_path=test_data_dir + '3x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (psf == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = image.PSF.from_fits_with_scale(file_path=test_data_dir + '4x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

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

            from autolens.model.profiles import light_profiles as lp

            grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full((3, 3), False),
                                                                                      pixel_scales=(1.0, 1.0))

            gaussian = lp.EllipticalGaussian(centre=(0.1, 0.1), axis_ratio=0.9, phi=45.0, intensity=1.0, sigma=1.0)
            profile_gaussian_1d = gaussian.intensities_from_grid(grid)
            profile_gaussian_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
                array_1d=profile_gaussian_1d, shape=(3, 3))
            profile_psf = image.PSF(array=profile_gaussian_2d, pixel_scale=1.0, renormalize=True)

            imaging_psf = image.PSF.simulate_as_gaussian(shape=(3, 3), pixel_scale=1.0, centre=(0.1, 0.1),
                                                         axis_ratio=0.9, phi=45.0, sigma=1.0)

            assert profile_psf == pytest.approx(imaging_psf, 1e-4)

    class TestSimulateAsAlmaGaussian(object):

        def test__identical_to_astropy_gaussian_model__circular_no_rotation(self):

            pixel_scale = 0.1

            x_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            y_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            gaussian_astropy = functional_models.Gaussian2D(amplitude=1.0, x_mean=2.0, y_mean=2.0,
                                                            x_stddev=x_stddev, y_stddev=y_stddev, theta=0.0)

            shape = (5, 5)
            y, x = np.mgrid[0:shape[1], 0:shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = image.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                        y_stddev=2.0e-5, x_stddev=2.0e-5, theta=0.0)

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__circular_no_rotation_different_pixel_scale(self):

            pixel_scale = 0.02

            x_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            y_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            gaussian_astropy = functional_models.Gaussian2D(amplitude=1.0, x_mean=2.0, y_mean=2.0,
                                                            x_stddev=x_stddev, y_stddev=y_stddev, theta=0.0)

            shape = (5, 5)
            y, x = np.mgrid[0:shape[1], 0:shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = image.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                        y_stddev=2.0e-5, x_stddev=2.0e-5, theta=0.0)

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_ellipticity_from_x_and_y_stddev(self):

            pixel_scale = 0.1

            x_stddev = 1.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            y_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            theta_deg=0.0
            theta=Angle(theta_deg, 'deg').radian

            gaussian_astropy = functional_models.Gaussian2D(amplitude=1.0, x_mean=2.0, y_mean=2.0,
                                                            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)

            shape = (5, 5)
            y, x = np.mgrid[0:shape[1], 0:shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = image.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                        y_stddev=2.0e-5, x_stddev=1.0e-5, theta=theta_deg)

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_different_ellipticity_from_x_and_y_stddev(self):

            pixel_scale = 0.1

            x_stddev = 3.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            y_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            theta_deg=0.0
            theta=Angle(theta_deg, 'deg').radian

            gaussian_astropy = functional_models.Gaussian2D(amplitude=1.0, x_mean=2.0, y_mean=2.0,
                                                            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)

            shape = (5, 5)
            y, x = np.mgrid[0:shape[1], 0:shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = image.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                        y_stddev=2.0e-5, x_stddev=3.0e-5, theta=theta_deg)

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_rotation_angle_30(self):

            pixel_scale = 0.1

            x_stddev = 1.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            y_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            theta_deg=30.0
            theta=Angle(theta_deg, 'deg').radian

            gaussian_astropy = functional_models.Gaussian2D(amplitude=1.0, x_mean=1.0, y_mean=1.0,
                                                            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)

            shape = (3, 3)
            y, x = np.mgrid[0:shape[1], 0:shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = image.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                        y_stddev=2.0e-5, x_stddev=1.0e-5, theta=theta_deg)

            assert psf_astropy == pytest.approx(psf, 1e-4)

        def test__identical_to_astropy_gaussian_model__include_rotation_angle_230(self):

            pixel_scale = 0.1

            x_stddev = 1.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            y_stddev = 2.0e-5 * (units.deg).to(units.arcsec) / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            theta_deg=230.0
            theta=Angle(theta_deg, 'deg').radian

            gaussian_astropy = functional_models.Gaussian2D(amplitude=1.0, x_mean=1.0, y_mean=1.0,
                                                            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta)

            shape = (3, 3)
            y, x = np.mgrid[0:shape[1], 0:shape[0]]
            psf_astropy = gaussian_astropy(x, y)
            psf_astropy /= np.sum(psf_astropy)

            psf = image.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                        y_stddev=2.0e-5, x_stddev=1.0e-5, theta=theta_deg)

            assert psf_astropy == pytest.approx(psf, 1e-4)


class TestExposureTimeMap(object):

    class TestFromExposureTimeAndBackgroundNoiseMap:

        def test__from_background_noise_map__covnerts_to_exposure_times(self):

            background_noise_map = np.array([[1.0, 4.0, 8.0],
                                             [1.0, 4.0, 8.0]])

            exposure_time_map = image.ExposureTimeMap.from_exposure_time_and_background_noise_map(pixel_scale=0.1,
                                                                                                  exposure_time=1.0, background_noise_map=background_noise_map)

            assert (exposure_time_map == np.array([[0.125, 0.5, 1.0],
                                                   [0.125, 0.5, 1.0]])).all()
            assert exposure_time_map.origin == (0.0, 0.0)

            exposure_time_map = image.ExposureTimeMap.from_exposure_time_and_background_noise_map(pixel_scale=0.1,
                                                                                                  exposure_time=3.0, background_noise_map=background_noise_map)

            assert (exposure_time_map == np.array([[0.375, 1.5, 3.0],
                                                   [0.375, 1.5, 3.0]])).all()
            assert exposure_time_map.origin == (0.0, 0.0)


class TestLoadImagingFromFits(object):

    def test__no_settings_just_pass_fits(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          renormalize_psf=False)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert im.background_noise_map == None
        assert im.poisson_noise_map == None
        assert im.exposure_time_map == None
        assert im.background_sky_map == None

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1

    def test__optional_array_paths_included__loads_optional_array(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__all_files_in_one_fits__load_using_different_hdus(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_multiple_hdu.fits', image_hdu=0,
                                          pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_multiple_hdu.fits', psf_hdu=1,
                                          noise_map_path=test_data_dir + '3x3_multiple_hdu.fits', noise_map_hdu=2,
                                          background_noise_map_path=test_data_dir + '3x3_multiple_hdu.fits',
                                          background_noise_map_hdu=3,
                                          poisson_noise_map_path=test_data_dir + '3x3_multiple_hdu.fits',
                                          poisson_noise_map_hdu=4,
                                          exposure_time_map_path=test_data_dir + '3x3_multiple_hdu.fits',
                                          exposure_time_map_hdu=5,
                                          background_sky_map_path = test_data_dir + '3x3_multiple_hdu.fits',
                                          background_sky_map_hdu=6,
                                          renormalize_psf=False)


        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                          noise_map_path=test_data_dir + '3x3_ones.fits',
                                          psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          exposure_time_map_from_single_value=3.0,
                                          renormalize_psf=False)

        assert (im.exposure_time_map == 3.0*np.ones((3,3))).all()

    def test__exposure_time_map_from_background_noise_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          exposure_time_map_from_single_value=3.0,
                                          exposure_time_map_from_background_noise_map=True,
                                          renormalize_psf=False)

        assert (im.exposure_time_map == 3.0*np.ones((3,3))).all()

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          exposure_time_map_from_single_value=6.0,
                                          exposure_time_map_from_background_noise_map=True,
                                          renormalize_psf=False)

        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()

    def test__pad_shape_of_image_arrays_and_psf(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          resized_image_shape=(5,5), resized_psf_shape=(7,7),
                                          renormalize_psf=False)

        padded_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 1.0, 1.0, 0.0],
                                 [0.0, 1.0, 1.0, 1.0, 0.0],
                                 [0.0, 1.0, 1.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0]])

        psf_padded_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                                     [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                                     [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        assert (im == padded_array).all()
        assert (im.psf == psf_padded_array).all()
        assert (im.noise_map == 3.0*padded_array).all()
        assert (im.background_noise_map == 4.0 * padded_array).all()
        assert (im.poisson_noise_map == 5.0 * padded_array).all()
        assert (im.exposure_time_map == 6.0 * padded_array).all()
        assert (im.background_sky_map == 7.0 * padded_array).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          exposure_time_map_from_single_value=3.0, pixel_scale=0.1,
                                          resized_image_shape=(5, 5), resized_psf_shape=(7, 7),
                                          renormalize_psf=False)

        exposure_padded_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 3.0, 3.0, 3.0, 0.0],
                                 [0.0, 3.0, 3.0, 3.0, 0.0],
                                 [0.0, 3.0, 3.0, 3.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0]])

        assert (im == padded_array).all()
        assert (im.exposure_time_map == exposure_padded_array).all()

    def test__trim_shape_of_image_arrays_and_psf(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          resized_image_shape=(1,1), resized_psf_shape=(1,1),
                                          renormalize_psf=False)

        trimmed_array = np.array([[1.0]])

        assert (im == trimmed_array).all()
        assert (im.psf == 2.0*trimmed_array).all()
        assert (im.noise_map == 3.0*trimmed_array).all()
        assert (im.background_noise_map == 4.0 * trimmed_array).all()
        assert (im.poisson_noise_map == 5.0 * trimmed_array).all()
        assert (im.exposure_time_map == 6.0 * trimmed_array).all()
        assert (im.background_sky_map == 7.0 * trimmed_array).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_noise_map_from_weight_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          convert_noise_map_from_weight_map=True,
                                          renormalize_psf=False)

        noise_map_converted = image.NoiseMap.from_weight_map(weight_map=3.0 * np.ones((3, 3)), pixel_scale=0.1)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == noise_map_converted).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_noise_map_from_inverse_noise_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          convert_noise_map_from_inverse_noise_map=True,
                                          renormalize_psf=False)

        noise_map_converted = image.NoiseMap.from_inverse_noise_map(inverse_noise_map=3.0 * np.ones((3, 3)),
                                                                    pixel_scale=0.1)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == noise_map_converted).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_from_image_and_background_noise_map=True,
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False)

        noise_map_converted = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=im,
                                                              background_noise_map=im.background_noise_map,
                                                              gain=2.0, exposure_time_map=im.exposure_time_map)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == noise_map_converted).all()
        assert (im.noise_map == (np.sqrt((24.0)**2.0 + (6.0))/(6.0))*np.ones((3,3)))
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map__include_convert_from_electrons(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_from_image_and_background_noise_map=True,
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          convert_from_electrons=True,
                                          renormalize_psf=False)

        noise_map_converted = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1,
            image=1.0*np.ones((3,3)), background_noise_map=4.0*np.ones((3,3)),
            gain=None, exposure_time_map=im.exposure_time_map, convert_from_electrons=True)

        noise_map_converted = noise_map_converted / 6.0

        assert (im == np.ones((3,3)) / 6.0).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == noise_map_converted).all()
        assert (im.noise_map == np.sqrt(17.0)*np.ones((3,3)) / 6.0).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3)) / 6.0).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3)) / 6.0).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3)) / 6.0).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map__include_convert_from_adus(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_from_image_and_background_noise_map=True,
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          gain=2.0, convert_from_adus=True,
                                          renormalize_psf=False)

        noise_map_converted = image.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1,
            image=1.0*np.ones((3,3)), background_noise_map=4.0*np.ones((3,3)),
            gain=2.0, exposure_time_map=im.exposure_time_map, convert_from_adus=True)

        noise_map_converted = 2.0 * noise_map_converted / 6.0

        assert (im == 2.0*np.ones((3,3)) / 6.0).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == noise_map_converted).all()
        assert (im.noise_map == np.sqrt(66.0)*np.ones((3,3)) / 6.0).all()
        assert (im.background_noise_map == 2.0*4.0*np.ones((3,3)) / 6.0).all()
        assert (im.poisson_noise_map == 2.0*5.0*np.ones((3,3)) / 6.0).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 2.0*7.0*np.ones((3,3)) / 6.0).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_background_noise_map_from_weight_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          convert_background_noise_map_from_weight_map=True)

        background_noise_map_converted = image.NoiseMap.from_weight_map(weight_map=4.0 * np.ones((3, 3)), pixel_scale=0.1)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == background_noise_map_converted).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_background_noise_map_from_inverse_noise_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          convert_background_noise_map_from_inverse_noise_map=True)

        background_noise_map_converted = image.NoiseMap.from_inverse_noise_map(inverse_noise_map=4.0 * np.ones((3, 3)),
                                                                               pixel_scale=0.1)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == background_noise_map_converted).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__poisson_noise_map_from_image(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          poisson_noise_map_from_image=True)

        poisson_noise_map_converted = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1,
                    image=np.ones((3,3)), exposure_time_map=im.exposure_time_map, gain=None)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == (np.sqrt(6.0)/(6.0))*np.ones((3,3)))
        assert (im.poisson_noise_map == poisson_noise_map_converted).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__poisson_noise_map_from_image__include_convert_from_electrons(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          poisson_noise_map_from_image=True, convert_from_electrons=True)

        poisson_noise_map_counts = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1,
                     image=np.ones((3,3)), exposure_time_map=im.exposure_time_map, gain=None, convert_from_electrons=True)

        poisson_noise_map_converted = poisson_noise_map_counts / 6.0

        assert (im == np.ones((3,3)) / 6.0).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3)) / 6.0).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3)) / 6.0).all()
        assert (im.poisson_noise_map == np.ones((3,3)) / 6.0)
        assert (im.poisson_noise_map == poisson_noise_map_converted).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3)) / 6.0).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__poisson_noise_map_from_image__include_convert_from_adus(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          poisson_noise_map_from_image=True, gain=2.0, convert_from_adus=True)

        poisson_noise_map_counts = image.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1,
                 image=np.ones((3,3)), exposure_time_map=im.exposure_time_map, gain=2.0, convert_from_adus=True)

        poisson_noise_map_converted = 2.0 * poisson_noise_map_counts / 6.0

        assert (im == 2.0*np.ones((3,3)) / 6.0).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 2.0*3.0*np.ones((3,3)) / 6.0).all()
        assert (im.background_noise_map == 2.0*4.0*np.ones((3,3)) / 6.0).all()
        assert (im.poisson_noise_map == np.sqrt(2.0*np.ones((3,3))) / 6.0)
        assert (im.poisson_noise_map == poisson_noise_map_converted).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 2.0*7.0*np.ones((3,3)) / 6.0).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_poisson_noise_map_from_weight_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          convert_poisson_noise_map_from_weight_map=True)

        poisson_noise_map_converted = image.NoiseMap.from_weight_map(weight_map=5.0 * np.ones((3, 3)), pixel_scale=0.1)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == poisson_noise_map_converted).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_poisson_noise_map_from_inverse_noise_map(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          convert_poisson_noise_map_from_inverse_noise_map=True)

        poisson_noise_map_converted = image.NoiseMap.from_inverse_noise_map(inverse_noise_map=5.0 * np.ones((3, 3)),
                                                                            pixel_scale=0.1)

        assert (im == np.ones((3, 3))).all()
        assert (im.psf == 2.0 * np.ones((3, 3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == poisson_noise_map_converted).all()
        assert (im.exposure_time_map == 6.0 * np.ones((3, 3))).all()
        assert (im.background_sky_map == 7.0 * np.ones((3, 3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__psf_renormalized_true__renormalized_psf(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=True)

        assert (im == np.ones((3,3))).all()
        assert im.psf == pytest.approx((1.0/9.0)*np.ones((3,3)), 1e-2)
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_image_from_electrons_using_exposure_time(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          convert_from_electrons=True)

        assert (im == np.ones((3,3)) / 6.0).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3)) / 6.0).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3)) / 6.0).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3)) / 6.0).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3)) / 6.0).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__convert_image_from_adus_using_exposure_time_and_gain(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False,
                                          gain=2.0, convert_from_adus=True)

        assert (im == 2.0*np.ones((3,3)) / 6.0).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 2.0*3.0*np.ones((3,3)) / 6.0).all()
        assert (im.background_noise_map == 2.0*4.0*np.ones((3,3)) / 6.0).all()
        assert (im.poisson_noise_map == 2.0*5.0*np.ones((3,3)) / 6.0).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 2.0*7.0*np.ones((3,3)) / 6.0).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1

    def test__no_noise_map_input__raises_imaging_exception(self):

        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         psf_path=test_data_dir + '3x3_twos.fits')

    def test__multiple_noise_map_options__raises_imaging_exception(self):

        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         psf_path=test_data_dir + '3x3_twos.fits',
                                         noise_map_path=test_data_dir + '3x3_threes.fits',
                                         convert_noise_map_from_inverse_noise_map=True,
                                         convert_noise_map_from_weight_map=True)

        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         psf_path=test_data_dir + '3x3_twos.fits',
                                         noise_map_path=test_data_dir + '3x3_threes.fits',
                                         convert_noise_map_from_inverse_noise_map=True,
                                         noise_map_from_image_and_background_noise_map=True)

        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         psf_path=test_data_dir + '3x3_twos.fits',
                                         noise_map_path=test_data_dir + '3x3_threes.fits',
                                         noise_map_from_image_and_background_noise_map=True,
                                         convert_noise_map_from_weight_map=True)

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):

        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                         psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         noise_map_path=test_data_dir + '3x3_threes.fits',
                                         exposure_time_map_path=test_data_dir + '3x3_ones.fits',
                                         exposure_time_map_from_single_value=1.0)

    def test__noise_map_from_image_and_background_noise_map_exceptions(self):

        # need background noise map - raise error if not present
        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                             psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                             noise_map_path=test_data_dir + '3x3_threes.fits',
                                             exposure_time_map_from_single_value=1.0,
                                             noise_map_from_image_and_background_noise_map=True)

        # Dont need gain if image is in electrons
        image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                     psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                     noise_map_path=test_data_dir + '3x3_threes.fits',
                                     background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                     exposure_time_map_from_single_value=1.0,
                                     noise_map_from_image_and_background_noise_map=True,
                                     convert_from_electrons=True)

        # Need gain if image is in adus
        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                         psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         noise_map_path=test_data_dir + '3x3_threes.fits',
                                         background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                         noise_map_from_image_and_background_noise_map=True,
                                         convert_from_adus=True)

        # No error if data already in adus
        image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                     psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                     noise_map_path=test_data_dir + '3x3_threes.fits',
                                     background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                     exposure_time_map_from_single_value=1.0,
                                     noise_map_from_image_and_background_noise_map=True,
                                     gain=1.0,
                                     convert_from_adus=True)

    def test__poisson_noise_map_from_image_exceptions(self):

        # Dont need gain if image is in e/s
        image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                     psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                     noise_map_path=test_data_dir + '3x3_threes.fits',
                                     exposure_time_map_from_single_value=1.0,
                                     poisson_noise_map_from_image=True)

        # No exposure time - not load
        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                     psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                     poisson_noise_map_from_image=True,
                                     convert_from_electrons=True)

        # Need gain if data in adus
        with pytest.raises(exc.ImagingException):
            image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                         psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                         noise_map_path=test_data_dir + '3x3_threes.fits',
                                         background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                         exposure_time_map_from_single_value=1.0,
                                         poisson_noise_map_from_image=True,
                                         convert_from_adus=True)


class TestOutputImagingToFits(object):

    def test__output_all_arrays(self):

        im = image.load_imaging_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                          psf_path=test_data_dir + '3x3_twos.fits',
                                          noise_map_path=test_data_dir + '3x3_threes.fits',
                                          background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                          poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                          exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                          background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                          renormalize_psf=False)

        output_data_dir = "{}/../test_files/array/output_test/".format(os.path.dirname(os.path.realpath(__file__)))
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        image.output_imaging_to_fits(image=im,
                                     image_path=output_data_dir + 'regular.fits',
                                     psf_path=output_data_dir + 'psf.fits',
                                     noise_map_path=output_data_dir + 'noise_map.fits',
                                     background_noise_map_path=output_data_dir + 'background_noise_map.fits',
                                     poisson_noise_map_path=output_data_dir + 'poisson_noise_map.fits',
                                     exposure_time_map_path=output_data_dir + 'exposure_time_map.fits',
                                     background_sky_map_path=output_data_dir + 'background_sky_map.fits')

        im = image.load_imaging_from_fits(image_path=output_data_dir + 'regular.fits', pixel_scale=0.1,
                                          psf_path=output_data_dir + 'psf.fits',
                                          noise_map_path=output_data_dir + 'noise_map.fits',
                                          background_noise_map_path=output_data_dir + 'background_noise_map.fits',
                                          poisson_noise_map_path=output_data_dir + 'poisson_noise_map.fits',
                                          exposure_time_map_path=output_data_dir + 'exposure_time_map.fits',
                                          background_sky_map_path=output_data_dir + 'background_sky_map.fits',
                                          renormalize_psf=False)

        assert (im == np.ones((3,3))).all()
        assert (im.psf == 2.0*np.ones((3,3))).all()
        assert (im.noise_map == 3.0*np.ones((3,3))).all()
        assert (im.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (im.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (im.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (im.background_sky_map == 7.0*np.ones((3,3))).all()

        assert im.pixel_scale == 0.1
        assert im.psf.pixel_scale == 0.1
        assert im.noise_map.pixel_scale == 0.1
        assert im.background_noise_map.pixel_scale == 0.1
        assert im.poisson_noise_map.pixel_scale == 0.1
        assert im.exposure_time_map.pixel_scale == 0.1
        assert im.background_sky_map.pixel_scale == 0.1