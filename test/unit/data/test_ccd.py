import os
import shutil

import numpy as np
import pytest
from astropy.modeling import functional_models
from astropy import units
from astropy.coordinates import Angle

from autolens import exc
from autolens.data.array import scaled_array
from autolens.data import ccd
from autolens.data.array.util import array_util
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))
test_positions_dir = "{}/../test_files/positions/".format(os.path.dirname(os.path.realpath(__file__)))


class TestCCDData:

    class TestConstructor:

        def test__setup_image__correct_attributes(self):

            array = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            psf = ccd.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)
            noise_map = 5.0 * np.ones((3, 3))

            ccd_data = ccd.CCDData(image=array, pixel_scale=0.1, noise_map=noise_map, psf=psf,
                                   background_noise_map=7.0 * np.ones((3, 3)),
                                   poisson_noise_map=9.0 * np.ones((3, 3)),
                                   exposure_time_map=11.0 * np.ones((3, 3)))

            assert ccd_data.image == pytest.approx(np.array([[1.0, 2.0, 3.0],
                                                 [4.0, 5.0, 6.0],
                                                 [7.0, 8.0, 9.0]]), 1e-2)
            assert (ccd_data.psf == 3.0 * np.ones((3, 3))).all()
            assert (ccd_data.noise_map == 5.0 * np.ones((3, 3))).all()
            assert (ccd_data.background_noise_map == 7.0 * np.ones((3, 3))).all()
            assert (ccd_data.poisson_noise_map == 9.0 * np.ones((3, 3))).all()
            assert (ccd_data.exposure_time_map == 11.0 * np.ones((3, 3))).all()
            assert ccd_data.origin == (0.0, 0.0)

    class TestSimulateCCD(object):

        def test__setup_with_all_features_off(self):

            image = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, exposure_time_map=exposure_map,
                                                                 pixel_scale=0.1, add_noise=False)

            assert (ccd_simulated.exposure_time_map == np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1
            assert (ccd_simulated.image == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()
            assert ccd_simulated.origin == (0.0, 0.0)

        def test__setup_with_background_sky_on__noise_off__no_noise_in_image(self):
            image = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

            background_sky = scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, pixel_scale=0.1,
                                                                 exposure_time_map=exposure_map,
                                                                 background_sky_map=background_sky,
                                                                 add_noise=False,
                                                                 seed=1)

            assert (ccd_simulated.exposure_time_map == 1.0 * np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1

            assert (ccd_simulated.image == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()

            assert (ccd_simulated.background_noise_map == 4.0 * np.ones((3, 3))).all()

        def test__setup_with_background_sky_on__noise_on_so_background_adds_noise_to_image(self):
            image = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

            background_sky = scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, pixel_scale=0.1,
                                                                 background_sky_map=background_sky,
                                                                 exposure_time_map=exposure_map,
                                                                 add_noise=True,
                                                                 seed=1)

            assert (ccd_simulated.exposure_time_map == 1.0 * np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1

            assert (ccd_simulated.image == np.array([[1.0, 5.0, 4.0],
                                         [1.0, 2.0, 1.0],
                                         [5.0, 2.0, 7.0]])).all()

            assert (ccd_simulated.poisson_noise_map == np.array([[np.sqrt(1.0), np.sqrt(5.0), np.sqrt(4.0)],
                                                           [np.sqrt(1.0), np.sqrt(2.0), np.sqrt(1.0)],
                                                           [np.sqrt(5.0), np.sqrt(2.0), np.sqrt(7.0)]])).all()

            assert (ccd_simulated.background_noise_map == 4.0 * np.ones((3, 3))).all()

        def test__setup_with_psf_blurring_on__blurs_image_and_trims_psf_edge_off(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = ccd.PSF(array=np.array([[0.0, 1.0, 0.0],
                                          [1.0, 2.0, 1.0],
                                          [0.0, 1.0, 0.0]]), pixel_scale=1.0)

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, pixel_scale=0.1,
                                                                 exposure_time_map=exposure_map, psf=psf,
                                                                 add_noise=False)

            assert (ccd_simulated.image == np.array([[0.0, 1.0, 0.0],
                                         [1.0, 2.0, 1.0],
                                         [0.0, 1.0, 0.0]])).all()
            assert (ccd_simulated.exposure_time_map == np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1

        def test__setup_with_background_sky_and_psf_on__psf_does_no_blurring__image_and_sky_both_trimmed(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = ccd.PSF(array=np.array([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]]), pixel_scale=1.0)

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

            background_sky = scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, pixel_scale=0.1,
                                                                 exposure_time_map=exposure_map,
                                                                 psf=psf, background_sky_map=background_sky,
                                                                 add_noise=False, seed=1)

            assert (ccd_simulated.exposure_time_map == 1.0 * np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1

            assert (ccd_simulated.image == np.array([[0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0]])).all()

            assert (ccd_simulated.background_noise_map == 4.0 * np.ones((3, 3))).all()

        def test__setup_with_noise(self):
            image = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, pixel_scale=0.1,
                                                                 exposure_time_map=exposure_map,
                                                                 add_noise=True, seed=1)

            assert (ccd_simulated.exposure_time_map == 20.0 * np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1

            assert ccd_simulated.image == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                      [0.0, 1.05, 0.0],
                                                      [0.0, 0.0, 0.0]]), 1e-2)

            # Because of the regular value is 1.05, the estimated Poisson noise_map_1d is:
            # sqrt((1.05 * 20))/20 = 0.2291

            assert ccd_simulated.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                        [0.0, 0.2291, 0.0],
                                                                        [0.0, 0.0, 0.0]]), 1e-2)

            assert ccd_simulated.noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                 [0.0, 0.2291, 0.0],
                                                                 [0.0, 0.0, 0.0]]), 1e-2)

        def test__setup_with__psf_blurring_and_poisson_noise_on__poisson_noise_added_to_blurred_image(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = ccd.PSF(array=np.array([[0.0, 1.0, 0.0],
                                          [1.0, 2.0, 1.0],
                                          [0.0, 1.0, 0.0]]), pixel_scale=1.0)

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_variable_arrays(array=image, pixel_scale=0.1,
                                                                 exposure_time_map=exposure_map,
                                                                 psf=psf,
                                                                 add_noise=True, seed=1)

            assert (ccd_simulated.exposure_time_map == 20.0 * np.ones((3, 3))).all()
            assert ccd_simulated.pixel_scale == 0.1
            assert ccd_simulated.image == pytest.approx(np.array([[0.0, 1.05, 0.0],
                                                      [1.3, 2.35, 1.05],
                                                      [0.0, 1.05, 0.0]]), 1e-2)

            # The estimated Poisson noises are:
            # sqrt((2.35 * 20))/20 = 0.3427
            # sqrt((1.3 * 20))/20 = 0.2549
            # sqrt((1.05 * 20))/20 = 0.2291

            assert ccd_simulated.poisson_noise_map == pytest.approx(np.array([[0.0, 0.2291, 0.0],
                                                                        [0.2549, 0.3427, 0.2291],
                                                                        [0.0, 0.2291, 0.0]]), 1e-2)

        def test__simulate_function__turns_exposure_time_and_sky_level_to_arrays(self):
            
            image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            psf = ccd.PSF(array=np.array([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]]), pixel_scale=1.0)

            exposure_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)
            background_sky = scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)
            simulated_ccd_variable = ccd.CCDData.simulate_variable_arrays(array=image,
                                                                          exposure_time_map=exposure_map, psf=psf,
                                                                          background_sky_map=background_sky,
                                                                          pixel_scale=0.1, add_noise=False, seed=1)

            image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

            ccd_simulated = ccd.CCDData.simulate(array=image, pixel_scale=0.1, exposure_time=1.0,
                                                 background_sky_level=16.0, psf=psf, seed=1)

            assert (simulated_ccd_variable.exposure_time_map == ccd_simulated.exposure_time_map).all()
            assert simulated_ccd_variable.pixel_scale == ccd_simulated.pixel_scale
            assert simulated_ccd_variable.image == pytest.approx(ccd_simulated.image, 1e-4)
            assert (simulated_ccd_variable.background_noise_map == ccd_simulated.background_noise_map).all()

        def test__target_signal_to_noise__no_background_sky(self):
            image = np.array([[0.01, 0.02, 0.01],
                            [0.01, 5.0, 0.01],
                            [0.01, 0.01, 0.01]])

            exposure_time = scaled_array.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_to_target_signal_to_noise(array=image, pixel_scale=0.1,
                                                                           target_signal_to_noise=30.0,
                                                                           exposure_time_map=exposure_time,
                                                                           seed=1)

            assert 29.3 < ccd_simulated.signal_to_noise_max < 30.7

        def test__target_signal_to_noise__background_sky_and_poisson(self):
            image = np.array([[0.01, 0.02, 0.01],
                            [0.01, 5.0, 0.01],
                            [0.01, 0.01, 0.01]])

            exposure_time = scaled_array.ScaledSquarePixelArray.single_value(value=2.0, pixel_scale=0.1, shape=image.shape)
            background_sky = scaled_array.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=image.shape)

            ccd_simulated = ccd.CCDData.simulate_to_target_signal_to_noise(array=image, pixel_scale=0.1,
                                                                           target_signal_to_noise=30.0,
                                                                           exposure_time_map=exposure_time,
                                                                           background_sky_map=background_sky,
                                                                           seed=1)

            assert 29.2 < ccd_simulated.signal_to_noise_max < 30.8

    class TestSimulatePoissonNoise(object):

        def test__input_image_all_0s__exposure_time_all_1s__all_noise_values_are_0s(self):

            image = np.zeros((2, 2))

            exposure_time = scaled_array.ScaledSquarePixelArray.single_value(1.0, image.shape, pixel_scale=0.1)
            simulated_poisson_image = image + ccd.generate_poisson_noise(image, exposure_time, seed=1)

            assert simulated_poisson_image.shape == (2, 2)
            assert (simulated_poisson_image == np.zeros((2, 2))).all()

        def test__input_image_includes_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            image = np.array([[10., 0.],
                            [0., 10.]])

            exposure_time = scaled_array.ScaledSquarePixelArray.single_value(1.0, image.shape, pixel_scale=0.1)
            poisson_noise_map = ccd.generate_poisson_noise(image, exposure_time, seed=1)
            simulated_poisson_image = image + poisson_noise_map

            assert simulated_poisson_image.shape == (2, 2)

            # Use known noise_map_1d map for given seed.
            assert (poisson_noise_map == np.array([[(10.0 - 9.0), 0],
                                                   [0, (10.0 - 6.0)]])).all()
            assert (simulated_poisson_image == np.array([[11, 0],
                                                 [0, 14]])).all()

            assert (simulated_poisson_image - poisson_noise_map == image).all()

        def test__input_image_is_all_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
            image = np.array([[10., 10.],
                            [10., 10.]])

            exposure_time = scaled_array.ScaledSquarePixelArray.single_value(1.0, image.shape, pixel_scale=0.1)
            poisson_noise_map = ccd.generate_poisson_noise(image, exposure_time, seed=1)
            simulated_poisson_image = image + poisson_noise_map

            assert simulated_poisson_image.shape == (2, 2)

            # Use known noise_map_1d map for given seed.
            assert (poisson_noise_map == np.array([[1, 4],
                                                   [3, 1]])).all()

            assert (simulated_poisson_image == np.array([[11, 14],
                                                 [13, 11]])).all()

            assert (simulated_poisson_image - poisson_noise_map == image).all()

        def test__input_image_has_1000000s__exposure_times_is_1s__these_give_positive_noise_values_near_1000(self):
            image = np.array([[10000000., 0.],
                            [0., 10000000.]])

            exposure_time = scaled_array.ScaledSquarePixelArray(array=np.ones((2, 2)), pixel_scale=0.1)

            poisson_noise_map = ccd.generate_poisson_noise(image, exposure_time, seed=2)

            simulated_poisson_image = image + poisson_noise_map

            assert simulated_poisson_image.shape == (2, 2)

            # Use known noise_map_1d map for given seed.
            assert (poisson_noise_map == np.array([[571, 0],
                                                   [0, -441]])).all()

            assert (simulated_poisson_image == np.array([[10000000.0 + 571, 0.],
                                                 [0., 10000000.0 - 441]])).all()

            assert (simulated_poisson_image - poisson_noise_map == image).all()

        def test__two_images_same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(self):
            image_0 = np.array([[10., 0.],
                              [0., 10.]])

            exposure_time_0 = scaled_array.ScaledSquarePixelArray(array=np.ones((2, 2)), pixel_scale=0.1)

            image_1 = np.array([[5., 0.],
                              [0., 5.]])

            exposure_time_1 = scaled_array.ScaledSquarePixelArray(array=2.0 * np.ones((2, 2)), pixel_scale=0.1)

            simulated_poisson_image_0 = image_0 + ccd.generate_poisson_noise(image_0, exposure_time_0, seed=1)
            simulated_poisson_image_1 = image_1 + ccd.generate_poisson_noise(image_1, exposure_time_1, seed=1)

            assert (simulated_poisson_image_0 / 2.0 == simulated_poisson_image_1).all()

        def test__same_as_above_but_range_of_image_values_and_exposure_times(self):
            image_0 = np.array([[10., 20.],
                              [30., 40.]])

            exposure_time_0 = scaled_array.ScaledSquarePixelArray(array=np.array([[2., 2.],
                                                                         [3., 4.]]), pixel_scale=0.1)

            image_1 = np.array([[20., 20.],
                              [45., 20.]])

            exposure_time_1 = scaled_array.ScaledSquarePixelArray(array=np.array([[1., 2.],
                                                                         [2., 8.]]), pixel_scale=0.1)

            simulated_poisson_image_0 = image_0 + ccd.generate_poisson_noise(image_0, exposure_time_0, seed=1)
            simulated_poisson_image_1 = image_1 + ccd.generate_poisson_noise(image_1, exposure_time_1, seed=1)

            assert (simulated_poisson_image_0[0, 0] == simulated_poisson_image_1[0, 0] / 2.0).all()
            assert simulated_poisson_image_0[0, 1] == simulated_poisson_image_1[0, 1]
            assert (simulated_poisson_image_0[1, 0] * 1.5 == pytest.approx(simulated_poisson_image_1[1, 0], 1e-2)).all()
            assert (simulated_poisson_image_0[1, 1] / 2.0 == simulated_poisson_image_1[1, 1]).all()

    class TestEstimateNoiseFromImage:

        def test__image_and_exposure_time_all_1s__no_background__noise_is_all_1s(self):
            # CCD (eps) = 1.0
            # Background (eps) = 0.0
            # Exposure times = 1.0 s
            # CCD (counts) = 1.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(1.0 + 0.0**2) = 1.0
            # Noise (eps) = 1.0 / 1.0

            array = np.ones((3, 3))
            exposure_time = np.ones((3, 3))
            background_noise = np.zeros((3, 3))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert (ccd_data.estimated_noise_map == np.ones((3, 3))).all()

        def test__image_all_4s__exposure_time_all_1s__no_background__noise_is_all_2s(self):
            # CCD (eps) = 4.0
            # Background (eps) = 0.0
            # Exposure times = 1.0 s
            # CCD (counts) = 4.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
            # Noise (eps) = 2.0 / 1.0

            array = 4.0 * np.ones((4, 2))

            exposure_time = np.ones((4, 2))
            background_noise = np.zeros((4, 2))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert (ccd_data.estimated_noise_map == 2.0 * np.ones((4, 2))).all()

        def test__image_all_1s__exposure_time_all_4s__no_background__noise_is_all_2_divided_4_so_halves(self):
            # CCD (eps) = 1.0
            # Background (eps) = 0.0
            # Exposure times = 4.0 s
            # CCD (counts) = 4.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
            # Noise (eps) = 2.0 / 4.0 = 0.5

            array = np.ones((1, 5))

            exposure_time = 4.0 * np.ones((1, 5))

            background_noise = np.zeros((1, 5))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert (ccd_data.estimated_noise_map == 0.5 * np.ones((1, 5))).all()

        def test__image_and_exposure_times_range_of_values__no_background__noises_estimates_correct(self):
            array = np.array([[5.0, 3.0],
                              [10.0, 20.0]])

            exposure_time = scaled_array.ScaledSquarePixelArray(np.array([[1.0, 2.0],
                                                                 [3.0, 4.0]]), pixel_scale=1.0)

            background_noise = np.zeros((2, 2))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert (ccd_data.estimated_noise_map == np.array([[np.sqrt(5.0), np.sqrt(6.0) / 2.0],
                                                              [np.sqrt(30.0) / 3.0, np.sqrt(80.0) / 4.0]])).all()

        def test__image_and_exposure_times_all_1s__background_is_float_sqrt_3__noise_is_all_2s(self):
            # CCD (eps) = 1.0
            # Background (eps) = sqrt(3.0)
            # Exposure times = 1.0 s
            # CCD (counts) = 1.0
            # Background (counts) = sqrt(3.0)

            # Noise (counts) = sqrt(1.0 + sqrt(3.0)**2) = sqrt(1.0 + 3.0) = 2.0
            # Noise (eps) = 2.0 / 1.0 = 2.0

            array = np.ones((3, 3))

            exposure_time = np.ones((3, 3))

            background_noise = 3.0 ** 0.5 * np.ones((3, 3))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert ccd_data.estimated_noise_map == pytest.approx(2.0 * np.ones((3, 3)), 1e-2)

        def test__image_and_exposure_times_all_1s__background_is_float_5__noise_all_correct(self):
            # CCD (eps) = 1.0
            # Background (eps) = 5.0
            # Exposure times = 1.0 s
            # CCD (counts) = 1.0
            # Background (counts) = 5.0

            # Noise (counts) = sqrt(1.0 + 5**2)
            # Noise (eps) = sqrt(1.0 + 5**2) / 1.0

            array = np.ones((2, 3))

            exposure_time = np.ones((2, 3))

            background_noise = 5 * np.ones((2, 3))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert ccd_data.estimated_noise_map == pytest.approx(
                np.array([[np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
                          [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)]]), 1e-2)

        def test__image_all_1s__exposure_times_all_2s__background_is_float_5__noise_all_correct(self):
            # CCD (eps) = 1.0
            # Background (eps) = 5.0
            # Exposure times = 2.0 s
            # CCD (counts) = 2.0
            # Background (counts) = 10.0

            # Noise (counts) = sqrt(2.0 + 10**2) = sqrt(2.0 + 100.0)
            # Noise (eps) = sqrt(2.0 + 100.0) / 2.0

            array = np.ones((2, 3))

            exposure_time = 2.0 * np.ones((2, 3))
            background_noise = 5.0 * np.ones((2, 3))

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert ccd_data.estimated_noise_map == pytest.approx(
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

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert ccd_data.estimated_noise_map == pytest.approx(np.array([[np.sqrt(1.0 + 144.0), np.sqrt(2.0 + 144.0)],
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

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert ccd_data.estimated_noise_map == \
                   pytest.approx(np.array([[np.sqrt(5.0 + 81.0), np.sqrt(6.0 + 18.0 ** 2.0) / 2.0],
                                           [np.sqrt(30.0 + 27.0 ** 2.0) / 3.0,
                                            np.sqrt(80.0 + 36.0 ** 2.0) / 4.0]]), 1e-2)

        def test__image_and_exposure_times_and_background_are_all_ranges_of_values__noise_estimates_correct(self):
            # Use same pattern as above, noting that we are now also using a variable background signal_to_noise_ratio map.

            array = np.array([[5.0, 3.0],
                              [10.0, 20.0]])

            exposure_time = np.array([[1.0, 2.0],
                                      [3.0, 4.0]])

            background_noise = np.array([[5.0, 6.0],
                                         [7.0, 8.0]])

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0),
                                   exposure_time_map=exposure_time, background_noise_map=background_noise)

            assert ccd_data.estimated_noise_map == pytest.approx(
                np.array([[np.sqrt(5.0 + 5.0 ** 2.0), np.sqrt(6.0 + 12.0 ** 2.0) / 2.0],
                          [np.sqrt(30.0 + 21.0 ** 2.0) / 3.0,
                           np.sqrt(80.0 + 32.0 ** 2.0) / 4.0]]),
                1e-2)

    class TestEstimateDataGrid(object):

        def test__via_edges__input_all_ones__sky_bg_level_1(self):
            
            ccd_data = ccd.CCDData(image=np.ones((3, 3)), noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            
            sky_noise = ccd_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):
            image_array = np.array([[1, 1, 1],
                                    [1, 100, 1],
                                    [1, 1, 1]])

            ccd_data = ccd.CCDData(image=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            sky_noise = ccd_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):
            image_array = np.array([[1, 1, 1],
                                    [1, 100, 1],
                                    [1, 100, 1],
                                    [1, 1, 1]])

            ccd_data = ccd.CCDData(image=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            sky_noise = ccd_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image_array = np.array([[1, 1, 1, 1],
                                    [1, 100, 100, 1],
                                    [1, 100, 100, 1],
                                    [1, 1, 1, 1]])

            ccd_data = ccd.CCDData(image=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            sky_noise = ccd_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):
            image_array = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 100, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])

            ccd_data = ccd.CCDData(image=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            sky_noise = ccd_data.background_noise_from_edges(no_edges=2)

            assert sky_noise == 0.0

        def test__via_edges__6x5_image_two_edges__values(self):
            image_array = np.array([[0, 1, 2, 3, 4],
                                    [5, 6, 7, 8, 9],
                                    [10, 11, 100, 12, 13],
                                    [14, 15, 100, 16, 17],
                                    [18, 19, 20, 21, 22],
                                    [23, 24, 25, 26, 27]])

            ccd_data = ccd.CCDData(image=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            sky_noise = ccd_data.background_noise_from_edges(no_edges=2)

            assert sky_noise == np.std(np.arange(28))

        def test__via_edges__7x7_image_three_edges__values(self):
            image_array = np.array([[0, 1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [21, 22, 23, 100, 24, 25, 26],
                                    [27, 28, 29, 30, 31, 32, 33],
                                    [34, 35, 36, 37, 38, 39, 40],
                                    [41, 42, 43, 44, 45, 46, 47]])

            ccd_data = ccd.CCDData(image=image_array, noise_map=np.ones((3, 3)), psf=np.ones((3, 3)),
                                   pixel_scale=0.1)
            sky_noise = ccd_data.background_noise_from_edges(no_edges=3)

            assert sky_noise == np.std(np.arange(48))

    class TestSignalToNoise:

        def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):

            array = np.array([[1.0, 2.0],
                              [3.0, 4.0]])

            noise = np.array([[10.0, 10.0],
                              [30.0, 4.0]])

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((2, 2)), pixel_scale=1.0), noise_map=noise)

            assert (ccd_data.signal_to_noise_map == np.array([[0.1, 0.2],
                                                         [0.1, 1.0]])).all()
            assert ccd_data.signal_to_noise_max == 1.0


        def test__same_as_above__but_image_has_negative_values__replaced_with_zeros(self):

            array = np.array([[-1.0, 2.0],
                              [3.0, -4.0]])

            noise = np.array([[10.0, 10.0],
                              [30.0, 4.0]])

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((2, 2)), pixel_scale=1.0), noise_map=noise)

            assert (ccd_data.signal_to_noise_map == np.array([[0.0, 0.2],
                                                              [0.1, 0.0]])).all()
            assert ccd_data.signal_to_noise_max == 0.2

    class TestAbsoluteSignalToNoise:

        def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(self):

            array = np.array([[-1.0, 2.0],
                              [3.0, -4.0]])

            noise = np.array([[10.0, 10.0],
                              [30.0, 4.0]])

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((2, 2)), pixel_scale=1.0), noise_map=noise)

            assert (ccd_data.absolute_signal_to_noise_map == np.array([[0.1, 0.2],
                                                         [0.1, 1.0]])).all()
            assert ccd_data.absolute_signal_to_noise_max == 1.0

    class TestPotentialChiSquaredMap:

        def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(self):

            array = np.array([[-1.0, 2.0],
                              [3.0, -4.0]])

            noise = np.array([[10.0, 10.0],
                              [30.0, 4.0]])

            ccd_data = ccd.CCDData(image=array, pixel_scale=1.0,
                                   psf=ccd.PSF(array=np.ones((2, 2)), pixel_scale=1.0), noise_map=noise)

            assert (ccd_data.potential_chi_squared_map == np.array([[0.1**2.0, 0.2**2.0],
                                                                    [0.1**2.0, 1.0**2.0]])).all()
            assert ccd_data.potential_chi_squared_max == 1.0

    class TestNewCCDModifiedImage:

        def test__ccd_data_returns_with_modified_image(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((4, 4)), pixel_scale=1.0)
            image_array[2, 2] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((4, 4)), pixel_scale=1.0)
            noise_map_array[2,2] = 3.0

            background_noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((4, 4)), pixel_scale=1.0)
            background_noise_map_array[2,2] = 4.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(np.ones((4, 4)), pixel_scale=1.0)
            exposure_time_map_array[2,2] = 5.0

            background_sky_map_array = scaled_array.ScaledSquarePixelArray(np.ones((4, 4)), pixel_scale=1.0)
            background_sky_map_array[2,2] = 6.0

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            modified_image = scaled_array.ScaledSquarePixelArray(np.ones((4, 4)), pixel_scale=1.0)
            modified_image[2, 2] = 10.0

            ccd_data = ccd_data.new_ccd_data_with_modified_image(modified_image=modified_image)

            assert (ccd_data.image == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 10.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 4.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.exposure_time_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 5.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_sky_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 6.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()

            assert (ccd_data.poisson_noise_map == None)

            assert ccd_data.pixel_scale == 1.0
            assert (ccd_data.psf == np.zeros((3,3))).all()
            assert ccd_data.origin == (0.0, 0.0)

    class TestNewCCDBinnedUp:

        def test__all_components_binned_up_correct(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3:5, 3] = 2.0
            binned_image_util = array_util.bin_up_array_2d_using_mean(array_2d=image_array, bin_up_factor=2)

            noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            noise_map_array[3,3:5] = 3.0
            binned_noise_map_util = array_util.bin_up_array_2d_using_quadrature(array_2d=noise_map_array,
                                                                                      bin_up_factor=2)

            background_noise_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_noise_map_array[3:5,3] = 4.0
            binned_background_noise_map_util = array_util.bin_up_array_2d_using_quadrature(
                array_2d=background_noise_map_array, bin_up_factor=2)

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            exposure_time_map_array[3,3:5] = 5.0
            binned_exposure_time_map_util = array_util.bin_up_array_2d_using_sum(array_2d=exposure_time_map_array,
                                                                                       bin_up_factor=2)

            background_sky_map_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            background_sky_map_array[3,3:5] = 6.0
            binned_background_sky_map_util = array_util.bin_up_array_2d_using_mean(
                array_2d=background_sky_map_array, bin_up_factor=2)

            psf = ccd.PSF(array=np.ones((3,5)), pixel_scale=1.0)
            psf_util = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5)

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=psf,
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            ccd_data = ccd_data.new_ccd_data_with_binned_up_arrays(bin_up_factor=2)

            assert (ccd_data.image == binned_image_util).all()
            assert (ccd_data.psf == psf_util).all()
            assert (ccd_data.noise_map == binned_noise_map_util).all()
            assert (ccd_data.background_noise_map == binned_background_noise_map_util).all()
            assert (ccd_data.exposure_time_map == binned_exposure_time_map_util).all()
            assert (ccd_data.background_sky_map == binned_background_sky_map_util).all()
            assert (ccd_data.poisson_noise_map == None)

            assert ccd_data.pixel_scale == 2.0
            assert ccd_data.image.pixel_scale == 2.0
            assert ccd_data.psf.pixel_scale == pytest.approx(1.66666666666, 1.0e-4)
            assert ccd_data.noise_map.pixel_scale == 2.0
            assert ccd_data.background_noise_map.pixel_scale == 2.0
            assert ccd_data.exposure_time_map.pixel_scale == 2.0
            assert ccd_data.background_sky_map.pixel_scale == 2.0

            assert ccd_data.origin == (0.0, 0.0)

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

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            ccd_data = ccd_data.new_ccd_data_with_resized_arrays(new_shape=(4, 4))

            assert (ccd_data.image == np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_noise_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 4.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.exposure_time_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 5.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_sky_map == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 6.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()

            assert (ccd_data.poisson_noise_map == None)

            assert ccd_data.pixel_scale == 1.0
            assert (ccd_data.psf == np.zeros((3,3))).all()
            assert ccd_data.origin == (0.0, 0.0)

        def test__resize_psf(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0))

            ccd_data = ccd_data.new_ccd_data_with_resized_psf(new_shape=(1, 1))

            assert (ccd_data.image == np.ones((6,6))).all()
            assert ccd_data.pixel_scale == 1.0
            assert (ccd_data.psf == np.zeros((1,1))).all()
            assert ccd_data.origin == (0.0, 0.0)

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

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            ccd_data = ccd_data.new_ccd_data_with_resized_arrays(new_shape=(3, 3), new_centre_pixels=(3, 3))

            assert (ccd_data.image == np.array([[1.0, 1.0, 1.0],
                                    [1.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.noise_map ==  np.array([[1.0, 1.0, 1.0],
                                               [1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_noise_map ==  np.array([[1.0, 1.0, 1.0],
                                                          [1.0, 4.0, 1.0],
                                                          [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.exposure_time_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 5.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_sky_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 6.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()

            assert (ccd_data.poisson_noise_map == None)

            assert ccd_data.pixel_scale == 1.0
            assert (ccd_data.psf == np.zeros((3,3))).all()
            assert ccd_data.origin == (0.0, 0.0)

        def test__input_new_centre_arcsec__arrays_use_new_centre__psf_does_not(self):

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

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   exposure_time_map=exposure_time_map_array, background_sky_map=background_sky_map_array)

            ccd_data = ccd_data.new_ccd_data_with_resized_arrays(new_shape=(3, 3), new_centre_arcsec=(-0.5, 0.5))

            assert (ccd_data.image == np.array([[1.0, 1.0, 1.0],
                                    [1.0, 2.0, 1.0],
                                    [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.noise_map ==  np.array([[1.0, 1.0, 1.0],
                                               [1.0, 3.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_noise_map ==  np.array([[1.0, 1.0, 1.0],
                                                          [1.0, 4.0, 1.0],
                                                          [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.exposure_time_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 5.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()
            assert (ccd_data.background_sky_map ==  np.array([[1.0, 1.0, 1.0],
                                                        [1.0, 6.0, 1.0],
                                                        [1.0, 1.0, 1.0]])).all()

            assert (ccd_data.poisson_noise_map == None)

            assert ccd_data.pixel_scale == 1.0
            assert (ccd_data.psf == np.zeros((3,3))).all()
            assert ccd_data.origin == (0.0, 0.0)

        def test__input_both_centres__raises_error(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0))

            with pytest.raises(exc.ImagingException):
                ccd_data.new_ccd_data_with_resized_arrays(new_shape=(3, 3), new_centre_pixels=(3, 3),
                                                          new_centre_arcsec=(-0.5, 0.5))

    class TestNewImageConvertedFrom:

        def test__counts__all_arrays_in_units_of_flux_are_converted(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((3, 3)), pixel_scale=1.0)
            noise_map_array = scaled_array.ScaledSquarePixelArray(2.0 * np.ones((3, 3)), pixel_scale=1.0)
            background_noise_map_array = scaled_array.ScaledSquarePixelArray(3.0 * np.ones((3, 3)), pixel_scale=1.0)
            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(0.5 * np.ones((3, 3)), pixel_scale=1.0)
            background_sky_map_array = scaled_array.ScaledSquarePixelArray(6.0 * np.ones((3, 3)), pixel_scale=1.0)

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   poisson_noise_map=None, exposure_time_map=exposure_time_map_array,
                                   background_sky_map=background_sky_map_array)

            ccd_data = ccd_data.new_ccd_data_converted_from_electrons()

            assert (ccd_data.image == 2.0*np.ones((3,3))).all()
            assert (ccd_data.noise_map == 4.0*np.ones((3,3))).all()
            assert (ccd_data.background_noise_map == 6.0*np.ones((3,3))).all()
            assert ccd_data.poisson_noise_map == None
            assert (ccd_data.background_sky_map == 12.0*np.ones((3,3))).all()
            assert ccd_data.origin == (0.0, 0.0)

        def test__adus__all_arrays_in_units_of_flux_are_converted(self):

            image_array = scaled_array.ScaledSquarePixelArray(np.ones((3, 3)), pixel_scale=1.0)
            noise_map_array = scaled_array.ScaledSquarePixelArray(2.0 * np.ones((3, 3)), pixel_scale=1.0)
            background_noise_map_array = scaled_array.ScaledSquarePixelArray(3.0 * np.ones((3, 3)), pixel_scale=1.0)
            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(0.5 * np.ones((3, 3)), pixel_scale=1.0)
            background_sky_map_array = scaled_array.ScaledSquarePixelArray(6.0 * np.ones((3, 3)), pixel_scale=1.0)

            ccd_data = ccd.CCDData(image=image_array, pixel_scale=1.0, psf=ccd.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                                   noise_map=noise_map_array, background_noise_map=background_noise_map_array,
                                   poisson_noise_map=None, exposure_time_map=exposure_time_map_array,
                                   background_sky_map=background_sky_map_array)

            ccd_data = ccd_data.new_ccd_data_converted_from_adus(gain=2.0)

            assert (ccd_data.image == 2.0*2.0*np.ones((3,3))).all()
            assert (ccd_data.noise_map == 2.0*4.0*np.ones((3,3))).all()
            assert (ccd_data.background_noise_map == 2.0*6.0*np.ones((3,3))).all()
            assert ccd_data.poisson_noise_map == None
            assert (ccd_data.background_sky_map == 2.0*12.0*np.ones((3,3))).all()
            assert ccd_data.origin == (0.0, 0.0)

    class TestNewImageWithPoissonNoiseAdded:

        def test__mock_image_all_1s__poisson_noise_is_added_correct(self):

            psf = ccd.PSF(array=np.ones((3, 3)), pixel_scale=3.0, renormalize=False)
            ccd_data = ccd.CCDData(image=np.ones((4, 4)), pixel_scale=3., psf=psf, noise_map=np.ones((4, 4)),
                                   exposure_time_map=3.0 * np.ones((4, 4)), background_sky_map=4.0 * np.ones((4, 4)))

            mock_image = np.ones((4, 4))
            mock_image_with_sky = mock_image + 4.0 * np.ones((4, 4))
            mock_image_with_sky_and_noise = mock_image_with_sky + ccd.generate_poisson_noise(image=mock_image_with_sky,
                                                                                             exposure_time_map=3.0 * np.ones(
                                                                                                (4, 4)), seed=1)

            mock_image_with_noise = mock_image_with_sky_and_noise - 4.0 * np.ones((4, 4))

            ccd_with_noise = ccd_data.new_ccd_data_with_poisson_noise_added(seed=1)

            assert (ccd_with_noise.image == mock_image_with_noise).all()


class TestNoiseMap(object):

    class TestFromWeightMap:

        def test__weight_map_no_zeros__uses_1_over_sqrt_value(self):

            weight_map = np.array([[1.0, 4.0, 16.0],
                                   [1.0, 4.0, 16.0]])

            noise_map = ccd.NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.5, 0.25],
                                           [1.0, 0.5, 0.25]])).all()
            assert noise_map.origin == (0.0, 0.0)

        def test__weight_map_no_zeros__zeros_set_to_10000000(self):

            weight_map = np.array([[1.0, 4.0, 0.0],
                                   [1.0, 4.0, 16.0]])

            noise_map = ccd.NoiseMap.from_weight_map(weight_map=weight_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.5, 1.0e8],
                                           [1.0, 0.5, 0.25]])).all()
            assert noise_map.origin == (0.0, 0.0)

    class TestFromInverseNoiseMap:

        def test__inverse_noise_map_no_zeros__uses_1_over_value(self):

            inverse_noise_map = np.array([[1.0, 4.0, 16.0],
                                          [1.0, 4.0, 16.0]])

            noise_map = ccd.NoiseMap.from_inverse_noise_map(inverse_noise_map=inverse_noise_map, pixel_scale=1.0)

            assert (noise_map == np.array([[1.0, 0.25, 0.0625],
                                           [1.0, 0.25, 0.0625]])).all()
            assert noise_map.origin == (0.0, 0.0)

    class TestFromImageAndBackgroundNoiseMap:

        def test__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map,
                                                                         gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(2.)], [np.sqrt(2.), np.sqrt(2.)]])).all()

        def test__image_all_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_3s(self):

            ccd_data = np.array([[2.0, 2.0], [2.0, 2.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map,
                                                                         gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(3.), np.sqrt(3.)], [np.sqrt(3.), np.sqrt(3.)]])).all()

        def test__image_all_1s__bg_noise_all_2s__exposure_time_all_1s__noise_map_all_sqrt_5s(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[2.0, 2.0], [2.0, 2.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map,
                                                                         gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(5.), np.sqrt(5.)], [np.sqrt(5.), np.sqrt(5.)]])).all()

        def test__image_all_1s__bg_noise_all_1s__exposure_time_all_2s__noise_map_all_sqrt_6s_over_2(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[2.0, 2.0], [2.0, 2.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map,
                                                                         gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(6.) / 2.0, np.sqrt(6.) / 2.0],
                                           [np.sqrt(6.) / 2.0, np.sqrt(6.) / 2.0]])).all()

        def test__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_1s(self):

            ccd_data = np.array([[-2.0, -2.0], [-2.0, -2.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map,
                                                                         gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__same_as_above__use_different_values_in_different_array_elemets(self):

            ccd_data = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[4.0, 3.0], [2.0, 1.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map,
                                                                         gain=1.0, exposure_time_map=exposure_time_map)

            assert (noise_map == np.array([[np.sqrt(20.) / 4.0, np.sqrt(15.) / 3.0],
                                           [np.sqrt(20.) / 2.0, np.sqrt(12.)]])).all()

        def test__convert_from_electrons__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                         gain=2.0, convert_from_electrons=True)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(2.)], [np.sqrt(2.), np.sqrt(2.)]])).all()

        def test__convert_from_electrons__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_10s__noise_map_all_1s(self):

            ccd_data = np.array([[-2.0, -2.0], [-2.0, -2.0]])
            background_noise_map = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[10.0, 10.0], [10.0, 10.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                         gain=1.0, convert_from_electrons=True)

            assert (noise_map == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__convert_from_electrons__same_as_above__use_different_values_in_different_array_elemets(self):

            ccd_data = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[10.0, 11.0], [12.0, 13.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                         gain=4.0, convert_from_electrons=True)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(3.)], [np.sqrt(6.), np.sqrt(12.)]])).all()

        def test__convert_from_adus__same_as_above__gain_is_1__same_values(self):

            ccd_data = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[10.0, 11.0], [12.0, 13.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                         gain=1.0, convert_from_adus=True)

            assert (noise_map == np.array([[np.sqrt(2.), np.sqrt(3.)], [np.sqrt(6.), np.sqrt(12.)]])).all()

        def test__convert_from_adus__same_as_above__gain_is_2__values_change(self):

            ccd_data = np.array([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = np.array([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = np.array([[10.0, 11.0], [12.0, 13.0]])

            noise_map = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data,
                                                                         background_noise_map=background_noise_map, exposure_time_map=exposure_time_map,
                                                                         gain=2.0, convert_from_adus=True)

            assert (noise_map == np.array([[np.sqrt(6.) / 2.0, np.sqrt(8.) / 2.0],
                                           [np.sqrt(20.) / 2.0, np.sqrt(42.) / 2.0]])).all()


class TestPoissonNoiseMap(object):

    class TestFromImageAndExposureTimeMap:

        def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=ccd_data,
                                                                                     exposure_time_map=exposure_time_map, gain=1.0)

            assert (poisson_noise_map == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__image_all_2s_and_3s__exposure_time_all_1s__noise_map_all_sqrt_2s_and_3s(self):

            ccd_data = np.array([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=ccd_data,
                                                                                     exposure_time_map=exposure_time_map, gain=1.0)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]])).all()

        def test__image_all_1s__exposure_time_all__2s_and_3s__noise_map_all_sqrt_2s_and_3s(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[2.0, 2.0], [3.0, 3.0]])

            poisson_noise_map = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=ccd_data,
                                                                                     exposure_time_map=exposure_time_map, gain=1.0)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
                                                   [np.sqrt(3.0) / 3.0, np.sqrt(3.0) / 3.0]])).all()

        def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s__gain_is_2__ignores_gain(self):

            ccd_data = np.array([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = np.array([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=ccd_data,
                                                                                     exposure_time_map=exposure_time_map, gain=2.0)

            assert (poisson_noise_map == np.array([[np.sqrt(1.0), np.sqrt(1.0)],
                                                   [np.sqrt(1.0), np.sqrt(1.0)]])).all()

        def test__convert_from_electrons_is_true__image_already_in_counts_so_exposure_time_ignored(self):

            ccd_data = np.array([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = np.array([[10.0, 10.0], [10.0, 10.0]])

            poisson_noise_map = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=ccd_data,
                                                                                     exposure_time_map=exposure_time_map, gain=4.0,
                                                                                     convert_from_electrons=True)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]])).all()

        def test__same_as_above__convert_from_adus__includes_gain_multiplication(self):

            ccd_data = np.array([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = np.array([[10.0, 10.0], [10.0, 10.0]])

            poisson_noise_map = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1, image=ccd_data,
                                                                                     exposure_time_map=exposure_time_map, gain=2.0,
                                                                                     convert_from_adus=True)

            assert (poisson_noise_map == np.array([[np.sqrt(2.0*2.0) / 2.0, np.sqrt(2.0*2.0) / 2.0],
                                                   [np.sqrt(2.0*3.0) / 2.0, np.sqrt(2.0*3.0) / 2.0]])).all()


class TestPSF(object):

    class TestConstructors(object):

        def test__init__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = ccd.PSF(array=np.ones((3, 3)), pixel_scale=1.0, renormalize=False)

            assert psf.shape == (3, 3)
            assert psf.pixel_scale == 1.0
            assert (psf == np.ones((3, 3))).all()
            assert psf.origin == (0.0, 0.0)

        def test__init__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = ccd.PSF(array=np.ones((4, 3)), pixel_scale=1.0, renormalize=False)

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.shape == (4, 3)
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = ccd.PSF.from_fits_with_scale(file_path=test_data_dir + '3x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (psf == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = ccd.PSF.from_fits_with_scale(file_path=test_data_dir + '4x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

    class TestRenormalize(object):

        def test__input_is_already_normalized__no_change(self):
            psf_data = np.ones((3, 3)) / 9.0

            psf = ccd.PSF(array=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            psf_data = np.ones((3, 3))

            psf = ccd.PSF(array=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__same_as_above__renomalized_false_does_not_renormalize(self):
            psf_data = np.ones((3, 3))

            psf = ccd.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            assert psf == pytest.approx(np.ones((3, 3)), 1e-3)

    class TestBinnedUp(object):

        def test__psf_is_even_x_even__rescaled_to_odd_x_odd__no_use_of_dimension_trimming(self):

            array_2d = np.ones((6, 6))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5, renormalize=True)
            assert psf.pixel_scale == 2.0
            assert psf == (1.0/9.0)*np.ones((3,3))

            array_2d = np.ones((9, 9))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.333333333333333, renormalize=True)
            assert psf.pixel_scale == 3.0
            assert psf == (1.0/9.0)*np.ones((3,3))

            array_2d = np.ones((18, 6))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5, renormalize=True)
            assert psf.pixel_scale == 2.0
            assert psf == (1.0/27.0)*np.ones((9,3))

            array_2d = np.ones((6, 18))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5, renormalize=True)
            assert psf.pixel_scale == 2.0
            assert psf == (1.0/27.0)*np.ones((3,9))

        def test__psf_is_even_x_even_after_binning_up__resized_to_odd_x_odd_with_shape_plus_one(self):

            array_2d = np.ones((2,2))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=2.0, renormalize=True)
            assert psf.pixel_scale == 0.4
            assert psf == (1.0/25.0)*np.ones((5,5))

            array_2d = np.ones((40, 40))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.1, renormalize=True)
            assert psf.pixel_scale == 8.0
            assert psf == (1.0/25.0)*np.ones((5,5))

            array_2d = np.ones((2,4))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=2.0, renormalize=True)
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0/45.0)*np.ones((5,9))

            array_2d = np.ones((4,2))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=2.0, renormalize=True)
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0/45.0)*np.ones((9,5))

        def test__psf_is_odd_and_even_after_binning_up__resized_to_odd_and_odd_with_shape_plus_one(self):

            array_2d = np.ones((6,4))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5, renormalize=True)
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0/9.0)*np.ones((3,3))

            array_2d = np.ones((9, 12))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.33333333333, renormalize=True)
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((3, 5))

            array_2d = np.ones((4,6))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.5, renormalize=True)
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0/9.0)*np.ones((3,3))

            array_2d = np.ones((12, 9))
            psf = ccd.PSF(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_psf_with_rescaled_odd_dimensioned_array(rescale_factor=0.33333333333, renormalize=True)
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((5, 3))


    class TestNewRenormalizedPsf(object):

        def test__input_is_already_normalized__no_change(self):

            psf_data = np.ones((3, 3)) / 9.0

            psf = ccd.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            psf_new = psf.new_psf_with_renormalized_array()

            assert psf_new == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            psf_data = np.ones((3, 3))

            psf = ccd.PSF(array=psf_data, pixel_scale=1.0, renormalize=False)

            psf_new = psf.new_psf_with_renormalized_array()

            assert psf_new == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    class TestConvolve(object):

        def test__kernel_is_not_odd_x_odd__raises_error(self):
            kernel = np.array([[0.0, 1.0],
                               [1.0, 2.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            with pytest.raises(exc.KernelException):
                psf.convolve(np.ones((5, 5)))

        def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            image = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == kernel).all()

        def test__image_is_4x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == np.array([[0.0, 1.0, 0.0, 0.0],
                                             [1.0, 2.0, 1.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0]])).all()

        def test__image_is_4x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            image = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == np.array([[0.0, 1.0, 0.0],
                                             [1.0, 2.0, 1.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0]])).all()

        def test__image_is_3x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == np.array([[0.0, 1.0, 0.0, 0.0],
                                             [1.0, 2.0, 1.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0]])).all()

        def test__image_is_4x4_has_two_central_values__kernel_is_asymmetric__blurred_image_follows_convolution(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == np.array([[1.0, 1.0, 1.0, 0.0],
                                             [2.0, 3.0, 2.0, 1.0],
                                             [1.0, 5.0, 5.0, 1.0],
                                             [0.0, 1.0, 3.0, 3.0]])).all()

        def test__image_is_4x4_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
                self):
            image = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]])

            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == np.array([[1.0, 1.0, 0.0, 0.0],
                                             [2.0, 1.0, 1.0, 1.0],
                                             [3.0, 3.0, 2.0, 2.0],
                                             [0.0, 0.0, 1.0, 3.0]])).all()

        def test__image_is_4x4_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
                self):
            image = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == np.array([[2.0, 1.0, 0.0, 0.0],
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
            profile_psf = ccd.PSF(array=profile_gaussian_2d, pixel_scale=1.0, renormalize=True)

            imaging_psf = ccd.PSF.simulate_as_gaussian(shape=(3, 3), pixel_scale=1.0, centre=(0.1, 0.1),
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

            psf = ccd.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
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

            psf = ccd.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
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

            psf = ccd.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
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

            psf = ccd.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
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

            psf = ccd.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
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

            psf = ccd.PSF.simulate_as_gaussian_via_alma_fits_header_parameters(shape=shape, pixel_scale=pixel_scale,
                                                                               y_stddev=2.0e-5, x_stddev=1.0e-5, theta=theta_deg)

            assert psf_astropy == pytest.approx(psf, 1e-4)


class TestExposureTimeMap(object):

    class TestFromExposureTimeAndBackgroundNoiseMap:

        def test__from_background_noise_map__covnerts_to_exposure_times(self):

            background_noise_map = np.array([[1.0, 4.0, 8.0],
                                             [1.0, 4.0, 8.0]])

            exposure_time_map = ccd.ExposureTimeMap.from_exposure_time_and_inverse_noise_map(pixel_scale=0.1,
                                                                                             exposure_time=1.0, inverse_noise_map=background_noise_map)

            assert (exposure_time_map == np.array([[0.125, 0.5, 1.0],
                                                   [0.125, 0.5, 1.0]])).all()
            assert exposure_time_map.origin == (0.0, 0.0)

            exposure_time_map = ccd.ExposureTimeMap.from_exposure_time_and_inverse_noise_map(pixel_scale=0.1,
                                                                                             exposure_time=3.0, inverse_noise_map=background_noise_map)

            assert (exposure_time_map == np.array([[0.375, 1.5, 3.0],
                                                   [0.375, 1.5, 3.0]])).all()
            assert exposure_time_map.origin == (0.0, 0.0)


class TestCCDFromFits(object):

    def test__no_settings_just_pass_fits(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               renormalize_psf=False)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert ccd_data.background_noise_map == None
        assert ccd_data.poisson_noise_map == None
        assert ccd_data.exposure_time_map == None
        assert ccd_data.background_sky_map == None

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1

    def test__optional_array_paths_included__loads_optional_array(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__all_files_in_one_fits__load_using_different_hdus(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_multiple_hdu.fits', image_hdu=0,
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


        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                               noise_map_path=test_data_dir + '3x3_ones.fits',
                                               psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               exposure_time_map_from_single_value=3.0,
                                               renormalize_psf=False)

        assert (ccd_data.exposure_time_map == 3.0*np.ones((3,3))).all()

    def test__exposure_time_map_from_inverse_noise_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               exposure_time_map_from_single_value=3.0,
                                               exposure_time_map_from_inverse_noise_map=True,
                                               renormalize_psf=False)

        assert (ccd_data.exposure_time_map == 3.0*np.ones((3,3))).all()

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               exposure_time_map_from_single_value=6.0,
                                               exposure_time_map_from_inverse_noise_map=True,
                                               renormalize_psf=False)

        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()

    def test__exposure_time_map_from_inverse_noise_map__background_noise_is_converted_from_inverse_noise_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_ones_central_two.fits',
                                               convert_background_noise_map_from_inverse_noise_map=True,
                                               exposure_time_map_from_single_value=3.0,
                                               exposure_time_map_from_inverse_noise_map=True,
                                               renormalize_psf=False)

        inverse_noise_map = np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])

        background_noise_map_converted = ccd.NoiseMap.from_inverse_noise_map(inverse_noise_map=inverse_noise_map,
                                                                             pixel_scale=0.1)

        assert (ccd_data.background_noise_map == np.array([[1.0, 1.0, 1.0],
                                                          [1.0, 0.5, 1.0],
                                                          [1.0, 1.0, 1.0]])).all()
        assert (ccd_data.background_noise_map == background_noise_map_converted).all()

        assert (ccd_data.exposure_time_map == np.array([[1.5, 1.5, 1.5],
                                                        [1.5, 3.0, 1.5],
                                                        [1.5, 1.5, 1.5]])).all()

    def test__pad_shape_of_image_arrays_and_psf(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               resized_ccd_shape=(5, 5), resized_psf_shape=(7, 7),
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

        assert (ccd_data.image == padded_array).all()
        assert (ccd_data.psf == psf_padded_array).all()
        assert (ccd_data.noise_map == 3.0*padded_array).all()
        assert (ccd_data.background_noise_map == 4.0 * padded_array).all()
        assert (ccd_data.poisson_noise_map == 5.0 * padded_array).all()
        assert (ccd_data.exposure_time_map == 6.0 * padded_array).all()
        assert (ccd_data.background_sky_map == 7.0 * padded_array).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               exposure_time_map_from_single_value=3.0, pixel_scale=0.1,
                                               resized_ccd_shape=(5, 5), resized_psf_shape=(7, 7),
                                               renormalize_psf=False)

        exposure_padded_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 3.0, 3.0, 3.0, 0.0],
                                 [0.0, 3.0, 3.0, 3.0, 0.0],
                                 [0.0, 3.0, 3.0, 3.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0]])

        assert (ccd_data.image == padded_array).all()
        assert (ccd_data.exposure_time_map == exposure_padded_array).all()

    def test__trim_shape_of_image_arrays_and_psf(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               resized_ccd_shape=(1, 1), resized_psf_shape=(1, 1),
                                               renormalize_psf=False)

        trimmed_array = np.array([[1.0]])

        assert (ccd_data.image == trimmed_array).all()
        assert (ccd_data.psf == 2.0*trimmed_array).all()
        assert (ccd_data.noise_map == 3.0*trimmed_array).all()
        assert (ccd_data.background_noise_map == 4.0 * trimmed_array).all()
        assert (ccd_data.poisson_noise_map == 5.0 * trimmed_array).all()
        assert (ccd_data.exposure_time_map == 6.0 * trimmed_array).all()
        assert (ccd_data.background_sky_map == 7.0 * trimmed_array).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_noise_map_from_weight_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               convert_noise_map_from_weight_map=True,
                                               renormalize_psf=False)

        noise_map_converted = ccd.NoiseMap.from_weight_map(weight_map=3.0 * np.ones((3, 3)), pixel_scale=0.1)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == noise_map_converted).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_noise_map_from_inverse_noise_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               convert_noise_map_from_inverse_noise_map=True,
                                               renormalize_psf=False)

        noise_map_converted = ccd.NoiseMap.from_inverse_noise_map(inverse_noise_map=3.0 * np.ones((3, 3)),
                                                                  pixel_scale=0.1)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == noise_map_converted).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_from_image_and_background_noise_map=True,
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False)

        noise_map_converted = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1, image=ccd_data.image,
                                                                               background_noise_map=ccd_data.background_noise_map,
                                                                               gain=2.0, exposure_time_map=ccd_data.exposure_time_map)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == noise_map_converted).all()
        assert (ccd_data.noise_map == (np.sqrt((24.0)**2.0 + (6.0))/(6.0))*np.ones((3,3)))
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map__include_convert_from_electrons(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_from_image_and_background_noise_map=True,
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               convert_from_electrons=True,
                                               renormalize_psf=False)

        noise_map_converted = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1,
                                                                               image=1.0*np.ones((3,3)), background_noise_map=4.0*np.ones((3,3)),
                                                                               gain=None, exposure_time_map=ccd_data.exposure_time_map, convert_from_electrons=True)

        noise_map_converted = noise_map_converted / 6.0

        assert (ccd_data.image == np.ones((3,3)) / 6.0).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == noise_map_converted).all()
        assert (ccd_data.noise_map == np.sqrt(17.0)*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3)) / 6.0).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map__include_convert_from_adus(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_from_image_and_background_noise_map=True,
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               gain=2.0, convert_from_adus=True,
                                               renormalize_psf=False)

        noise_map_converted = ccd.NoiseMap.from_image_and_background_noise_map(pixel_scale=0.1,
                                                                               image=1.0*np.ones((3,3)), background_noise_map=4.0*np.ones((3,3)),
                                                                               gain=2.0, exposure_time_map=ccd_data.exposure_time_map, convert_from_adus=True)

        noise_map_converted = 2.0 * noise_map_converted / 6.0

        assert (ccd_data.image == 2.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == noise_map_converted).all()
        assert (ccd_data.noise_map == np.sqrt(66.0)*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.background_noise_map == 2.0*4.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.poisson_noise_map == 2.0*5.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 2.0*7.0*np.ones((3,3)) / 6.0).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_background_noise_map_from_weight_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               convert_background_noise_map_from_weight_map=True)

        background_noise_map_converted = ccd.NoiseMap.from_weight_map(weight_map=4.0 * np.ones((3, 3)), pixel_scale=0.1)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == background_noise_map_converted).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_background_noise_map_from_inverse_noise_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               convert_background_noise_map_from_inverse_noise_map=True)

        background_noise_map_converted = ccd.NoiseMap.from_inverse_noise_map(inverse_noise_map=4.0 * np.ones((3, 3)),
                                                                             pixel_scale=0.1)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == background_noise_map_converted).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__poisson_noise_map_from_image(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               poisson_noise_map_from_image=True)

        poisson_noise_map_converted = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1,
                                                                                           image=np.ones((3,3)), exposure_time_map=ccd_data.exposure_time_map, gain=None)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == (np.sqrt(6.0)/(6.0))*np.ones((3,3)))
        assert (ccd_data.poisson_noise_map == poisson_noise_map_converted).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__poisson_noise_map_from_image__include_convert_from_electrons(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               poisson_noise_map_from_image=True, convert_from_electrons=True)

        poisson_noise_map_counts = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1,
                                                                                        image=np.ones((3,3)), exposure_time_map=ccd_data.exposure_time_map, gain=None, convert_from_electrons=True)

        poisson_noise_map_converted = poisson_noise_map_counts / 6.0

        assert (ccd_data.image == np.ones((3,3)) / 6.0).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.poisson_noise_map == np.ones((3,3)) / 6.0)
        assert (ccd_data.poisson_noise_map == poisson_noise_map_converted).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3)) / 6.0).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__poisson_noise_map_from_image__include_convert_from_adus(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               poisson_noise_map_from_image=True, gain=2.0, convert_from_adus=True)

        poisson_noise_map_counts = ccd.PoissonNoiseMap.from_image_and_exposure_time_map(pixel_scale=0.1,
                                                                                        image=np.ones((3,3)), exposure_time_map=ccd_data.exposure_time_map, gain=2.0, convert_from_adus=True)

        poisson_noise_map_converted = 2.0 * poisson_noise_map_counts / 6.0

        assert (ccd_data.image == 2.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 2.0*3.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.background_noise_map == 2.0*4.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.poisson_noise_map == np.sqrt(2.0*np.ones((3,3))) / 6.0)
        assert (ccd_data.poisson_noise_map == poisson_noise_map_converted).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 2.0*7.0*np.ones((3,3)) / 6.0).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_poisson_noise_map_from_weight_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               convert_poisson_noise_map_from_weight_map=True)

        poisson_noise_map_converted = ccd.NoiseMap.from_weight_map(weight_map=5.0 * np.ones((3, 3)), pixel_scale=0.1)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == poisson_noise_map_converted).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_poisson_noise_map_from_inverse_noise_map(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               convert_poisson_noise_map_from_inverse_noise_map=True)

        poisson_noise_map_converted = ccd.NoiseMap.from_inverse_noise_map(inverse_noise_map=5.0 * np.ones((3, 3)),
                                                                          pixel_scale=0.1)

        assert (ccd_data.image == np.ones((3, 3))).all()
        assert (ccd_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == poisson_noise_map_converted).all()
        assert (ccd_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()
        assert (ccd_data.background_sky_map == 7.0 * np.ones((3, 3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__psf_renormalized_true__renormalized_psf(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=True)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert ccd_data.psf == pytest.approx((1.0/9.0)*np.ones((3,3)), 1e-2)
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_image_from_electrons_using_exposure_time(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               convert_from_electrons=True)

        assert (ccd_data.image == np.ones((3,3)) / 6.0).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3)) / 6.0).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__convert_image_from_adus_using_exposure_time_and_gain(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                               psf_path=test_data_dir + '3x3_twos.fits',
                                               noise_map_path=test_data_dir + '3x3_threes.fits',
                                               background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                               poisson_noise_map_path=test_data_dir + '3x3_fives.fits',
                                               exposure_time_map_path=test_data_dir + '3x3_sixes.fits',
                                               background_sky_map_path=test_data_dir + '3x3_sevens.fits',
                                               renormalize_psf=False,
                                               gain=2.0, convert_from_adus=True)

        assert (ccd_data.image == 2.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 2.0*3.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.background_noise_map == 2.0*4.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.poisson_noise_map == 2.0*5.0*np.ones((3,3)) / 6.0).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 2.0*7.0*np.ones((3,3)) / 6.0).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1

    def test__no_noise_map_input__raises_imaging_exception(self):

        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        psf_path=test_data_dir + '3x3_twos.fits')

    def test__multiple_noise_map_options__raises_imaging_exception(self):

        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        psf_path=test_data_dir + '3x3_twos.fits',
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        convert_noise_map_from_inverse_noise_map=True,
                                        convert_noise_map_from_weight_map=True)

        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        psf_path=test_data_dir + '3x3_twos.fits',
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        convert_noise_map_from_inverse_noise_map=True,
                                        noise_map_from_image_and_background_noise_map=True)

        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        psf_path=test_data_dir + '3x3_twos.fits',
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        noise_map_from_image_and_background_noise_map=True,
                                        convert_noise_map_from_weight_map=True)

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):

        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                        psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        exposure_time_map_path=test_data_dir + '3x3_ones.fits',
                                        exposure_time_map_from_single_value=1.0)

    def test__noise_map_from_image_and_background_noise_map_exceptions(self):

        # need background noise_map map - raise error if not present
        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                        psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        exposure_time_map_from_single_value=1.0,
                                        noise_map_from_image_and_background_noise_map=True)

        # Dont need gain if datas is in electrons
        ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                    psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                    noise_map_path=test_data_dir + '3x3_threes.fits',
                                    background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                    exposure_time_map_from_single_value=1.0,
                                    noise_map_from_image_and_background_noise_map=True,
                                    convert_from_electrons=True)

        # Need gain if datas is in adus
        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                        psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                        noise_map_from_image_and_background_noise_map=True,
                                        convert_from_adus=True)

        # No error if datas already in adus
        ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                    psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                    noise_map_path=test_data_dir + '3x3_threes.fits',
                                    background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                    exposure_time_map_from_single_value=1.0,
                                    noise_map_from_image_and_background_noise_map=True,
                                    gain=1.0,
                                    convert_from_adus=True)

    def test__poisson_noise_map_from_image_exceptions(self):

        # Dont need gain if datas is in e/s
        ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                    psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                    noise_map_path=test_data_dir + '3x3_threes.fits',
                                    exposure_time_map_from_single_value=1.0,
                                    poisson_noise_map_from_image=True)

        # No exposure time - not load
        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                        psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        poisson_noise_map_from_image=True,
                                        convert_from_electrons=True)

        # Need gain if datas in adus
        with pytest.raises(exc.ImagingException):
            ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits',
                                        psf_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
                                        noise_map_path=test_data_dir + '3x3_threes.fits',
                                        background_noise_map_path=test_data_dir + '3x3_fours.fits',
                                        exposure_time_map_from_single_value=1.0,
                                        poisson_noise_map_from_image=True,
                                        convert_from_adus=True)

    def test__output_all_arrays(self):

        ccd_data = ccd.load_ccd_data_from_fits(image_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1,
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

        ccd.output_ccd_data_to_fits(ccd_data=ccd_data,
                                    image_path=output_data_dir + 'image.fits',
                                    psf_path=output_data_dir + 'psf.fits',
                                    noise_map_path=output_data_dir + 'noise_map.fits',
                                    background_noise_map_path=output_data_dir + 'background_noise_map.fits',
                                    poisson_noise_map_path=output_data_dir + 'poisson_noise_map.fits',
                                    exposure_time_map_path=output_data_dir + 'exposure_time_map.fits',
                                    background_sky_map_path=output_data_dir + 'background_sky_map.fits')

        ccd_data = ccd.load_ccd_data_from_fits(image_path=output_data_dir + 'image.fits', pixel_scale=0.1,
                                               psf_path=output_data_dir + 'psf.fits',
                                               noise_map_path=output_data_dir + 'noise_map.fits',
                                               background_noise_map_path=output_data_dir + 'background_noise_map.fits',
                                               poisson_noise_map_path=output_data_dir + 'poisson_noise_map.fits',
                                               exposure_time_map_path=output_data_dir + 'exposure_time_map.fits',
                                               background_sky_map_path=output_data_dir + 'background_sky_map.fits',
                                               renormalize_psf=False)

        assert (ccd_data.image == np.ones((3,3))).all()
        assert (ccd_data.psf == 2.0*np.ones((3,3))).all()
        assert (ccd_data.noise_map == 3.0*np.ones((3,3))).all()
        assert (ccd_data.background_noise_map == 4.0*np.ones((3,3))).all()
        assert (ccd_data.poisson_noise_map == 5.0*np.ones((3,3))).all()
        assert (ccd_data.exposure_time_map == 6.0*np.ones((3,3))).all()
        assert (ccd_data.background_sky_map == 7.0*np.ones((3,3))).all()

        assert ccd_data.pixel_scale == 0.1
        assert ccd_data.psf.pixel_scale == 0.1
        assert ccd_data.noise_map.pixel_scale == 0.1
        assert ccd_data.background_noise_map.pixel_scale == 0.1
        assert ccd_data.poisson_noise_map.pixel_scale == 0.1
        assert ccd_data.exposure_time_map.pixel_scale == 0.1
        assert ccd_data.background_sky_map.pixel_scale == 0.1


class TestPositionsToFile(object):

    def test__load_positions__retains_list_structure(self):

        positions = ccd.load_positions(positions_path=test_positions_dir + 'positions_test.dat')

        assert positions == [[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0], [5.0, 6.0]]]

    def test__output_positions(self):

        positions = [[[4.0, 4.0], [5.0, 5.0]], [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]]

        output_data_dir = "{}/../test_files/positions/output_test/".format(os.path.dirname(os.path.realpath(__file__)))
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        ccd.output_positions(positions=positions, positions_path=output_data_dir+'positions_test.dat')

        positions = ccd.load_positions(positions_path=output_data_dir+'positions_test.dat')

        assert positions == [[[4.0, 4.0], [5.0, 5.0]], [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]]