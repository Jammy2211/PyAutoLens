import os
import shutil

import numpy as np
import pytest

from autolens import exc
from autolens.data.array import grids
from autolens.data.array import scaled_array
from autolens.data import ccd
from autolens.data import simulated_ccd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))
test_positions_dir = "{}/../test_files/positions/".format(os.path.dirname(os.path.realpath(__file__)))


class TestSimulateCCD(object):

    def test__setup_image__correct_attributes(self):

        array = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])

        psf = ccd.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)
        noise_map = 5.0 * np.ones((3, 3))

        ccd_data = simulated_ccd.SimulatedCCDData(image=array, pixel_scale=0.1, noise_map=noise_map, psf=psf,
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

    def test__setup_with_all_features_off(self):

        image = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, exposure_time=1.0, exposure_time_map=exposure_time_map, pixel_scale=0.1, add_noise=False)

        assert (ccd_data_simulated.exposure_time_map == np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1
        assert (ccd_data_simulated.image == np.array([[0.0, 0.0, 0.0],
                                                         [0.0, 1.0, 0.0],
                                                         [0.0, 0.0, 0.0]])).all()
        assert ccd_data_simulated.origin == (0.0, 0.0)

    def test__setup_with_background_sky_on__noise_off__no_noise_in_image__noise_map_is_noise_value(self):
        image = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1,
                                                                        shape=image.shape)

        background_sky_map =  scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1,
                                                                          shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=1.0, exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map, add_noise=False, noise_if_add_noise_false=0.2, noise_seed=1)

        assert (ccd_data_simulated.exposure_time_map == 1.0 * np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1

        assert (ccd_data_simulated.image == np.array([[0.0, 0.0, 0.0],
                                                         [0.0, 1.0, 0.0],
                                                         [0.0, 0.0, 0.0]])).all()
        assert (ccd_data_simulated.noise_map == 0.2*np.ones((3,3)))

        assert (ccd_data_simulated.background_noise_map == 4.0 * np.ones((3, 3))).all()

    def test__setup_with_background_sky_on__noise_on_so_background_adds_noise_to_image(self):
        image = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

        background_sky_map =  scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=1.0, exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map, add_noise=True, noise_seed=1)

        assert (ccd_data_simulated.exposure_time_map == 1.0 * np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1

        assert (ccd_data_simulated.image == np.array([[1.0, 5.0, 4.0],
                                                         [1.0, 2.0, 1.0],
                                                         [5.0, 2.0, 7.0]])).all()

        assert (ccd_data_simulated.poisson_noise_map == np.array([[np.sqrt(1.0), np.sqrt(5.0), np.sqrt(4.0)],
                                                       [np.sqrt(1.0), np.sqrt(2.0), np.sqrt(1.0)],
                                                       [np.sqrt(5.0), np.sqrt(2.0), np.sqrt(7.0)]])).all()

        assert (ccd_data_simulated.background_noise_map == 4.0 * np.ones((3, 3))).all()

    def test__setup_with_psf_blurring_on__blurs_image_and_trims_psf_edge_off(self):
        image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])

        psf = ccd.PSF(array=np.array([[0.0, 1.0, 0.0],
                                      [1.0, 2.0, 1.0],
                                      [0.0, 1.0, 0.0]]), pixel_scale=1.0)

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=1.0, exposure_time_map=exposure_time_map, psf=psf,
            add_noise=False)

        assert (ccd_data_simulated.image == np.array([[0.0, 1.0, 0.0],
                                                         [1.0, 2.0, 1.0],
                                                         [0.0, 1.0, 0.0]])).all()
        assert (ccd_data_simulated.exposure_time_map == np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1

    def test__setup_with_background_sky_and_psf_on__psf_does_no_blurring__image_and_sky_both_trimmed(self):
        image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])

        psf = ccd.PSF(array=np.array([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]]), pixel_scale=1.0)

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

        background_sky_map =  scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=1.0, exposure_time_map=exposure_time_map,
            psf=psf, background_sky_map=background_sky_map, add_noise=False, noise_seed=1)

        assert (ccd_data_simulated.exposure_time_map == 1.0 * np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1

        assert (ccd_data_simulated.image == np.array([[0.0, 0.0, 0.0],
                                                         [0.0, 1.0, 0.0],
                                                         [0.0, 0.0, 0.0]])).all()

        assert (ccd_data_simulated.background_noise_map == 4.0 * np.ones((3, 3))).all()

    def test__setup_with_noise(self):
        image = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]])

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=20.0, exposure_time_map=exposure_time_map,
            add_noise=True, noise_seed=1)

        assert (ccd_data_simulated.exposure_time_map == 20.0 * np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1

        assert ccd_data_simulated.image == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                      [0.0, 1.05, 0.0],
                                                                      [0.0, 0.0, 0.0]]), 1e-2)

        # Because of the regular value is 1.05, the estimated Poisson noise_map_1d is:
        # sqrt((1.05 * 20))/20 = 0.2291

        assert ccd_data_simulated.poisson_noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
                                                                    [0.0, 0.2291, 0.0],
                                                                    [0.0, 0.0, 0.0]]), 1e-2)

        assert ccd_data_simulated.noise_map == pytest.approx(np.array([[0.0, 0.0, 0.0],
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

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=20.0, pixel_scale=0.1, shape=image.shape)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=20.0, exposure_time_map=exposure_time_map,
            psf=psf, add_noise=True, noise_seed=1)

        assert (ccd_data_simulated.exposure_time_map == 20.0 * np.ones((3, 3))).all()
        assert ccd_data_simulated.pixel_scale == 0.1
        assert ccd_data_simulated.image == pytest.approx(np.array([[0.0, 1.05, 0.0],
                                                                      [1.3, 2.35, 1.05],
                                                                      [0.0, 1.05, 0.0]]), 1e-2)

        # The estimated Poisson noises are:
        # sqrt((2.35 * 20))/20 = 0.3427
        # sqrt((1.3 * 20))/20 = 0.2549
        # sqrt((1.05 * 20))/20 = 0.2291

        assert ccd_data_simulated.poisson_noise_map == pytest.approx(np.array([[0.0, 0.2291, 0.0],
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

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1, shape=image.shape)

        background_sky_map =  scaled_array.ScaledSquarePixelArray.single_value(value=16.0, pixel_scale=0.1, shape=image.shape)

        simulated_ccd_variable = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, exposure_time=1.0, exposure_time_map=exposure_time_map, psf=psf,
            background_sky_map=background_sky_map, pixel_scale=0.1, add_noise=False, noise_seed=1)

        image = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=image, pixel_scale=0.1, exposure_time=1.0, background_sky_level=16.0, psf=psf, add_noise=False,
            noise_seed=1)

        assert (simulated_ccd_variable.exposure_time_map == ccd_data_simulated.exposure_time_map).all()
        assert simulated_ccd_variable.pixel_scale == ccd_data_simulated.pixel_scale
        assert simulated_ccd_variable.image == pytest.approx(ccd_data_simulated.image, 1e-4)
        assert (simulated_ccd_variable.background_noise_map == ccd_data_simulated.background_noise_map).all()

    def test__noise_map_creates_nans_due_to_low_exposure_time__raises_error(self):

        image = np.ones((9, 9))

        psf = ccd.PSF.from_gaussian(shape=(3, 3), sigma=0.1, pixel_scale=0.2)

        exposure_time_map = scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1,
                                                                        shape=image.shape)

        background_sky_map =  scaled_array.ScaledSquarePixelArray.single_value(value=1.0, pixel_scale=0.1,
                                                                          shape=image.shape)

        with pytest.raises(exc.DataException):
            simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
                image=image, psf=psf, pixel_scale=0.1, exposure_time=1.0,
                exposure_time_map=exposure_time_map, background_sky_map=background_sky_map,
                add_noise=True, noise_seed=1)

    def test__from_deflections_and_source_galaxies__same_as_manual_calculation_using_tracer(self):

        psf = ccd.PSF(array=np.array([[0.0, 1.0, 0.0],
                                      [1.0, 2.0, 1.0],
                                      [0.0, 1.0, 0.0]]), pixel_scale=1.0)

        image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
            shape=(10, 10), pixel_scale=1.0, psf_shape=psf.shape, sub_grid_size=1)

        g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

        g1 = g.Galaxy(redshift=1.0, light=lp.SphericalSersic(intensity=1.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grid_stack=image_plane_grid_stack)

        deflections = np.stack((tracer.deflections_y, tracer.deflections_x), axis=-1)

        ccd_data_simulated_via_deflections = simulated_ccd.SimulatedCCDData.from_deflections_source_galaxies_and_exposure_arrays(
            deflections=deflections, pixel_scale=1.0, source_galaxies=[g1], exposure_time=10000.0,
            background_sky_level=100.0, add_noise=True, noise_seed=1)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=tracer.profile_image_plane_image_2d, pixel_scale=1.0, exposure_time=10000.0,
            background_sky_level=100.0, add_noise=True, noise_seed=1)

        assert (ccd_data_simulated_via_deflections.image == ccd_data_simulated.image).all()
        assert (ccd_data_simulated_via_deflections.psf == ccd_data_simulated.psf).all()
        assert (ccd_data_simulated_via_deflections.noise_map == ccd_data_simulated.noise_map).all()
        assert (ccd_data_simulated_via_deflections.background_sky_map == ccd_data_simulated.background_sky_map).all()
        assert (ccd_data_simulated_via_deflections.exposure_time_map == ccd_data_simulated.exposure_time_map).all()


    def test__from_tracer__same_as_manual_tracer_input(self):

        psf = ccd.PSF(array=np.array([[0.0, 1.0, 0.0],
                                      [1.0, 2.0, 1.0],
                                      [0.0, 1.0, 0.0]]), pixel_scale=1.0)

        image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
            shape=(20, 20), pixel_scale=0.05, psf_shape=psf.shape, sub_grid_size=1)

        lens_galaxy = g.Galaxy(
            redshift=0.5, light=lp.EllipticalSersic(intensity=1.0), mass=mp.EllipticalIsothermal(einstein_radius=1.6))

        source_galaxy = g.Galaxy(
            redshift=1.0,light=lp.EllipticalSersic(intensity=0.3))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                     image_plane_grid_stack=image_plane_grid_stack)

        ccd_data_simulated_via_tracer = simulated_ccd.SimulatedCCDData.from_tracer_and_exposure_arrays(
            tracer=tracer, pixel_scale=0.1, exposure_time=10000.0, psf=psf, background_sky_level=100.0,
            add_noise=True, noise_seed=1)

        ccd_data_simulated = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
            image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1, exposure_time=10000.0, psf=psf,
            background_sky_level=100.0, add_noise=True, noise_seed=1)

        assert (ccd_data_simulated_via_tracer.image == ccd_data_simulated.image).all()
        assert (ccd_data_simulated_via_tracer.psf == ccd_data_simulated.psf).all()
        assert (ccd_data_simulated_via_tracer.noise_map == ccd_data_simulated.noise_map).all()
        assert (ccd_data_simulated_via_tracer.background_sky_map == ccd_data_simulated.background_sky_map).all()
        assert (ccd_data_simulated_via_tracer.exposure_time_map == ccd_data_simulated.exposure_time_map).all()


class TestSimulatePoissonNoise(object):

    def test__input_image_all_0s__exposure_time_all_1s__all_noise_values_are_0s(self):

        image = np.zeros((2, 2))

        exposure_time = scaled_array.ScaledSquarePixelArray.single_value(1.0, image.shape, pixel_scale=0.1)
        simulated_poisson_image = image + simulated_ccd.generate_poisson_noise(image, exposure_time, seed=1)

        assert simulated_poisson_image.shape == (2, 2)
        assert (simulated_poisson_image == np.zeros((2, 2))).all()

    def test__input_image_includes_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(self):
        image = np.array([[10., 0.],
                        [0., 10.]])

        exposure_time = scaled_array.ScaledSquarePixelArray.single_value(1.0, image.shape, pixel_scale=0.1)
        poisson_noise_map = simulated_ccd.generate_poisson_noise(image, exposure_time, seed=1)
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
        poisson_noise_map = simulated_ccd.generate_poisson_noise(image, exposure_time, seed=1)
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

        poisson_noise_map = simulated_ccd.generate_poisson_noise(image, exposure_time, seed=2)

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

        simulated_poisson_image_0 = image_0 + simulated_ccd.generate_poisson_noise(image_0, exposure_time_0, seed=1)
        simulated_poisson_image_1 = image_1 + simulated_ccd.generate_poisson_noise(image_1, exposure_time_1, seed=1)

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

        simulated_poisson_image_0 = image_0 + simulated_ccd.generate_poisson_noise(image_0, exposure_time_0, seed=1)
        simulated_poisson_image_1 = image_1 + simulated_ccd.generate_poisson_noise(image_1, exposure_time_1, seed=1)

        assert (simulated_poisson_image_0[0, 0] == simulated_poisson_image_1[0, 0] / 2.0).all()
        assert simulated_poisson_image_0[0, 1] == simulated_poisson_image_1[0, 1]
        assert (simulated_poisson_image_0[1, 0] * 1.5 == pytest.approx(simulated_poisson_image_1[1, 0], 1e-2)).all()
        assert (simulated_poisson_image_0[1, 1] / 2.0 == simulated_poisson_image_1[1, 1]).all()