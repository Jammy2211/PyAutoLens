import os
import shutil

import numpy as np
import pytest

from autolens import exc
from autolens.data.array import scaled_array
from autolens.data.instrument import abstract_data
from autolens.data.instrument import interferometer
from autolens.data.array.util import grid_util, mapping_util

test_data_dir = "{}/../../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestInterferometerData(object):
    class TestNewInterferometerDataResized:
        def test__all_components_resized__psf_and_primary_beam_are_not(self):
            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            image_array[3, 3] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            noise_map_array[3, 3] = 3.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            exposure_time_map_array[3, 3] = 5.0

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=interferometer.PrimaryBeam(
                    np.zeros((5, 5)), pixel_scale=1.0
                ),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_with_resized_arrays(
                new_shape=(4, 4)
            )

            assert (
                interferometer_data.image
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 2.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()
            assert (
                interferometer_data.noise_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 3.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()

            assert (
                interferometer_data.exposure_time_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 5.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()

            assert interferometer_data.pixel_scale == 1.0
            assert (interferometer_data.psf == np.zeros((3, 3))).all()
            assert (interferometer_data.primary_beam == np.zeros((5, 5))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__resize_psf(self):
            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=interferometer.PrimaryBeam(
                    np.zeros((5, 5)), pixel_scale=1.0
                ),
                noise_map=1,
                exposure_time_map=1,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_with_resized_psf(
                new_shape=(1, 1)
            )

            assert (interferometer_data.image == np.ones((6, 6))).all()
            assert interferometer_data.pixel_scale == 1.0
            assert (interferometer_data.psf == np.zeros((1, 1))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__resize_primary_beam(self):
            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=1,
                primary_beam=interferometer.PrimaryBeam(
                    np.zeros((5, 5)), pixel_scale=1.0
                ),
                noise_map=1,
                exposure_time_map=1,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_with_resized_primary_beam(
                new_shape=(1, 1)
            )

            assert (interferometer_data.image == np.ones((6, 6))).all()
            assert interferometer_data.pixel_scale == 1.0
            assert (interferometer_data.primary_beam == np.zeros((1, 1))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__input_new_centre_pixels__arrays_use_new_centre__psf_and_primary_beam_do_not(
            self
        ):
            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            image_array[3, 3] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            noise_map_array[3, 3] = 3.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            exposure_time_map_array[3, 3] = 5.0

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=interferometer.PrimaryBeam(
                    np.zeros((5, 5)), pixel_scale=1.0
                ),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_with_resized_arrays(
                new_shape=(3, 3), new_centre_pixels=(3, 3)
            )

            assert (
                interferometer_data.image
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert (
                interferometer_data.noise_map
                == np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()

            assert (
                interferometer_data.exposure_time_map
                == np.array([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()

            assert interferometer_data.pixel_scale == 1.0
            assert (interferometer_data.psf == np.zeros((3, 3))).all()
            assert (interferometer_data.primary_beam == np.zeros((5, 5))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__input_new_centre_arcsec__arrays_use_new_centre__psf_and_primary_beam_do_not(
            self
        ):
            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            image_array[3, 3] = 2.0

            noise_map_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            noise_map_array[3, 3] = 3.0

            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            exposure_time_map_array[3, 3] = 5.0

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=interferometer.PrimaryBeam(
                    np.zeros((5, 5)), pixel_scale=1.0
                ),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_with_resized_arrays(
                new_shape=(3, 3), new_centre_arcsec=(-0.5, 0.5)
            )

            assert (
                interferometer_data.image
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert (
                interferometer_data.noise_map
                == np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert (
                interferometer_data.exposure_time_map
                == np.array([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()

            assert interferometer_data.pixel_scale == 1.0
            assert (interferometer_data.psf == np.zeros((3, 3))).all()
            assert (interferometer_data.primary_beam == np.zeros((5, 5))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__input_both_centres__raises_error(self):
            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=interferometer.PrimaryBeam(
                    np.zeros((5, 5)), pixel_scale=1.0
                ),
                noise_map=1,
                exposure_time_map=1,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            with pytest.raises(exc.DataException):
                interferometer_data.new_interferometer_data_with_resized_arrays(
                    new_shape=(3, 3),
                    new_centre_pixels=(3, 3),
                    new_centre_arcsec=(-0.5, 0.5),
                )

    class TestNewImageConvertedFrom:
        def test__counts__all_arrays_in_units_of_flux_are_converted(self):

            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((3, 3)), pixel_scale=1.0
            )
            noise_map_array = scaled_array.ScaledSquarePixelArray(
                2.0 * np.ones((3, 3)), pixel_scale=1.0
            )
            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(
                0.5 * np.ones((3, 3)), pixel_scale=1.0
            )

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                primary_beam=1,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = (
                interferometer_data.new_interferometer_data_converted_from_electrons()
            )

            assert (interferometer_data.image == 2.0 * np.ones((3, 3))).all()
            assert (interferometer_data.noise_map == 4.0 * np.ones((3, 3))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__adus__all_arrays_in_units_of_flux_are_converted(self):

            image_array = scaled_array.ScaledSquarePixelArray(
                np.ones((3, 3)), pixel_scale=1.0
            )
            noise_map_array = scaled_array.ScaledSquarePixelArray(
                2.0 * np.ones((3, 3)), pixel_scale=1.0
            )
            exposure_time_map_array = scaled_array.ScaledSquarePixelArray(
                0.5 * np.ones((3, 3)), pixel_scale=1.0
            )

            interferometer_data = interferometer.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=abstract_data.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                primary_beam=1,
                real_visibilities=1,
                imaginary_visibilities=1,
                visibilities_noise_map=1,
                u_wavelengths=1,
                v_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_converted_from_adus(
                gain=2.0
            )

            assert (interferometer_data.image == 2.0 * 2.0 * np.ones((3, 3))).all()
            assert (interferometer_data.noise_map == 2.0 * 4.0 * np.ones((3, 3))).all()
            assert interferometer_data.origin == (0.0, 0.0)


class TestPrimaryBeam(object):
    class TestConstructors(object):
        def test__init__input_primary_beam__all_attributes_correct_including_data_inheritance(
            self
        ):
            psf = interferometer.PrimaryBeam(
                array=np.ones((3, 3)), pixel_scale=1.0, renormalize=False
            )

            assert psf.shape == (3, 3)
            assert psf.pixel_scale == 1.0
            assert (psf == np.ones((3, 3))).all()
            assert psf.origin == (0.0, 0.0)

            psf = interferometer.PrimaryBeam(
                array=np.ones((4, 3)), pixel_scale=1.0, renormalize=False
            )

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.shape == (4, 3)
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_primary_beam_3x3__all_attributes_correct_including_data_inheritance(
            self
        ):
            psf = interferometer.PrimaryBeam.from_fits_with_scale(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scale=1.0
            )

            assert (psf == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

            psf = interferometer.PrimaryBeam.from_fits_with_scale(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scale=1.0
            )

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

    class TestRenormalize(object):
        def test__input_is_already_normalized__no_change(self):
            primary_beam_data = np.ones((3, 3)) / 9.0

            psf = interferometer.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=True
            )

            assert psf == pytest.approx(primary_beam_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            primary_beam_data = np.ones((3, 3))

            psf = interferometer.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=True
            )

            assert psf == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__same_as_above__renomalized_false_does_not_renormalize(self):
            primary_beam_data = np.ones((3, 3))

            psf = interferometer.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=False
            )

            assert psf == pytest.approx(np.ones((3, 3)), 1e-3)

    class TestBinnedUp(object):
        def test__primary_beam_is_even_x_even__rescaled_to_odd_x_odd__no_use_of_dimension_trimming(
            self
        ):

            array_2d = np.ones((6, 6))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((9, 9))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.333333333333333, renormalize=True
            )
            assert psf.pixel_scale == 3.0
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((18, 6))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 27.0) * np.ones((9, 3))

            array_2d = np.ones((6, 18))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 27.0) * np.ones((3, 9))

        def test__primary_beam_is_even_x_even_after_binning_up__resized_to_odd_x_odd_with_shape_plus_one(
            self
        ):

            array_2d = np.ones((2, 2))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == 0.4
            assert psf == (1.0 / 25.0) * np.ones((5, 5))

            array_2d = np.ones((40, 40))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.1, renormalize=True
            )
            assert psf.pixel_scale == 8.0
            assert psf == (1.0 / 25.0) * np.ones((5, 5))

            array_2d = np.ones((2, 4))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0 / 45.0) * np.ones((5, 9))

            array_2d = np.ones((4, 2))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0 / 45.0) * np.ones((9, 5))

        def test__primary_beam_is_odd_and_even_after_binning_up__resized_to_odd_and_odd_with_shape_plus_one(
            self
        ):

            array_2d = np.ones((6, 4))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((9, 12))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.33333333333, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((3, 5))

            array_2d = np.ones((4, 6))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((12, 9))
            psf = interferometer.PrimaryBeam(
                array=array_2d, pixel_scale=1.0, renormalize=False
            )
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.33333333333, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((5, 3))

    class TestNewRenormalizedPrimaryBeam(object):
        def test__input_is_already_normalized__no_change(self):

            primary_beam_data = np.ones((3, 3)) / 9.0

            psf = interferometer.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=False
            )

            primary_beam_new = psf.new_primary_beam_with_renormalized_array()

            assert primary_beam_new == pytest.approx(primary_beam_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            primary_beam_data = np.ones((3, 3))

            psf = interferometer.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=False
            )

            primary_beam_new = psf.new_primary_beam_with_renormalized_array()

            assert primary_beam_new == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    class TestConvolve(object):
        def test__kernel_is_not_odd_x_odd__raises_error(self):
            kernel = np.array([[0.0, 1.0], [1.0, 2.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            with pytest.raises(exc.KernelException):
                psf.convolve(np.ones((5, 5)))

        def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (blurred_image == kernel).all()

        def test__image_is_4x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ).all()

        def test__image_is_4x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
                )
            ).all()

        def test__image_is_3x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                )
            ).all()

        def test__image_is_4x4_has_two_central_values__kernel_is_asymmetric__blurred_image_follows_convolution(
            self
        ):
            image = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 0.0],
                        [2.0, 3.0, 2.0, 1.0],
                        [1.0, 5.0, 5.0, 1.0],
                        [0.0, 1.0, 3.0, 3.0],
                    ]
                )
            ).all()

        def test__image_is_4x4_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
            self
        ):
            image = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [2.0, 1.0, 1.0, 1.0],
                        [3.0, 3.0, 2.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                    ]
                )
            ).all()

        def test__image_is_4x4_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
            self
        ):
            image = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

            psf = interferometer.PrimaryBeam(array=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve(image)

            assert (
                blurred_image
                == np.array(
                    [
                        [2.0, 1.0, 0.0, 0.0],
                        [3.0, 3.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 2.0, 2.0],
                    ]
                )
            ).all()

    class TestFromGaussian(object):
        def test__identical_to_gaussian_light_profile(self):

            from autolens.model.profiles import light_profiles as lp

            grid = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
                mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=1
            )

            gaussian = lp.EllipticalGaussian(
                centre=(0.1, 0.1), axis_ratio=0.9, phi=45.0, intensity=1.0, sigma=1.0
            )
            profile_gaussian_1d = gaussian.intensities_from_grid(grid)
            profile_gaussian_2d = mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_grid_size(
                sub_array_1d=profile_gaussian_1d,
                mask=np.full(fill_value=False, shape=(3, 3)),
                sub_grid_size=1,
            )
            profile_psf = interferometer.PrimaryBeam(
                array=profile_gaussian_2d, pixel_scale=1.0, renormalize=True
            )

            imaging_psf = interferometer.PrimaryBeam.from_gaussian(
                shape=(3, 3),
                pixel_scale=1.0,
                centre=(0.1, 0.1),
                axis_ratio=0.9,
                phi=45.0,
                sigma=1.0,
            )

            assert profile_psf == pytest.approx(imaging_psf, 1e-4)


class TestInterferometerFromFits(object):
    def test__no_settings_just_pass_fits(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            visibilities_noise_map_path=test_data_dir + "3_threes.fits",
            u_wavelengths_path=test_data_dir + "3_fours.fits",
            v_wavelengths_path=test_data_dir + "3_fives.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (interferometer_data.real_visibilities == np.ones(3)).all()
        assert (interferometer_data.imaginary_visibilities == 2.0 * np.ones(3)).all()
        assert (interferometer_data.visibilities_noise_map == 3.0 * np.ones(3)).all()
        assert (interferometer_data.u_wavelengths == 4.0 * np.ones(3)).all()
        assert (interferometer_data.v_wavelengths == 5.0 * np.ones(3)).all()
        assert (interferometer_data.primary_beam == 5.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1

    def test__optional_array_paths_included__loads_optional_array(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (interferometer_data.primary_beam == 5.0 * np.ones((3, 3))).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1

    def test__all_files_in_one_fits__load_using_different_hdus(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_multiple_hdu.fits",
            image_hdu=0,
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_multiple_hdu.fits",
            psf_hdu=1,
            noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            noise_map_hdu=2,
            primary_beam_path=test_data_dir + "3x3_multiple_hdu.fits",
            primary_beam_hdu=3,
            exposure_time_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            exposure_time_map_hdu=5,
            real_visibilities_path=test_data_dir + "3_multiple_hdu.fits",
            real_visibilities_hdu=0,
            imaginary_visibilities_path=test_data_dir + "3_multiple_hdu.fits",
            imaginary_visibilities_hdu=1,
            visibilities_noise_map_path=test_data_dir + "3_multiple_hdu.fits",
            visibilities_noise_map_hdu=2,
            u_wavelengths_path=test_data_dir + "3_multiple_hdu.fits",
            u_wavelengths_hdu=3,
            v_wavelengths_path=test_data_dir + "3_multiple_hdu.fits",
            v_wavelengths_hdu=4,
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (interferometer_data.primary_beam == 4.0 * np.ones((3, 3))).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()
        assert (interferometer_data.real_visibilities == np.ones(3)).all()
        assert (interferometer_data.imaginary_visibilities == 2.0 * np.ones(3)).all()
        assert (interferometer_data.visibilities_noise_map == 3.0 * np.ones(3)).all()
        assert (interferometer_data.u_wavelengths == 4.0 * np.ones(3)).all()
        assert (interferometer_data.v_wavelengths == 5.0 * np.ones(3)).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(
        self
    ):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            noise_map_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_ones.fits",
            primary_beam_path=test_data_dir + "3x3_ones.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            pixel_scale=0.1,
            exposure_time_map_from_single_value=3.0,
            renormalize_psf=False,
        )

        assert (interferometer_data.exposure_time_map == 3.0 * np.ones((3, 3))).all()

    def test__pad_shape_of_image_arrays_and_psf(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            resized_interferometer_shape=(5, 5),
            resized_psf_shape=(7, 7),
            resized_primary_beam_shape=(9, 9),
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        exposure_padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 6.0, 6.0, 0.0],
                [0.0, 6.0, 6.0, 6.0, 0.0],
                [0.0, 6.0, 6.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        psf_padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        primary_beam_padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert (interferometer_data.image == padded_array).all()
        assert (interferometer_data.psf == psf_padded_array).all()
        assert (interferometer_data.noise_map == 3.0 * padded_array).all()
        assert (interferometer_data.exposure_time_map == 6.0 * padded_array).all()
        assert (interferometer_data.primary_beam == primary_beam_padded_array).all()
        assert (interferometer_data.exposure_time_map == exposure_padded_array).all()

        assert interferometer_data.pixel_scale == 0.1

    def test__trim_shape_of_image_arrays_and_psf(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            resized_interferometer_shape=(1, 1),
            resized_psf_shape=(1, 1),
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        trimmed_array = np.array([[1.0]])

        assert (interferometer_data.image == trimmed_array).all()
        assert (interferometer_data.psf == 2.0 * trimmed_array).all()
        assert (interferometer_data.noise_map == 3.0 * trimmed_array).all()
        assert (interferometer_data.primary_beam == 5.0 * trimmed_array).all()
        assert (interferometer_data.exposure_time_map == 6.0 * trimmed_array).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__convert_noise_map_from_weight_map(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            convert_noise_map_from_weight_map=True,
            renormalize_psf=False,
        )

        noise_map_converted = interferometer.NoiseMap.from_weight_map(
            weight_map=3.0 * np.ones((3, 3)), pixel_scale=0.1
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == noise_map_converted).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__convert_noise_map_from_inverse_noise_map(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            convert_noise_map_from_inverse_noise_map=True,
            renormalize_psf=False,
        )

        noise_map_converted = interferometer.NoiseMap.from_inverse_noise_map(
            inverse_noise_map=3.0 * np.ones((3, 3)), pixel_scale=0.1
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == noise_map_converted).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__psf_renormalized_true__renormalized_psf(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_psf=True,
            renormalize_primary_beam=False,
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert interferometer_data.psf == pytest.approx(
            (1.0 / 9.0) * np.ones((3, 3)), 1e-2
        )
        assert (interferometer_data.primary_beam == 5.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__primary_beam_renormalized_true__renormalized_psf(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_psf=False,
            renormalize_primary_beam=True,
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert interferometer_data.primary_beam == pytest.approx(
            (1.0 / 9.0) * np.ones((3, 3)), 1e-2
        )
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__convert_image_from_electrons_using_exposure_time(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_psf=False,
            renormalize_primary_beam=False,
            convert_from_electrons=True,
        )

        assert (interferometer_data.image == np.ones((3, 3)) / 6.0).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3)) / 6.0).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__convert_image_from_adus_using_exposure_time_and_gain(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_psf=False,
            gain=2.0,
            convert_from_adus=True,
        )

        assert (interferometer_data.image == 2.0 * np.ones((3, 3)) / 6.0).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (
            interferometer_data.noise_map == 2.0 * 3.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__no_noise_map_input__raises_imaging_exception(self):

        with pytest.raises(exc.DataException):
            interferometer.load_interferometer_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
            )

    def test__multiple_noise_map_options__raises_imaging_exception(self):

        with pytest.raises(exc.DataException):
            interferometer.load_interferometer_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
                noise_map_path=test_data_dir + "3x3_threes.fits",
                convert_noise_map_from_inverse_noise_map=True,
                convert_noise_map_from_weight_map=True,
            )

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):

        with pytest.raises(exc.DataException):
            interferometer.load_interferometer_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                exposure_time_map_path=test_data_dir + "3x3_ones.fits",
                exposure_time_map_from_single_value=1.0,
            )

    def test__output_all_arrays(self):

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            visibilities_noise_map_path=test_data_dir + "3_threes.fits",
            u_wavelengths_path=test_data_dir + "3_fours.fits",
            v_wavelengths_path=test_data_dir + "3_fives.fits",
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        output_data_dir = "{}/../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        interferometer.output_interferometer_data_to_fits(
            interferometer_data=interferometer_data,
            image_path=output_data_dir + "image.fits",
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            real_visibilities_path=output_data_dir + "real_visibilities.fits",
            imaginary_visibilities_path=output_data_dir + "imaginary_visibilities.fits",
            visibilities_noise_map_path=output_data_dir + "visibilities_noise_map.fits",
            u_wavelengths_path=output_data_dir + "u_wavelengths.fits",
            v_wavelengths_path=output_data_dir + "v_wavelengths.fits",
            overwrite=True,
        )

        interferometer_data = interferometer.load_interferometer_data_from_fits(
            image_path=output_data_dir + "image.fits",
            pixel_scale=0.1,
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            real_visibilities_path=output_data_dir + "real_visibilities.fits",
            imaginary_visibilities_path=output_data_dir + "imaginary_visibilities.fits",
            visibilities_noise_map_path=output_data_dir + "visibilities_noise_map.fits",
            u_wavelengths_path=output_data_dir + "u_wavelengths.fits",
            v_wavelengths_path=output_data_dir + "v_wavelengths.fits",
            renormalize_psf=False,
            renormalize_primary_beam=False,
        )

        assert (interferometer_data.image == np.ones((3, 3))).all()
        assert (interferometer_data.psf == 2.0 * np.ones((3, 3))).all()
        assert (interferometer_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (interferometer_data.primary_beam == 5.0 * np.ones((3, 3))).all()
        assert (interferometer_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()
        assert (interferometer_data.real_visibilities == np.ones(3)).all()
        assert (interferometer_data.imaginary_visibilities == 2.0 * np.ones(3)).all()
        assert (interferometer_data.visibilities_noise_map == 3.0 * np.ones(3)).all()
        assert (interferometer_data.u_wavelengths == 4.0 * np.ones(3)).all()
        assert (interferometer_data.v_wavelengths == 5.0 * np.ones(3)).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1
