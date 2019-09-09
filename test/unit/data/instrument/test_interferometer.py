import autolens as al
import os
import shutil

import numpy as np
import pytest
from autolens import exc

test_data_dir = "{}/../../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestInterferometerData(object):
    class TestNewInterferometerDataResized:
        def test__all_components_resized__psf_and_primary_beam_are_not(self):
            image_array = al.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3, 3] = 2.0

            noise_map_array = al.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            noise_map_array[3, 3] = 3.0

            exposure_time_map_array = al.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            exposure_time_map_array[3, 3] = 5.0

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=al.PrimaryBeam(np.zeros((5, 5)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
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
            image_array = al.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=al.PrimaryBeam(np.zeros((5, 5)), pixel_scale=1.0),
                noise_map=1,
                exposure_time_map=1,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
            )

            interferometer_data = interferometer_data.new_interferometer_data_with_resized_psf(
                new_shape=(1, 1)
            )

            assert (interferometer_data.image == np.ones((6, 6))).all()
            assert interferometer_data.pixel_scale == 1.0
            assert (interferometer_data.psf == np.zeros((1, 1))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__resize_primary_beam(self):
            image_array = al.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=1,
                primary_beam=al.PrimaryBeam(np.zeros((5, 5)), pixel_scale=1.0),
                noise_map=1,
                exposure_time_map=1,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
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
            image_array = al.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3, 3] = 2.0

            noise_map_array = al.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            noise_map_array[3, 3] = 3.0

            exposure_time_map_array = al.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            exposure_time_map_array[3, 3] = 5.0

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=al.PrimaryBeam(np.zeros((5, 5)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
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
            image_array = al.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            image_array[3, 3] = 2.0

            noise_map_array = al.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            noise_map_array[3, 3] = 3.0

            exposure_time_map_array = al.ScaledSquarePixelArray(
                np.ones((6, 6)), pixel_scale=1.0
            )
            exposure_time_map_array[3, 3] = 5.0

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=al.PrimaryBeam(np.zeros((5, 5)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
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
            image_array = al.ScaledSquarePixelArray(np.ones((6, 6)), pixel_scale=1.0)
            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                primary_beam=al.PrimaryBeam(np.zeros((5, 5)), pixel_scale=1.0),
                noise_map=1,
                exposure_time_map=1,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
            )

            with pytest.raises(exc.DataException):
                interferometer_data.new_interferometer_data_with_resized_arrays(
                    new_shape=(3, 3),
                    new_centre_pixels=(3, 3),
                    new_centre_arcsec=(-0.5, 0.5),
                )

    class TestNewImageConvertedFrom:
        def test__counts__all_arrays_in_units_of_flux_are_converted(self):

            image_array = al.ScaledSquarePixelArray(np.ones((3, 3)), pixel_scale=1.0)
            noise_map_array = al.ScaledSquarePixelArray(
                2.0 * np.ones((3, 3)), pixel_scale=1.0
            )
            exposure_time_map_array = al.ScaledSquarePixelArray(
                0.5 * np.ones((3, 3)), pixel_scale=1.0
            )

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                primary_beam=1,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
            )

            interferometer_data = (
                interferometer_data.new_interferometer_data_converted_from_electrons()
            )

            assert (interferometer_data.image == 2.0 * np.ones((3, 3))).all()
            assert (interferometer_data.noise_map == 4.0 * np.ones((3, 3))).all()
            assert interferometer_data.origin == (0.0, 0.0)

        def test__adus__all_arrays_in_units_of_flux_are_converted(self):

            image_array = al.ScaledSquarePixelArray(np.ones((3, 3)), pixel_scale=1.0)
            noise_map_array = al.ScaledSquarePixelArray(
                2.0 * np.ones((3, 3)), pixel_scale=1.0
            )
            exposure_time_map_array = al.ScaledSquarePixelArray(
                0.5 * np.ones((3, 3)), pixel_scale=1.0
            )

            interferometer_data = al.InterferometerData(
                image=image_array,
                pixel_scale=1.0,
                psf=al.PSF(np.zeros((3, 3)), pixel_scale=1.0),
                noise_map=noise_map_array,
                exposure_time_map=exposure_time_map_array,
                primary_beam=1,
                visibilities=np.array([[1, 1]]),
                visibilities_noise_map=1,
                uv_wavelengths=1,
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
            psf = al.PrimaryBeam(
                array=np.ones((3, 3)), pixel_scale=1.0, renormalize=False
            )

            assert psf.shape == (3, 3)
            assert psf.pixel_scale == 1.0
            assert (psf == np.ones((3, 3))).all()
            assert psf.origin == (0.0, 0.0)

            psf = al.PrimaryBeam(
                array=np.ones((4, 3)), pixel_scale=1.0, renormalize=False
            )

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.shape == (4, 3)
            assert psf.origin == (0.0, 0.0)

        def test__from_fits__input_primary_beam_3x3__all_attributes_correct_including_data_inheritance(
            self
        ):
            psf = al.PrimaryBeam.from_fits_with_scale(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scale=1.0
            )

            assert (psf == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

            psf = al.PrimaryBeam.from_fits_with_scale(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scale=1.0
            )

            assert (psf == np.ones((4, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.origin == (0.0, 0.0)

    class TestRenormalize(object):
        def test__input_is_already_normalized__no_change(self):
            primary_beam_data = np.ones((3, 3)) / 9.0

            psf = al.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=True
            )

            assert psf == pytest.approx(primary_beam_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            primary_beam_data = np.ones((3, 3))

            psf = al.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=True
            )

            assert psf == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__same_as_above__renomalized_false_does_not_renormalize(self):
            primary_beam_data = np.ones((3, 3))

            psf = al.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=False
            )

            assert psf == pytest.approx(np.ones((3, 3)), 1e-3)

    class TestBinnedUp(object):
        def test__primary_beam_is_even_x_even__rescaled_to_odd_x_odd__no_use_of_dimension_trimming(
            self
        ):

            array_2d = np.ones((6, 6))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((9, 9))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.333333333333333, renormalize=True
            )
            assert psf.pixel_scale == 3.0
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((18, 6))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 27.0) * np.ones((9, 3))

            array_2d = np.ones((6, 18))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == 2.0
            assert psf == (1.0 / 27.0) * np.ones((3, 9))

        def test__primary_beam_is_even_x_even_after_binning_up__resized_to_odd_x_odd_with_shape_plus_one(
            self
        ):

            array_2d = np.ones((2, 2))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == 0.4
            assert psf == (1.0 / 25.0) * np.ones((5, 5))

            array_2d = np.ones((40, 40))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.1, renormalize=True
            )
            assert psf.pixel_scale == 8.0
            assert psf == (1.0 / 25.0) * np.ones((5, 5))

            array_2d = np.ones((2, 4))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0 / 45.0) * np.ones((5, 9))

            array_2d = np.ones((4, 2))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=2.0, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(0.4444444, 1.0e-4)
            assert psf == (1.0 / 45.0) * np.ones((9, 5))

        def test__primary_beam_is_odd_and_even_after_binning_up__resized_to_odd_and_odd_with_shape_plus_one(
            self
        ):

            array_2d = np.ones((6, 4))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((9, 12))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.33333333333, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((3, 5))

            array_2d = np.ones((4, 6))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.5, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(2.0, 1.0e-4)
            assert psf == (1.0 / 9.0) * np.ones((3, 3))

            array_2d = np.ones((12, 9))
            psf = al.PrimaryBeam(array=array_2d, pixel_scale=1.0, renormalize=False)
            psf = psf.new_primary_beam_with_rescaled_odd_dimensioned_array(
                rescale_factor=0.33333333333, renormalize=True
            )
            assert psf.pixel_scale == pytest.approx(3.0, 1.0e-4)
            assert psf == (1.0 / 15.0) * np.ones((5, 3))

    class TestNewRenormalizedPrimaryBeam(object):
        def test__input_is_already_normalized__no_change(self):

            primary_beam_data = np.ones((3, 3)) / 9.0

            psf = al.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=False
            )

            primary_beam_new = psf.new_primary_beam_with_renormalized_array()

            assert primary_beam_new == pytest.approx(primary_beam_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            primary_beam_data = np.ones((3, 3))

            psf = al.PrimaryBeam(
                array=primary_beam_data, pixel_scale=1.0, renormalize=False
            )

            primary_beam_new = psf.new_primary_beam_with_renormalized_array()

            assert primary_beam_new == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    class TestConvolve(object):
        def test__kernel_is_not_odd_x_odd__raises_error(self):
            kernel = np.array([[0.0, 1.0], [1.0, 2.0]])

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

            with pytest.raises(exc.ConvolutionException):
                psf.convolve(np.ones((5, 5)))

        def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
            self
        ):
            image = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            psf = al.PrimaryBeam(array=kernel, pixel_scale=1.0)

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

            grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
                shape=(3, 3), pixel_scale=1.0, sub_grid_size=1
            )

            gaussian = al.light_profiles.EllipticalGaussian(
                centre=(0.1, 0.1), axis_ratio=0.9, phi=45.0, intensity=1.0, sigma=1.0
            )
            profile_gaussian = gaussian.profile_image_from_grid(
                grid=grid, return_in_2d=True, return_binned=True
            )

            profile_psf = al.PrimaryBeam(
                array=profile_gaussian, pixel_scale=1.0, renormalize=True
            )

            imaging_psf = al.PrimaryBeam.from_gaussian(
                shape=(3, 3),
                pixel_scale=1.0,
                centre=(0.1, 0.1),
                axis_ratio=0.9,
                phi=45.0,
                sigma=1.0,
            )

            assert profile_psf == pytest.approx(imaging_psf, 1e-4)


class TestSimulateInterferometerData(object):
    def test__setup_with_all_features_off(self, transformer_7x7_7):
        image = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = al.ScaledSquarePixelArray.single_value(
            value=1.0, pixel_scale=0.1, shape=image.shape
        )

        interferometer_data_simulated = al.SimulatedInterferometerData.from_image_and_exposure_arrays(
            image=image,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            pixel_scale=0.1,
            transformer=transformer_7x7_7,
            noise_sigma=None,
        )

        image_1d = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0])
        simulated_visibilities = transformer_7x7_7.visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert interferometer_data_simulated.visibilities == pytest.approx(
            simulated_visibilities, 1.0e-4
        )
        assert interferometer_data_simulated.pixel_scale == 0.1
        assert (
            interferometer_data_simulated.image
            == np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
        ).all()

    def test__setup_with_background_sky_on__noise_off__no_noise_in_image__noise_map_is_noise_value(
        self, transformer_7x7_7
    ):
        image = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = al.ScaledSquarePixelArray.single_value(
            value=1.0, pixel_scale=0.1, shape=image.shape
        )

        background_sky_map = al.ScaledSquarePixelArray.single_value(
            value=2.0, pixel_scale=0.1, shape=image.shape
        )

        interferometer_data_simulated = al.SimulatedInterferometerData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            transformer=transformer_7x7_7,
            noise_sigma=None,
            noise_if_add_noise_false=0.2,
            noise_seed=1,
        )

        image_1d = np.array([4.0, 2.0, 2.0, 2.0, 3.0, 2.0, 5.0, 2.0, 2.0])
        simulated_visibilities = transformer_7x7_7.visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert interferometer_data_simulated.visibilities == pytest.approx(
            simulated_visibilities, 1.0e-4
        )
        assert (
            interferometer_data_simulated.exposure_time_map == 1.0 * np.ones((3, 3))
        ).all()
        assert interferometer_data_simulated.pixel_scale == 0.1

        assert (
            interferometer_data_simulated.image
            == np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
        ).all()
        assert interferometer_data_simulated.visibilities_noise_map == 0.2 * np.ones(
            (6, 2)
        )

    def test__setup_with_noise(self, transformer_7x7_7):
        image = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = al.ScaledSquarePixelArray.single_value(
            value=20.0, pixel_scale=0.1, shape=image.shape
        )

        interferometer_data_simulated = al.SimulatedInterferometerData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=20.0,
            exposure_time_map=exposure_time_map,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        image_1d = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0])
        simulated_visibilities = transformer_7x7_7.visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (
            interferometer_data_simulated.exposure_time_map == 20.0 * np.ones((3, 3))
        ).all()
        assert interferometer_data_simulated.pixel_scale == 0.1

        assert interferometer_data_simulated.image == pytest.approx(
            np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]]), 1e-2
        )

        assert interferometer_data_simulated.visibilities[0, :] == pytest.approx(
            [1.728611, -2.582958], 1.0e-4
        )
        visibilities_noise_map_realization = (
            interferometer_data_simulated.visibilities - simulated_visibilities
        )

        assert visibilities_noise_map_realization == pytest.approx(
            interferometer_data_simulated.visibilities_noise_map_realization, 1.0e-4
        )

        assert interferometer_data_simulated.visibilities_noise_map == 0.1 * np.ones(
            (6, 2)
        )

    def test__from_deflections_and_galaxies__same_as_manual_calculation_using_tracer(
        self, transformer_7x7_7
    ):

        grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(10, 10), pixel_scale=1.0, sub_grid_size=1
        )

        g0 = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0),
        )

        g1 = al.Galaxy(
            redshift=1.0, light=al.light_profiles.SphericalSersic(intensity=1.0)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        deflections = tracer.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        interferometer_data_simulated_via_deflections = al.SimulatedInterferometerData.from_deflections_galaxies_and_exposure_arrays(
            deflections=deflections,
            pixel_scale=1.0,
            galaxies=[g1],
            exposure_time=10000.0,
            background_sky_level=100.0,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        tracer_profile_image_plane_image = tracer.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        interferometer_data_simulated = al.SimulatedInterferometerData.from_image_and_exposure_arrays(
            image=tracer_profile_image_plane_image,
            pixel_scale=1.0,
            exposure_time=10000.0,
            background_sky_level=100.0,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        assert (
            interferometer_data_simulated_via_deflections.image
            == interferometer_data_simulated.image
        ).all()

        assert (
            interferometer_data_simulated_via_deflections.exposure_time_map
            == interferometer_data_simulated.exposure_time_map
        ).all()
        assert (
            interferometer_data_simulated_via_deflections.visibilities
            == interferometer_data_simulated.visibilities
        ).all()

        assert (
            interferometer_data_simulated_via_deflections.visibilities_noise_map
            == interferometer_data_simulated.visibilities_noise_map
        ).all()

    def test__from_tracer__same_as_manual_tracer_input(self, transformer_7x7_7):

        grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(20, 20), pixel_scale=0.05, sub_grid_size=1
        )

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.light_profiles.EllipticalSersic(intensity=1.0),
            mass=al.mass_profiles.EllipticalIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.Galaxy(
            redshift=1.0, light=al.light_profiles.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        interferometer_data_simulated_via_tracer = al.SimulatedInterferometerData.from_tracer_grid_and_exposure_arrays(
            tracer=tracer,
            grid=grid,
            pixel_scale=0.1,
            exposure_time=10000.0,
            background_sky_level=100.0,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer_data_simulated = al.SimulatedInterferometerData.from_image_and_exposure_arrays(
            image=tracer.profile_image_from_grid(
                grid=grid, return_in_2d=True, return_binned=True
            ),
            pixel_scale=0.1,
            exposure_time=10000.0,
            background_sky_level=100.0,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        assert (
            interferometer_data_simulated_via_tracer.image
            == interferometer_data_simulated.image
        ).all()

        assert (
            interferometer_data_simulated_via_tracer.exposure_time_map
            == interferometer_data_simulated.exposure_time_map
        ).all()
        assert (
            interferometer_data_simulated_via_tracer.visibilities
            == interferometer_data_simulated.visibilities
        ).all()

        assert (
            interferometer_data_simulated_via_tracer.visibilities_noise_map
            == interferometer_data_simulated.visibilities_noise_map
        ).all()

    class TestCreateGaussianNoiseMap(object):
        def test__gaussian_noise_sigma_0__gaussian_noise_map_all_0__image_is_identical_to_input(
            self
        ):
            simulate_gaussian_noise = al.gaussian_noise_map_from_shape_and_sigma(
                shape=(9,), sigma=0.0, noise_seed=1
            )

            assert (simulate_gaussian_noise == np.zeros((9,))).all()

        def test__gaussian_noise_sigma_1__gaussian_noise_map_all_non_0__image_has_noise_added(
            self
        ):
            simulate_gaussian_noise = al.gaussian_noise_map_from_shape_and_sigma(
                shape=(9,), sigma=1.0, noise_seed=1
            )

            # Use seed to give us a known gaussian noises map we'll test for

            assert simulate_gaussian_noise == pytest.approx(
                np.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32]),
                1e-2,
            )


class TestInterferometerFromFits(object):
    def test__no_settings_just_pass_fits(self):

        interferometer_data = al.load_interferometer_data_from_fits(
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
        assert (interferometer_data.visibilities[:, 0] == np.ones(3)).all()
        assert (interferometer_data.visibilities[:, 1] == 2.0 * np.ones(3)).all()
        assert (interferometer_data.visibilities_noise_map == 3.0 * np.ones(3)).all()
        assert (interferometer_data.uv_wavelengths[:, 0] == 4.0 * np.ones(3)).all()
        assert (interferometer_data.uv_wavelengths[:, 1] == 5.0 * np.ones(3)).all()
        assert (interferometer_data.primary_beam == 5.0 * np.ones((3, 3))).all()

        assert interferometer_data.pixel_scale == 0.1

    def test__optional_array_paths_included__loads_optional_array(self):

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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
        assert (interferometer_data.visibilities[:, 0] == np.ones(3)).all()
        assert (interferometer_data.visibilities[:, 1] == 2.0 * np.ones(3)).all()
        assert (interferometer_data.visibilities_noise_map == 3.0 * np.ones(3)).all()
        assert (interferometer_data.uv_wavelengths[:, 0] == 4.0 * np.ones(3)).all()
        assert (interferometer_data.uv_wavelengths[:, 1] == 5.0 * np.ones(3)).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(
        self
    ):

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        noise_map_converted = al.NoiseMap.from_weight_map(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        noise_map_converted = al.NoiseMap.from_inverse_noise_map(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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
            al.load_interferometer_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
            )

    def test__multiple_noise_map_options__raises_imaging_exception(self):

        with pytest.raises(exc.DataException):
            al.load_interferometer_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
                noise_map_path=test_data_dir + "3x3_threes.fits",
                convert_noise_map_from_inverse_noise_map=True,
                convert_noise_map_from_weight_map=True,
            )

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):

        with pytest.raises(exc.DataException):
            al.load_interferometer_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                exposure_time_map_path=test_data_dir + "3x3_ones.fits",
                exposure_time_map_from_single_value=1.0,
            )

    def test__output_all_arrays(self):

        interferometer_data = al.load_interferometer_data_from_fits(
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

        al.output_interferometer_data_to_fits(
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

        interferometer_data = al.load_interferometer_data_from_fits(
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
        assert (interferometer_data.visibilities[:, 0] == np.ones(3)).all()
        assert (interferometer_data.visibilities[:, 1] == 2.0 * np.ones(3)).all()
        assert (interferometer_data.visibilities_noise_map == 3.0 * np.ones(3)).all()
        assert (interferometer_data.uv_wavelengths[:, 0] == 4.0 * np.ones(3)).all()
        assert (interferometer_data.uv_wavelengths[:, 1] == 5.0 * np.ones(3)).all()

        assert interferometer_data.pixel_scale == 0.1
        assert interferometer_data.psf.pixel_scale == 0.1
        assert interferometer_data.noise_map.pixel_scale == 0.1
        assert interferometer_data.exposure_time_map.pixel_scale == 0.1
