import os

import numpy as np
import pytest
import shutil
from astropy import units
from astropy.coordinates import Angle
from astropy.modeling import functional_models

import autoarray as aa
import autolens as al
from autolens import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestImagingData:
    class TestConstructor:
        def test__setup_image__correct_attributes(self):

            image = aa.Array.from_array_2d(
                array_2d=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            )
            psf = al.PSF.from_2d_and_pixel_scale(array_2d=3.0 * np.ones((3, 3)), pixel_scale=1.0)
            noise_map = aa.Array.from_array_2d(
                array_2d=5.0 * np.ones((3, 3)),
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=0.1,
                noise_map=noise_map,
                psf=psf,
                background_noise_map=aa.Array.from_array_2d(
                    array_2d=7.0 * np.ones((3, 3)),
                ),
                poisson_noise_map=aa.Array.from_array_2d(
                    array_2d=9.0 * np.ones((3, 3)),
                ),
                exposure_time_map=aa.Array.from_array_2d(
                    array_2d=11.0 * np.ones((3, 3)),
                ),
            )

            assert imaging_data.image.in_2d == pytest.approx(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), 1e-2
            )
            assert (imaging_data.psf.in_2d == 3.0 * np.ones((3, 3))).all()
            assert (imaging_data.noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
            assert (
                imaging_data.background_noise_map.in_2d == 7.0 * np.ones((3, 3))
            ).all()
            assert (imaging_data.poisson_noise_map.in_2d == 9.0 * np.ones((3, 3))).all()
            assert (
                imaging_data.exposure_time_map.in_2d == 11.0 * np.ones((3, 3))
            ).all()

    class TestEstimateNoiseFromImage:
        def test__image_and_exposure_time_all_1s__no_background__noise_is_all_1s(self):
            
            # Imaging (eps) = 1.0
            # Background (eps) = 0.0
            # Exposure times = 1.0 s
            # Imaging (counts) = 1.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(1.0 + 0.0**2) = 1.0
            # Noise (eps) = 1.0 / 1.0

            image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))
            exposure_time = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))
            background_noise = aa.Array.from_array_2d(
                array_2d=np.zeros((3, 3))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert (imaging_data.estimated_noise_map.in_2d == np.ones((3, 3))).all()

        def test__image_all_4s__exposure_time_all_1s__no_background__noise_is_all_2s(
            self
        ):
            # Imaging (eps) = 4.0
            # Background (eps) = 0.0
            # Exposure times = 1.0 s
            # Imaging (counts) = 4.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
            # Noise (eps) = 2.0 / 1.0

            image = aa.Array.from_array_2d(array_2d=4.0 * np.ones((4, 2)))

            exposure_time = aa.Array.from_array_2d(array_2d=np.ones((4, 2)))
            background_noise = aa.Array.from_array_2d(
                array_2d=np.zeros((4, 2))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert (
                imaging_data.estimated_noise_map.in_2d == 2.0 * np.ones((4, 2))
            ).all()

        def test__image_all_1s__exposure_time_all_4s__no_background__noise_is_all_2_divided_4_so_halves(
            self
        ):
            # Imaging (eps) = 1.0
            # Background (eps) = 0.0
            # Exposure times = 4.0 s
            # Imaging (counts) = 4.0
            # Background (counts) = 0.0

            # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
            # Noise (eps) = 2.0 / 4.0 = 0.5

            image = aa.Array.from_array_2d(array_2d=np.ones((1, 5)))

            exposure_time = aa.Array.from_array_2d(
                array_2d=4.0 * np.ones((1, 5))
            )

            background_noise = aa.Array.from_array_2d(
                array_2d=0 * np.ones((1, 5))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert (
                imaging_data.estimated_noise_map.in_2d == 0.5 * np.ones((1, 5))
            ).all()

        def test__image_and_exposure_times_range_of_values__no_background__noises_estimates_correct(
            self
        ):
            image = aa.Array.from_array_2d(
                array_2d=np.array([[5.0, 3.0], [10.0, 20.0]])
            )

            exposure_time = aa.Array.from_array_2d(
                np.array([[1.0, 2.0], [3.0, 4.0]])
            )

            background_noise = aa.Array.from_array_2d(
                array_2d=0.0 * np.ones((2, 2))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert (
                imaging_data.estimated_noise_map.in_2d
                == np.array(
                    [
                        [np.sqrt(5.0), np.sqrt(6.0) / 2.0],
                        [np.sqrt(30.0) / 3.0, np.sqrt(80.0) / 4.0],
                    ]
                )
            ).all()

        def test__image_and_exposure_times_all_1s__background_is_float_sqrt_3__noise_is_all_2s(
            self
        ):
            # Imaging (eps) = 1.0
            # Background (eps) = sqrt(3.0)
            # Exposure times = 1.0 s
            # Imaging (counts) = 1.0
            # Background (counts) = sqrt(3.0)

            # Noise (counts) = sqrt(1.0 + sqrt(3.0)**2) = sqrt(1.0 + 3.0) = 2.0
            # Noise (eps) = 2.0 / 1.0 = 2.0

            image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))

            exposure_time = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))

            background_noise = aa.Array.from_array_2d(
                array_2d=3.0 ** 0.5 * np.ones((3, 3))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert imaging_data.estimated_noise_map.in_2d == pytest.approx(
                2.0 * np.ones((3, 3)), 1e-2
            )

        def test__image_and_exposure_times_all_1s__background_is_float_5__noise_all_correct(
            self
        ):
            # Imaging (eps) = 1.0
            # Background (eps) = 5.0
            # Exposure times = 1.0 s
            # Imaging (counts) = 1.0
            # Background (counts) = 5.0

            # Noise (counts) = sqrt(1.0 + 5**2)
            # Noise (eps) = sqrt(1.0 + 5**2) / 1.0

            image = aa.Array.from_array_2d(array_2d=np.ones((2, 3)))

            exposure_time = aa.Array.from_array_2d(array_2d=np.ones((2, 3)))

            background_noise = aa.Array.from_array_2d(
                array_2d=5 * np.ones((2, 3))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert imaging_data.estimated_noise_map.in_2d == pytest.approx(
                np.array(
                    [
                        [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
                        [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
                    ]
                ),
                1e-2,
            )

        def test__image_all_1s__exposure_times_all_2s__background_is_float_5__noise_all_correct(
            self
        ):
            # Imaging (eps) = 1.0
            # Background (eps) = 5.0
            # Exposure times = 2.0 s
            # Imaging (counts) = 2.0
            # Background (counts) = 10.0

            # Noise (counts) = sqrt(2.0 + 10**2) = sqrt(2.0 + 100.0)
            # Noise (eps) = sqrt(2.0 + 100.0) / 2.0

            image = aa.Array.from_array_2d(array_2d=np.ones((2, 3)))

            exposure_time = aa.Array.from_array_2d(
                array_2d=2.0 * np.ones((2, 3))
            )
            background_noise = aa.Array.from_array_2d(
                array_2d=5.0 * np.ones((2, 3))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert imaging_data.estimated_noise_map.in_2d == pytest.approx(
                np.array(
                    [
                        [
                            np.sqrt(2.0 + 100.0) / 2.0,
                            np.sqrt(2.0 + 100.0) / 2.0,
                            np.sqrt(2.0 + 100.0) / 2.0,
                        ],
                        [
                            np.sqrt(2.0 + 100.0) / 2.0,
                            np.sqrt(2.0 + 100.0) / 2.0,
                            np.sqrt(2.0 + 100.0) / 2.0,
                        ],
                    ]
                ),
                1e-2,
            )

        def test__same_as_above_but_different_image_values_in_each_pixel_and_new_background_values(
            self
        ):
            # Can use pattern from previous test_autoarray for values

            image = aa.Array.from_array_2d(
                array_2d=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            )

            exposure_time = aa.Array.from_array_2d(array_2d=np.ones((3, 2)))
            background_noise = aa.Array.from_array_2d(
                array_2d=12.0 * np.ones((3, 2))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert imaging_data.estimated_noise_map.in_2d == pytest.approx(
                np.array(
                    [
                        [np.sqrt(1.0 + 144.0), np.sqrt(2.0 + 144.0)],
                        [np.sqrt(3.0 + 144.0), np.sqrt(4.0 + 144.0)],
                        [np.sqrt(5.0 + 144.0), np.sqrt(6.0 + 144.0)],
                    ]
                ),
                1e-2,
            )

        def test__image_and_exposure_times_range_of_values__background_has_value_9___noise_estimates_correct(
            self
        ):
            # Use same pattern as above, noting that here our background values are now being converts to counts using
            # different exposure time and then being squared.

            image = aa.Array.from_array_2d(
                array_2d=[[5.0, 3.0], [10.0, 20.0]]
            )

            exposure_time = aa.Array.from_array_2d(
                array_2d=[[1.0, 2.0], [3.0, 4.0]]
            )
            background_noise = aa.Array.from_array_2d(
                array_2d=9.0 * np.ones((2, 2))
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert imaging_data.estimated_noise_map.in_2d == pytest.approx(
                np.array(
                    [
                        [np.sqrt(5.0 + 81.0), np.sqrt(6.0 + 18.0 ** 2.0) / 2.0],
                        [
                            np.sqrt(30.0 + 27.0 ** 2.0) / 3.0,
                            np.sqrt(80.0 + 36.0 ** 2.0) / 4.0,
                        ],
                    ]
                ),
                1e-2,
            )

        def test__image_and_exposure_times_and_background_are_all_ranges_of_values__noise_estimates_correct(
            self
        ):
            # Use same pattern as above, noting that we are now also using a variable background signal_to_noise_ratio map.

            image = aa.Array.from_array_2d(
                array_2d=[[5.0, 3.0], [10.0, 20.0]]
            )

            exposure_time = aa.Array.from_array_2d(
                array_2d=[[1.0, 2.0], [3.0, 4.0]]
            )

            background_noise = aa.Array.from_array_2d(
                array_2d=[[5.0, 6.0], [7.0, 8.0]]
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 3)), pixel_scale=1.0),
                exposure_time_map=exposure_time,
                background_noise_map=background_noise,
            )

            assert imaging_data.estimated_noise_map.in_2d == pytest.approx(
                np.array(
                    [
                        [np.sqrt(5.0 + 5.0 ** 2.0), np.sqrt(6.0 + 12.0 ** 2.0) / 2.0],
                        [
                            np.sqrt(30.0 + 21.0 ** 2.0) / 3.0,
                            np.sqrt(80.0 + 32.0 ** 2.0) / 4.0,
                        ],
                    ]
                ),
                1e-2,
            )

    class TestEstimateDataGrid(object):
        def test__via_edges__input_all_ones__sky_bg_level_1(self):
            imaging_data = al.ImagingData(
                image=aa.Array.from_array_2d(np.ones((3, 3))),
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )

            sky_noise = imaging_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(
            self
        ):
            image = aa.Array.from_array_2d([[1, 1, 1], [1, 100, 1], [1, 1, 1]])

            imaging_data = al.ImagingData(
                image=image,
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )
            sky_noise = imaging_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):
            image = aa.Array.from_array_2d([[1, 1, 1], [1, 100, 1], [1, 100, 1], [1, 1, 1]])

            imaging_data = al.ImagingData(
                image=image,
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )
            sky_noise = imaging_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image = aa.Array.from_array_2d(
                [[1, 1, 1, 1], [1, 100, 100, 1], [1, 100, 100, 1], [1, 1, 1, 1]]
            )

            imaging_data = al.ImagingData(
                image=image,
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )
            sky_noise = imaging_data.background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(
            self
        ):
            image = aa.Array.from_array_2d(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 100, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
            )

            imaging_data = al.ImagingData(
                image=image,
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )
            sky_noise = imaging_data.background_noise_from_edges(no_edges=2)

            assert sky_noise == 0.0

        def test__via_edges__6x5_image_two_edges__values(self):
            image = aa.Array.from_array_2d(
                [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 100, 12, 13],
                    [14, 15, 100, 16, 17],
                    [18, 19, 20, 21, 22],
                    [23, 24, 25, 26, 27],
                ]
            )

            imaging_data = al.ImagingData(
                image=image,
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )
            sky_noise = imaging_data.background_noise_from_edges(no_edges=2)

            assert sky_noise == np.std(np.arange(28))

        def test__via_edges__7x7_image_three_edges__values(self):
            image = aa.Array.from_array_2d(
                [
                    [0, 1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12, 13],
                    [14, 15, 16, 17, 18, 19, 20],
                    [21, 22, 23, 100, 24, 25, 26],
                    [27, 28, 29, 30, 31, 32, 33],
                    [34, 35, 36, 37, 38, 39, 40],
                    [41, 42, 43, 44, 45, 46, 47],
                ]
            )

            imaging_data = al.ImagingData(
                image=image,
                noise_map=np.ones((3, 3)),
                psf=np.ones((3, 3)),
                pixel_scale=0.1,
            )
            sky_noise = imaging_data.background_noise_from_edges(no_edges=3)

            assert sky_noise == np.std(np.arange(48))

    class TestNewImagingDataResized:
        def test__all_components_resized__psf_is_not(self):
            image = aa.Array.from_array_2d(array_2d=np.ones((6, 6)))
            image[3, 3] = 2.0

            noise_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            noise_map_array[3, 3] = 3.0

            background_noise_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            background_noise_map_array[3, 3] = 4.0

            exposure_time_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            exposure_time_map_array[3, 3] = 5.0

            background_sky_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            background_sky_map_array[3, 3] = 6.0

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3))),
                noise_map=noise_map_array,
                background_noise_map=background_noise_map_array,
                exposure_time_map=exposure_time_map_array,
                background_sky_map=background_sky_map_array,
            )

            imaging_data = imaging_data.resized_data_from_new_shape(
                new_shape=(4, 4)
            )

            assert (
                imaging_data.image
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
                imaging_data.noise_map
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
                imaging_data.background_noise_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 4.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()
            assert (
                imaging_data.exposure_time_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 5.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()
            assert (
                imaging_data.background_sky_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 6.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()

            assert imaging_data.poisson_noise_map == None

            assert imaging_data.pixel_scale == 1.0
            assert (imaging_data.psf.in_2d == np.zeros((3, 3))).all()
            assert imaging_data.geometry.origin == (0.0, 0.0)

        def test__resize_psf(self):
            image = aa.Array.from_array_2d(array_2d=np.ones((6, 6)))

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3))),
            )

            imaging_data = imaging_data.resized_psf_data_from_new_shape(
                new_shape=(1, 1)
            )

            assert (imaging_data.image.in_2d == np.ones((6, 6))).all()
            assert imaging_data.pixel_scale == 1.0
            assert (imaging_data.psf.in_2d == np.zeros((1, 1))).all()

    class TestNewImagingModifiedImage:
        def test__imaging_data_returns_with_modified_image(self):
            image = aa.Array.from_array_2d(array_2d=np.ones((4, 4)))
            image[2, 2] = 2.0

            noise_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((4, 4))
            )
            noise_map_array[2, 2] = 3.0

            background_noise_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((4, 4))
            )
            background_noise_map_array[2, 2] = 4.0

            exposure_time_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((4, 4))
            )
            exposure_time_map_array[2, 2] = 5.0

            background_sky_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((4, 4))
            )
            background_sky_map_array[2, 2] = 6.0

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3))),
                noise_map=noise_map_array,
                background_noise_map=background_noise_map_array,
                exposure_time_map=exposure_time_map_array,
                background_sky_map=background_sky_map_array,
            )

            modified_image = aa.Array.from_array_2d(
                array_2d=np.ones((4, 4))
            )
            modified_image[2, 2] = 10.0

            imaging_data = imaging_data.modified_image_data_from_image(
                image=modified_image
            )

            assert (
                imaging_data.image
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 10.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()
            assert (
                imaging_data.noise_map
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
                imaging_data.background_noise_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 4.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()
            assert (
                imaging_data.exposure_time_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 5.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()
            assert (
                imaging_data.background_sky_map
                == np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 6.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ).all()

            assert imaging_data.poisson_noise_map == None

            assert imaging_data.pixel_scale == 1.0
            assert (imaging_data.psf.in_2d == np.zeros((3, 3))).all()
            assert imaging_data.geometry.origin == (0.0, 0.0)

    class TestNewImagingBinnedUp:
        def test__all_components_binned_up_correct(self):
            image = aa.Array.from_array_2d(array_2d=np.ones((6, 6)))
            image[3:5, 3] = 2.0
            binned_image_util = aa.binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                array_2d=image, bin_up_factor=2
            )

            noise_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            noise_map_array[3, 3:5] = 3.0
            binned_noise_map_util = aa.binning_util.binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(
                array_2d=noise_map_array, bin_up_factor=2
            )

            background_noise_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            background_noise_map_array[3:5, 3] = 4.0
            binned_background_noise_map_util = aa.binning_util.binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(
                array_2d=background_noise_map_array, bin_up_factor=2
            )

            exposure_time_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            exposure_time_map_array[3, 3:5] = 5.0
            binned_exposure_time_map_util = aa.binning_util.binned_array_2d_using_sum_from_array_2d_and_bin_up_factor(
                array_2d=exposure_time_map_array, bin_up_factor=2
            )

            background_sky_map_array = aa.Array.from_array_2d(
                array_2d=np.ones((6, 6))
            )
            background_sky_map_array[3, 3:5] = 6.0
            binned_background_sky_map_util = aa.binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                array_2d=background_sky_map_array, bin_up_factor=2
            )

            psf = al.PSF.from_2d_and_pixel_scale(array_2d=np.ones((3, 5)))
            psf_util = psf.rescaled_with_odd_dimensions_from_rescale_factor(
                rescale_factor=0.5
            )

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=psf,
                noise_map=noise_map_array,
                background_noise_map=background_noise_map_array,
                exposure_time_map=exposure_time_map_array,
                background_sky_map=background_sky_map_array,
            )

            imaging_data = imaging_data.binned_data_from_bin_up_factor(
                bin_up_factor=2
            )

            assert (imaging_data.image.in_2d == binned_image_util).all()
            assert (imaging_data.psf.in_2d == psf_util).all()
            assert (imaging_data.noise_map.in_2d == binned_noise_map_util).all()
            assert (
                imaging_data.background_noise_map.in_2d
                == binned_background_noise_map_util
            ).all()
            assert (
                imaging_data.exposure_time_map.in_2d == binned_exposure_time_map_util
            ).all()
            assert (
                imaging_data.background_sky_map.in_2d == binned_background_sky_map_util
            ).all()
            assert imaging_data.poisson_noise_map == None

            assert imaging_data.pixel_scale == 2.0
            assert imaging_data.image.pixel_scale == 2.0
            assert imaging_data.psf.mask.geometry.pixel_scale == pytest.approx(
                1.66666666666, 1.0e-4
            )
            assert imaging_data.noise_map.mask.geometry.pixel_scale == 2.0
            assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 2.0
            assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 2.0
            assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 2.0

            assert imaging_data.geometry.origin == (0.0, 0.0)

    class TestSNRLimit:
        def test__signal_to_noise_limit_above_max_signal_to_noise__signal_to_noise_map_unchanged(
            self
        ):
            image = aa.Array.from_array_2d(20.0 * np.ones((2, 2)))
            image[1, 1] = 5.0

            noise_map_array = aa.Array.from_array_2d(5.0 * np.ones((2, 2)))
            noise_map_array[1, 1] = 2.0

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3))),
                noise_map=noise_map_array,
                background_noise_map=1.0 * np.ones((2, 2)),
                exposure_time_map=2.0 * np.ones((2, 2)),
                background_sky_map=3.0 * np.ones((2, 2)),
            )

            imaging_data = imaging_data.signal_to_noise_limited_data_from_signal_to_noise_limit(
                signal_to_noise_limit=100.0
            )

            assert (imaging_data.image.in_2d == np.array([[20.0, 20.0], [20.0, 5.0]])).all()

            assert (
                imaging_data.noise_map.in_2d == np.array([[5.0, 5.0], [5.0, 2.0]])
            ).all()

            assert (
                imaging_data.signal_to_noise_map.in_2d
                == np.array([[4.0, 4.0], [4.0, 2.5]])
            ).all()

            assert imaging_data.pixel_scale == 1.0
            assert (imaging_data.psf.in_2d == np.zeros((3, 3))).all()
            assert (imaging_data.background_noise_map.in_2d == np.ones((2, 2))).all()
            assert (imaging_data.exposure_time_map.in_2d == 2.0 * np.ones((2, 2))).all()
            assert (imaging_data.background_sky_map.in_2d == 3.0 * np.ones((2, 2))).all()

        def test__signal_to_noise_limit_below_max_signal_to_noise__signal_to_noise_map_capped_to_limit(
            self
        ):
            image = aa.Array.from_array_2d(20.0 * np.ones((2, 2)))
            image[1, 1] = 5.0

            noise_map_array = aa.Array.from_array_2d(5.0 * np.ones((2, 2)))
            noise_map_array[1, 1] = 2.0

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3))),
                noise_map=noise_map_array,
                background_noise_map=1.0 * np.ones((2, 2)),
                exposure_time_map=2.0 * np.ones((2, 2)),
                background_sky_map=3.0 * np.ones((2, 2)),
            )

            imaging_data_capped = imaging_data.signal_to_noise_limited_data_from_signal_to_noise_limit(
                signal_to_noise_limit=2.0
            )

            assert (
                imaging_data_capped.image.in_2d == np.array([[20.0, 20.0], [20.0, 5.0]])
            ).all()

            assert (
                imaging_data_capped.noise_map.in_2d
                == np.array([[10.0, 10.0], [10.0, 2.5]])
            ).all()

            assert (
                imaging_data_capped.signal_to_noise_map
                == np.array([[2.0, 2.0], [2.0, 2.0]])
            ).all()

            assert imaging_data_capped.pixel_scale == 1.0
            assert (imaging_data_capped.psf.in_2d == np.zeros((3, 3))).all()
            assert (
                imaging_data_capped.background_noise_map.in_2d == np.ones((2, 2))
            ).all()
            assert (
                imaging_data_capped.exposure_time_map.in_2d == 2.0 * np.ones((2, 2))
            ).all()
            assert (
                imaging_data_capped.background_sky_map.in_2d == 3.0 * np.ones((2, 2))
            ).all()

            imaging_data_capped = imaging_data.signal_to_noise_limited_data_from_signal_to_noise_limit(
                signal_to_noise_limit=3.0
            )

            assert (
                imaging_data_capped.image.in_2d == np.array([[20.0, 20.0], [20.0, 5.0]])
            ).all()

            assert (
                imaging_data_capped.noise_map
                == np.array([[(20.0 / 3.0), (20.0 / 3.0)], [(20.0 / 3.0), 2.0]])
            ).all()

            assert (
                imaging_data_capped.signal_to_noise_map
                == np.array([[3.0, 3.0], [3.0, 2.5]])
            ).all()

            assert imaging_data_capped.pixel_scale == 1.0
            assert (imaging_data_capped.psf.in_2d == np.zeros((3, 3))).all()
            assert (
                imaging_data_capped.background_noise_map.in_2d == np.ones((2, 2))
            ).all()
            assert (
                imaging_data_capped.exposure_time_map.in_2d == 2.0 * np.ones((2, 2))
            ).all()
            assert (
                imaging_data_capped.background_sky_map.in_2d == 3.0 * np.ones((2, 2))
            ).all()

    class TestImageConvertedFrom:
        def test__counts__all_arrays_in_units_of_flux_are_converted(self):
            image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))
            noise_map_array = aa.Array.from_array_2d(2.0 * np.ones((3, 3)))
            background_noise_map_array = aa.Array.from_array_2d(
                3.0 * np.ones((3, 3))
            )
            exposure_time_map_array = aa.Array.from_array_2d(0.5 * np.ones((3, 3)))
            background_sky_map_array = aa.Array.from_array_2d(6.0 * np.ones((3, 3)))

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3)), pixel_scale=1.0),
                noise_map=noise_map_array,
                background_noise_map=background_noise_map_array,
                poisson_noise_map=None,
                exposure_time_map=exposure_time_map_array,
                background_sky_map=background_sky_map_array,
            )

            imaging_data = imaging_data.data_in_electrons()

            assert (imaging_data.image.in_2d == 2.0 * np.ones((3, 3))).all()
            assert (imaging_data.noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
            assert (
                imaging_data.background_noise_map.in_2d == 6.0 * np.ones((3, 3))
            ).all()
            assert imaging_data.poisson_noise_map == None
            assert (imaging_data.background_sky_map.in_2d == 12.0 * np.ones((3, 3))).all()

        def test__adus__all_arrays_in_units_of_flux_are_converted(self):
            image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))
            noise_map_array = aa.Array.from_array_2d(2.0 * np.ones((3, 3)))
            background_noise_map_array = aa.Array.from_array_2d(
                3.0 * np.ones((3, 3))
            )
            exposure_time_map_array = aa.Array.from_array_2d(0.5 * np.ones((3, 3)))
            background_sky_map_array = aa.Array.from_array_2d(6.0 * np.ones((3, 3)))

            imaging_data = al.ImagingData(
                image=image,
                pixel_scale=1.0,
                psf=al.PSF.from_2d_and_pixel_scale(array_2d=np.zeros((3, 3)), pixel_scale=1.0),
                noise_map=noise_map_array,
                background_noise_map=background_noise_map_array,
                poisson_noise_map=None,
                exposure_time_map=exposure_time_map_array,
                background_sky_map=background_sky_map_array,
            )

            imaging_data = imaging_data.data_in_adus_from_gain(gain=2.0)

            assert (imaging_data.image.in_2d == 2.0 * 2.0 * np.ones((3, 3))).all()
            assert (imaging_data.noise_map.in_2d == 2.0 * 4.0 * np.ones((3, 3))).all()
            assert (
                imaging_data.background_noise_map.in_2d == 2.0 * 6.0 * np.ones((3, 3))
            ).all()
            assert imaging_data.poisson_noise_map == None
            assert (
                imaging_data.background_sky_map.in_2d == 2.0 * 12.0 * np.ones((3, 3))
            ).all()

    class TestImageWithPoissonNoiseAdded:
        def test__mock_image_all_1s__poisson_noise_is_added_correct(self):
            psf = al.PSF.from_2d_and_pixel_scale(
                array_2d=np.ones((3, 3)), pixel_scale=3.0, renormalize=False
            )
            imaging_data = al.ImagingData(
                image=aa.Array.from_array_2d(array_2d=np.ones((4, 4))),
                pixel_scale=3.0,
                psf=psf,
                noise_map=aa.Array.from_array_2d(array_2d=np.ones((4, 4))),
                exposure_time_map=aa.Array.from_array_2d(
                    array_2d=3.0 * np.ones((4, 4))
                ),
                background_sky_map=aa.Array.from_array_2d(
                    array_2d=4.0 * np.ones((4, 4))
                ),
            )

            mock_image = aa.Array.from_array_2d(array_2d=np.ones((4, 4)))
            mock_image_with_sky = mock_image + 4.0 * np.ones((16,))
            mock_image_with_sky_and_noise = (
                mock_image_with_sky
                + al.generate_poisson_noise(
                    image=mock_image_with_sky,
                    exposure_time_map=aa.Array.from_array_2d(
                        array_2d=3.0 * np.ones((4, 4))
                    ),
                    seed=1,
                )
            )

            mock_image_with_noise = mock_image_with_sky_and_noise - 4.0 * np.ones(
                (16,)
            )

            imaging_with_noise = imaging_data.add_poisson_noise_to_data(
                seed=1
            )

            assert (imaging_with_noise.image == mock_image_with_noise).all()


class TestNoiseMap(object):
    class TestFromImageAndBackgroundNoiseMap:
        def test__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                gain=1.0,
                exposure_time_map=exposure_time_map,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(2.0), np.sqrt(2.0)]]
                )
            ).all()

        def test__image_all_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_3s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[2.0, 2.0], [2.0, 2.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                gain=1.0,
                exposure_time_map=exposure_time_map,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [[np.sqrt(3.0), np.sqrt(3.0)], [np.sqrt(3.0), np.sqrt(3.0)]]
                )
            ).all()

        def test__image_all_1s__bg_noise_all_2s__exposure_time_all_1s__noise_map_all_sqrt_5s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = aa.Array.from_array_2d([[2.0, 2.0], [2.0, 2.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                gain=1.0,
                exposure_time_map=exposure_time_map,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [[np.sqrt(5.0), np.sqrt(5.0)], [np.sqrt(5.0), np.sqrt(5.0)]]
                )
            ).all()

        def test__image_all_1s__bg_noise_all_1s__exposure_time_all_2s__noise_map_all_sqrt_6s_over_2(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[2.0, 2.0], [2.0, 2.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                gain=1.0,
                exposure_time_map=exposure_time_map,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [
                        [np.sqrt(6.0) / 2.0, np.sqrt(6.0) / 2.0],
                        [np.sqrt(6.0) / 2.0, np.sqrt(6.0) / 2.0],
                    ]
                )
            ).all()

        def test__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_1s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[-2.0, -2.0], [-2.0, -2.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                gain=1.0,
                exposure_time_map=exposure_time_map,
            )

            assert (noise_map.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__same_as_above__use_different_values_in_different_array_elemets(self):
            imaging_data = aa.Array.from_array_2d([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[4.0, 3.0], [2.0, 1.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                gain=1.0,
                exposure_time_map=exposure_time_map,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [
                        [np.sqrt(20.0) / 4.0, np.sqrt(15.0) / 3.0],
                        [np.sqrt(20.0) / 2.0, np.sqrt(12.0)],
                    ]
                )
            ).all()

        def test__convert_from_electrons__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                exposure_time_map=exposure_time_map,
                gain=2.0,
                convert_from_electrons=True,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(2.0), np.sqrt(2.0)]]
                )
            ).all()

        def test__convert_from_electrons__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_10s__noise_map_all_1s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[-2.0, -2.0], [-2.0, -2.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[10.0, 10.0], [10.0, 10.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                exposure_time_map=exposure_time_map,
                gain=1.0,
                convert_from_electrons=True,
            )

            assert (noise_map.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__convert_from_electrons__same_as_above__use_different_values_in_different_array_elemets(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[10.0, 11.0], [12.0, 13.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                exposure_time_map=exposure_time_map,
                gain=4.0,
                convert_from_electrons=True,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [[np.sqrt(2.0), np.sqrt(3.0)], [np.sqrt(6.0), np.sqrt(12.0)]]
                )
            ).all()

        def test__convert_from_adus__same_as_above__gain_is_1__same_values(self):
            imaging_data = aa.Array.from_array_2d([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[10.0, 11.0], [12.0, 13.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                exposure_time_map=exposure_time_map,
                gain=1.0,
                convert_from_adus=True,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [[np.sqrt(2.0), np.sqrt(3.0)], [np.sqrt(6.0), np.sqrt(12.0)]]
                )
            ).all()

        def test__convert_from_adus__same_as_above__gain_is_2__values_change(self):
            imaging_data = aa.Array.from_array_2d([[1.0, 2.0], [2.0, 3.0]])
            background_noise_map = aa.Array.from_array_2d([[1.0, 1.0], [2.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[10.0, 11.0], [12.0, 13.0]])

            noise_map = al.NoiseMap.from_image_and_background_noise_map(
                pixel_scale=0.1,
                image=imaging_data,
                background_noise_map=background_noise_map,
                exposure_time_map=exposure_time_map,
                gain=2.0,
                convert_from_adus=True,
            )

            assert (
                noise_map.in_2d
                == np.array(
                    [
                        [np.sqrt(6.0) / 2.0, np.sqrt(8.0) / 2.0],
                        [np.sqrt(20.0) / 2.0, np.sqrt(42.0) / 2.0],
                    ]
                )
            ).all()


class TestPoissonNoiseMap(object):
    class TestFromImageAndExposureTimeMap:
        def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s(self):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = al.PoissonNoiseMap.from_image_and_exposure_time_map(
                pixel_scale=0.1,
                image=imaging_data,
                exposure_time_map=exposure_time_map,
                gain=1.0,
            )

            assert (poisson_noise_map.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        def test__image_all_2s_and_3s__exposure_time_all_1s__noise_map_all_sqrt_2s_and_3s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = al.PoissonNoiseMap.from_image_and_exposure_time_map(
                pixel_scale=0.1,
                image=imaging_data,
                exposure_time_map=exposure_time_map,
                gain=1.0,
            )

            assert (
                poisson_noise_map.in_2d
                == np.array(
                    [[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]]
                )
            ).all()

        def test__image_all_1s__exposure_time_all__2s_and_3s__noise_map_all_sqrt_2s_and_3s(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[2.0, 2.0], [3.0, 3.0]])

            poisson_noise_map = al.PoissonNoiseMap.from_image_and_exposure_time_map(
                pixel_scale=0.1,
                image=imaging_data,
                exposure_time_map=exposure_time_map,
                gain=1.0,
            )

            assert (
                poisson_noise_map.in_2d
                == np.array(
                    [
                        [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
                        [np.sqrt(3.0) / 3.0, np.sqrt(3.0) / 3.0],
                    ]
                )
            ).all()

        def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s__gain_is_2__ignores_gain(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])
            exposure_time_map = aa.Array.from_array_2d([[1.0, 1.0], [1.0, 1.0]])

            poisson_noise_map = al.PoissonNoiseMap.from_image_and_exposure_time_map(
                pixel_scale=0.1,
                image=imaging_data,
                exposure_time_map=exposure_time_map,
                gain=2.0,
            )

            assert (
                poisson_noise_map.in_2d
                == np.array(
                    [[np.sqrt(1.0), np.sqrt(1.0)], [np.sqrt(1.0), np.sqrt(1.0)]]
                )
            ).all()

        def test__convert_from_electrons_is_true__image_already_in_counts_so_exposure_time_ignored(
            self
        ):
            imaging_data = aa.Array.from_array_2d([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[10.0, 10.0], [10.0, 10.0]])

            poisson_noise_map = al.PoissonNoiseMap.from_image_and_exposure_time_map(
                pixel_scale=0.1,
                image=imaging_data,
                exposure_time_map=exposure_time_map,
                gain=4.0,
                convert_from_electrons=True,
            )

            assert (
                poisson_noise_map.in_2d
                == np.array(
                    [[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]]
                )
            ).all()

        def test__same_as_above__convert_from_adus__includes_gain_multiplication(self):
            imaging_data = aa.Array.from_array_2d([[2.0, 2.0], [3.0, 3.0]])
            exposure_time_map = aa.Array.from_array_2d([[10.0, 10.0], [10.0, 10.0]])

            poisson_noise_map = al.PoissonNoiseMap.from_image_and_exposure_time_map(
                pixel_scale=0.1,
                image=imaging_data,
                exposure_time_map=exposure_time_map,
                gain=2.0,
                convert_from_adus=True,
            )

            assert (
                poisson_noise_map.in_2d
                == np.array(
                    [
                        [np.sqrt(2.0 * 2.0) / 2.0, np.sqrt(2.0 * 2.0) / 2.0],
                        [np.sqrt(2.0 * 3.0) / 2.0, np.sqrt(2.0 * 3.0) / 2.0],
                    ]
                )
            ).all()


class TestPSF(object):

    def test__init__input_psf__all_attributes_correct_including_data_inheritance(
        self
    ):
        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.ones((3, 3)), pixel_scale=1.0, renormalize=False
        )

        assert isinstance(psf, aa.Kernel)
        assert type(psf) == al.PSF
        assert psf.in_2d.shape == (3, 3)
        assert (psf.in_2d == np.ones((3, 3))).all()
        assert psf.mask.geometry.pixel_scales == (1.0, 1.0)
        assert psf.mask.origin == (0.0, 0.0)

        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.ones((4, 3)), pixel_scale=1.0, renormalize=False
        )

        assert psf.in_2d.shape == (4, 3)
        assert (psf.in_2d == np.ones((4, 3))).all()
        assert psf.mask.geometry.pixel_scales == (1.0, 1.0)
        assert psf.mask.origin == (0.0, 0.0)

    def test__from_fits__input_psf_3x3__all_attributes_correct_including_data_inheritance(
        self
    ):
        psf = al.PSF.from_fits_and_pixel_scale(
            file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scale=1.0
        )

        assert (psf.in_2d == np.ones((3, 3))).all()

        psf = al.PSF.from_fits_and_pixel_scale(
            file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scale=1.0
        )

        assert (psf.in_2d == np.ones((4, 3))).all()


class TestSimulateImaging(object):
    def test__setup_with_all_features_off(self):

        image = aa.Array.from_array_2d(
            array_2d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.in_2d.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            pixel_scale=0.1,
            add_noise=False,
        )

        assert (imaging_data_simulated.exposure_time_map.in_2d == np.ones((3, 3))).all()
        assert (
            imaging_data_simulated.image.in_2d
            == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()
        assert imaging_data_simulated.pixel_scale == 0.1

    def test__setup_with_background_sky_on__noise_off__no_noise_in_image__noise_map_is_noise_value(
        self
    ):

        image = aa.Array.from_array_2d(
            array_2d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        background_sky_map = aa.Array.from_single_value_and_shape_2d(
            value=16.0, shape_2d=image.mask.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            add_noise=False,
            noise_if_add_noise_false=0.2,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated.exposure_time_map.in_2d == 1.0 * np.ones((3, 3))
        ).all()
        assert imaging_data_simulated.pixel_scale == 0.1

        assert (
            imaging_data_simulated.image.in_2d
            == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()
        assert (imaging_data_simulated.noise_map.in_2d == 0.2 * np.ones((3, 3))).all()

        assert (
            imaging_data_simulated.background_noise_map.in_2d == 4.0 * np.ones((3, 3))
        ).all()

    def test__setup_with_background_sky_on__noise_on_so_background_adds_noise_to_image(
        self
    ):

        image = aa.Array.from_array_2d(
            array_2d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        background_sky_map = aa.Array.from_single_value_and_shape_2d(
            value=16.0, shape_2d=image.mask.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            add_noise=True,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated.exposure_time_map.in_2d == 1.0 * np.ones((3, 3))
        ).all()
        assert imaging_data_simulated.pixel_scale == 0.1

        assert (
            imaging_data_simulated.image.in_2d
            == np.array([[1.0, 5.0, 4.0], [1.0, 2.0, 1.0], [5.0, 2.0, 7.0]])
        ).all()

        assert (
            imaging_data_simulated.poisson_noise_map.in_2d
            == np.array(
                [
                    [np.sqrt(1.0), np.sqrt(5.0), np.sqrt(4.0)],
                    [np.sqrt(1.0), np.sqrt(2.0), np.sqrt(1.0)],
                    [np.sqrt(5.0), np.sqrt(2.0), np.sqrt(7.0)],
                ]
            )
        ).all()

        assert (
            imaging_data_simulated.background_noise_map.in_2d == 4.0 * np.ones((3, 3))
        ).all()

    def test__setup_with_psf_blurring_on__blurs_image_and_trims_psf_edge_off(self):
        image = aa.Array.from_array_2d(
            array_2d=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scale=1.0,
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            psf=psf,
            add_noise=False,
        )

        assert (
            imaging_data_simulated.image.in_2d
            == np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
        ).all()
        assert (imaging_data_simulated.exposure_time_map.in_2d == np.ones((3, 3))).all()
        assert imaging_data_simulated.pixel_scale == 0.1

    def test__setup_with_background_sky_and_psf_on__psf_does_no_blurring__image_and_sky_both_trimmed(
        self
    ):
        image = aa.Array.from_array_2d(
            array_2d=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            pixel_scale=1.0,
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        background_sky_map = aa.Array.from_single_value_and_shape_2d(
            value=16.0, shape_2d=image.mask.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            psf=psf,
            background_sky_map=background_sky_map,
            add_noise=False,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated.exposure_time_map.in_2d == 1.0 * np.ones((3, 3))
        ).all()
        assert imaging_data_simulated.pixel_scale == 0.1

        assert (
            imaging_data_simulated.image.in_2d
            == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            imaging_data_simulated.background_noise_map.in_2d == 4.0 * np.ones((3, 3))
        ).all()

    def test__setup_with_noise(self):
        image = aa.Array.from_array_2d(
            array_2d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=20.0, shape_2d=image.mask.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=20.0,
            exposure_time_map=exposure_time_map,
            add_noise=True,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated.exposure_time_map.in_2d == 20.0 * np.ones((3, 3))
        ).all()
        assert imaging_data_simulated.pixel_scale == 0.1

        assert imaging_data_simulated.image.in_2d == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 0.0]]), 1e-2
        )

        # Because of the value is 1.05, the estimated Poisson noise_map_1d is:
        # sqrt((1.05 * 20))/20 = 0.2291

        assert imaging_data_simulated.poisson_noise_map.in_2d == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.2291, 0.0], [0.0, 0.0, 0.0]]), 1e-2
        )

        assert imaging_data_simulated.noise_map.in_2d == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.2291, 0.0], [0.0, 0.0, 0.0]]), 1e-2
        )

    def test__setup_with__psf_blurring_and_poisson_noise_on__poisson_noise_added_to_blurred_image(
        self
    ):
        image = aa.Array.from_array_2d(
            array_2d=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scale=1.0,
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=20.0, shape_2d=image.mask.shape
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=20.0,
            exposure_time_map=exposure_time_map,
            psf=psf,
            add_noise=True,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated.exposure_time_map.in_2d == 20.0 * np.ones((3, 3))
        ).all()
        assert imaging_data_simulated.pixel_scale == 0.1
        assert imaging_data_simulated.image.in_2d == pytest.approx(
            np.array([[0.0, 1.05, 0.0], [1.3, 2.35, 1.05], [0.0, 1.05, 0.0]]), 1e-2
        )

        # The estimated Poisson noises are:
        # sqrt((2.35 * 20))/20 = 0.3427
        # sqrt((1.3 * 20))/20 = 0.2549
        # sqrt((1.05 * 20))/20 = 0.2291

        assert imaging_data_simulated.poisson_noise_map.in_2d == pytest.approx(
            np.array(
                [[0.0, 0.2291, 0.0], [0.2549, 0.3427, 0.2291], [0.0, 0.2291, 0.0]]
            ),
            1e-2,
        )

    def test__simulate_function__turns_exposure_time_and_sky_level_to_arrays(self):
        image = aa.Array.from_array_2d(
            array_2d=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            pixel_scale=1.0,
        )

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        background_sky_map = aa.Array.from_single_value_and_shape_2d(
            value=16.0, shape_2d=image.mask.shape
        )

        imaging_variable = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            psf=psf,
            background_sky_map=background_sky_map,
            pixel_scale=0.1,
            add_noise=False,
            noise_seed=1,
        )

        image = aa.Array.from_array_2d(
            array_2d=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=image,
            pixel_scale=0.1,
            exposure_time=1.0,
            background_sky_level=16.0,
            psf=psf,
            add_noise=False,
            noise_seed=1,
        )

        assert (
            imaging_variable.exposure_time_map.in_2d
            == imaging_data_simulated.exposure_time_map.in_2d
        ).all()
        assert imaging_variable.pixel_scale == imaging_data_simulated.pixel_scale
        assert imaging_variable.image.in_2d == pytest.approx(
            imaging_data_simulated.image.in_2d, 1e-4
        )
        assert (
            imaging_variable.background_noise_map.in_2d
            == imaging_data_simulated.background_noise_map.in_2d
        ).all()

    def test__noise_map_creates_nans_due_to_low_exposure_time__raises_error(self):
        image = aa.Array.from_array_2d(array_2d=np.ones((9, 9)))

        psf = al.PSF.from_gaussian(shape=(3, 3), sigma=0.1, pixel_scale=0.2)

        exposure_time_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        background_sky_map = aa.Array.from_single_value_and_shape_2d(
            value=1.0, shape_2d=image.mask.shape
        )

        with pytest.raises(exc.DataException):
            al.SimulatedImagingData.from_image_and_exposure_arrays(
                image=image,
                psf=psf,
                pixel_scale=0.1,
                exposure_time=1.0,
                exposure_time_map=exposure_time_map,
                background_sky_map=background_sky_map,
                add_noise=True,
                noise_seed=1,
            )

    def test__from_deflections_and_galaxies__same_as_manual_calculation_using_tracer(
        self
    ):

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(10, 10), pixel_scale=1.0, sub_size=1
        )

        g0 = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0),
        )

        g1 = al.Galaxy(
            redshift=1.0, light=al.light_profiles.SphericalSersic(intensity=1.0)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        deflections = tracer.deflections_from_grid(grid=grid)

        imaging_data_simulated_via_deflections = al.SimulatedImagingData.from_deflections_galaxies_and_exposure_arrays(
            deflections=deflections,
            pixel_scale=1.0,
            galaxies=[g1],
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        tracer_profile_image = tracer.profile_image_from_grid(grid=grid)

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=tracer_profile_image,
            pixel_scale=1.0,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated_via_deflections.image == imaging_data_simulated.image
        ).all()
        assert (
            imaging_data_simulated_via_deflections.psf
            == imaging_data_simulated.psf
        ).all()
        assert (
            imaging_data_simulated_via_deflections.noise_map
            == imaging_data_simulated.noise_map
        ).all()
        assert (
            imaging_data_simulated_via_deflections.background_sky_map
            == imaging_data_simulated.background_sky_map
        ).all()
        assert (
            imaging_data_simulated_via_deflections.exposure_time_map
            == imaging_data_simulated.exposure_time_map
        ).all()

    def test__from_tracer__same_as_manual_tracer_input(self):
        psf = al.PSF.from_2d_and_pixel_scale(
            array_2d=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scale=1.0,
        )

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(20, 20), pixel_scale=0.05, sub_size=1
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

        imaging_data_simulated_via_tracer = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
            tracer=tracer,
            grid=grid,
            pixel_scale=0.1,
            exposure_time=10000.0,
            psf=psf,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        imaging_data_simulated = al.SimulatedImagingData.from_image_and_exposure_arrays(
            image=tracer.padded_profile_image_2d_from_grid_and_psf_shape(
                grid=grid, psf_shape=(3, 3)
            ),
            pixel_scale=0.1,
            exposure_time=10000.0,
            psf=psf,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        assert (
            imaging_data_simulated_via_tracer.image == imaging_data_simulated.image
        ).all()
        assert (
            imaging_data_simulated_via_tracer.psf == imaging_data_simulated.psf
        ).all()
        assert (
            imaging_data_simulated_via_tracer.noise_map
            == imaging_data_simulated.noise_map
        ).all()
        assert (
            imaging_data_simulated_via_tracer.background_sky_map
            == imaging_data_simulated.background_sky_map
        ).all()
        assert (
            imaging_data_simulated_via_tracer.exposure_time_map
            == imaging_data_simulated.exposure_time_map
        ).all()


class TestSimulatePoissonNoise(object):
    def test__input_image_all_0s__exposure_time_all_1s__all_noise_values_are_0s(self):
        
        image = aa.Array.from_array_2d(np.zeros((2, 2)))
        exposure_time = aa.Array.from_array_2d(np.ones((2, 2)))
        simulated_poisson_image = image + al.generate_poisson_noise(
            image, exposure_time, seed=1
        )

        assert simulated_poisson_image.in_2d.shape == (2, 2)
        assert (simulated_poisson_image.in_2d == np.zeros((2, 2))).all()

    def test__input_image_includes_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(
        self
    ):
        image = aa.Array.from_array_2d([[10.0, 0.0], [0.0, 10.0]])

        exposure_time = aa.Array.from_array_2d(np.ones((2, 2)))
        poisson_noise_map = al.generate_poisson_noise(image, exposure_time, seed=1)
        simulated_poisson_image = image + poisson_noise_map

        assert simulated_poisson_image.in_2d.shape == (2, 2)

        # Use known noise_map_1d map for given seed.
        assert (
            poisson_noise_map.in_2d == np.array([[(10.0 - 9.0), 0], [0, (10.0 - 6.0)]])
        ).all()
        assert (simulated_poisson_image.in_2d == np.array([[11, 0], [0, 14]])).all()

        assert (simulated_poisson_image - poisson_noise_map == image).all()

    def test__input_image_is_all_10s__exposure_time_is_1s__gives_noise_values_near_1_to_5(
        self
    ):
        image = aa.Array.from_array_2d([[10.0, 10.0], [10.0, 10.0]])

        exposure_time = aa.Array.from_array_2d(np.ones((2, 2)))
        poisson_noise_map = al.generate_poisson_noise(image, exposure_time, seed=1)
        simulated_poisson_image = image + poisson_noise_map

        assert simulated_poisson_image.in_2d.shape == (2, 2)

        # Use known noise_map_1d map for given seed.
        assert (poisson_noise_map.in_2d == np.array([[1, 4], [3, 1]])).all()

        assert (simulated_poisson_image.in_2d == np.array([[11, 14], [13, 11]])).all()

        assert (simulated_poisson_image - poisson_noise_map == image).all()

    def test__input_image_has_1000000s__exposure_times_is_1s__these_give_positive_noise_values_near_1000(
        self
    ):
        image = aa.Array.from_array_2d([[10000000.0, 0.0], [0.0, 10000000.0]])

        exposure_time_map = aa.Array.from_array_2d(np.ones((2, 2)))

        poisson_noise_map = al.generate_poisson_noise(
            image=image, exposure_time_map=exposure_time_map, seed=2
        )

        simulated_poisson_image = image + poisson_noise_map

        assert simulated_poisson_image.in_2d.shape == (2, 2)

        # Use known noise_map_1d map for given seed.
        assert (poisson_noise_map.in_2d == np.array([[571, 0], [0, -441]])).all()

        assert (
            simulated_poisson_image.in_2d
            == np.array([[10000000.0 + 571, 0.0], [0.0, 10000000.0 - 441]])
        ).all()

        assert (simulated_poisson_image - poisson_noise_map == image).all()

    def test__two_images_same_in_counts_but_different_in_electrons_per_sec__noise_related_by_exposure_times(
        self
    ):
        image_0 = aa.Array.from_array_2d([[10.0, 0.0], [0.0, 10.0]])

        exposure_time_0 = aa.Array.from_array_2d(np.ones((2, 2)))

        image_1 = aa.Array.from_array_2d([[5.0, 0.0], [0.0, 5.0]])

        exposure_time_1 = 2.0 * aa.Array.from_array_2d(np.ones((2, 2)))

        simulated_poisson_image_0 = image_0 + al.generate_poisson_noise(
            image_0, exposure_time_0, seed=1
        )
        simulated_poisson_image_1 = image_1 + al.generate_poisson_noise(
            image_1, exposure_time_1, seed=1
        )

        assert (simulated_poisson_image_0 / 2.0 == simulated_poisson_image_1).all()

    def test__same_as_above_but_range_of_image_values_and_exposure_times(self):
        image_0 = aa.Array.from_array_2d([[10.0, 20.0], [30.0, 40.0]])

        exposure_time_0 = aa.Array.from_array_2d([[2.0, 2.0], [3.0, 4.0]])

        image_1 = aa.Array.from_array_2d([[20.0, 20.0], [45.0, 20.0]])

        exposure_time_1 = aa.Array.from_array_2d([[1.0, 2.0], [2.0, 8.0]])

        simulated_poisson_image_0 = image_0 + al.generate_poisson_noise(
            image_0, exposure_time_0, seed=1
        )
        simulated_poisson_image_1 = image_1 + al.generate_poisson_noise(
            image_1, exposure_time_1, seed=1
        )

        assert (
            simulated_poisson_image_0[0, 0] == simulated_poisson_image_1[0, 0] / 2.0
        ).all()
        assert simulated_poisson_image_0[0, 1] == simulated_poisson_image_1[0, 1]
        assert (
            simulated_poisson_image_0[1, 0] * 1.5
            == pytest.approx(simulated_poisson_image_1[1, 0], 1e-2)
        ).all()
        assert (
            simulated_poisson_image_0[1, 1] / 2.0 == simulated_poisson_image_1[1, 1]
        ).all()


class TestImagingFromFits(object):
    def test__no_settings_just_pass_fits(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            renormalize_psf=False,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert imaging_data.background_noise_map == None
        assert imaging_data.poisson_noise_map == None
        assert imaging_data.exposure_time_map == None
        assert imaging_data.background_sky_map == None

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1

    def test__optional_array_paths_included__loads_optional_array(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__all_files_in_one_fits__load_using_different_hdus(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_multiple_hdu.fits",
            image_hdu=0,
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_multiple_hdu.fits",
            psf_hdu=1,
            noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            noise_map_hdu=2,
            background_noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            background_noise_map_hdu=3,
            poisson_noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            poisson_noise_map_hdu=4,
            exposure_time_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            exposure_time_map_hdu=5,
            background_sky_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            background_sky_map_hdu=6,
            renormalize_psf=False,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(
        self
    ):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            noise_map_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            exposure_time_map_from_single_value=3.0,
            renormalize_psf=False,
        )

        assert (imaging_data.exposure_time_map.in_2d == 3.0 * np.ones((3, 3))).all()

    def test__exposure_time_map_from_inverse_noise_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            exposure_time_map_from_single_value=3.0,
            exposure_time_map_from_inverse_noise_map=True,
            renormalize_psf=False,
        )

        assert (imaging_data.exposure_time_map.in_2d == 3.0 * np.ones((3, 3))).all()

        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            exposure_time_map_from_single_value=6.0,
            exposure_time_map_from_inverse_noise_map=True,
            renormalize_psf=False,
        )

        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()

    def test__exposure_time_map_from_inverse_noise_map__background_noise_is_converted_from_inverse_noise_map(
        self
    ):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_ones_central_two.fits",
            convert_background_noise_map_from_inverse_noise_map=True,
            exposure_time_map_from_single_value=3.0,
            exposure_time_map_from_inverse_noise_map=True,
            renormalize_psf=False,
        )

        inverse_noise_map = aa.Array.from_array_2d(
            array_2d=np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]),
        )

        background_noise_map_converted = al.NoiseMap.from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map,
        )

        assert (
            imaging_data.background_noise_map.in_2d
            == np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert (
            imaging_data.background_noise_map.in_2d
            == background_noise_map_converted.in_2d
        ).all()

        assert (
            imaging_data.exposure_time_map.in_2d
            == np.array([[1.5, 1.5, 1.5], [1.5, 3.0, 1.5], [1.5, 1.5, 1.5]])
        ).all()

    def test__pad_shape_of_images_and_psf(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            resized_imaging_shape=(5, 5),
            resized_psf_shape=(7, 7),
            renormalize_psf=False,
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

        assert (imaging_data.image.in_2d == padded_array).all()
        assert (imaging_data.psf.in_2d == psf_padded_array).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * padded_array).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * padded_array).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * padded_array).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * padded_array).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * padded_array).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            exposure_time_map_from_single_value=3.0,
            pixel_scale=0.1,
            resized_imaging_shape=(5, 5),
            resized_psf_shape=(7, 7),
            renormalize_psf=False,
        )

        exposure_padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 3.0, 3.0, 0.0],
                [0.0, 3.0, 3.0, 3.0, 0.0],
                [0.0, 3.0, 3.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert (imaging_data.image.in_2d == padded_array).all()
        assert (imaging_data.exposure_time_map.in_2d == exposure_padded_array).all()

    def test__trim_shape_of_images_and_psf(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            resized_imaging_shape=(1, 1),
            resized_psf_shape=(1, 1),
            renormalize_psf=False,
        )

        trimmed_array = np.array([[1.0]])

        assert (imaging_data.image.in_2d == trimmed_array).all()
        assert (imaging_data.psf.in_2d == 2.0 * trimmed_array).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * trimmed_array).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * trimmed_array).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * trimmed_array).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * trimmed_array).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * trimmed_array).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_noise_map_from_weight_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            convert_noise_map_from_weight_map=True,
            renormalize_psf=False,
        )

        weight_map = aa.Array.from_array_2d(array_2d=3.0 * np.ones((3, 3)))

        noise_map_converted = al.NoiseMap.from_weight_map(
            weight_map=weight_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == noise_map_converted.in_2d).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_noise_map_from_inverse_noise_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            convert_noise_map_from_inverse_noise_map=True,
            renormalize_psf=False,
        )

        inverse_noise_map = aa.Array.from_array_2d(
            array_2d=3.0 * np.ones((3, 3)),
        )

        noise_map_converted = al.NoiseMap.from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == noise_map_converted.in_2d).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_from_image_and_background_noise_map=True,
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
        )

        noise_map_converted = al.NoiseMap.from_image_and_background_noise_map(
            image=imaging_data.image,
            background_noise_map=imaging_data.background_noise_map,
            gain=2.0,
            exposure_time_map=imaging_data.exposure_time_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == noise_map_converted.in_2d).all()
        assert (
            imaging_data.noise_map.in_2d
            == (np.sqrt((24.0) ** 2.0 + (6.0)) / (6.0)) * np.ones((3, 3))
        ).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map__include_convert_from_electrons(
        self
    ):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_from_image_and_background_noise_map=True,
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            convert_from_electrons=True,
            renormalize_psf=False,
        )

        image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))
        background_noise_map = aa.Array.from_array_2d(
            array_2d=4.0 * np.ones((3, 3))
        )

        noise_map_converted = al.NoiseMap.from_image_and_background_noise_map(
            image=image,
            background_noise_map=background_noise_map,
            gain=None,
            exposure_time_map=imaging_data.exposure_time_map,
            convert_from_electrons=True,
        )

        noise_map_converted = noise_map_converted / 6.0

        assert (imaging_data.image.in_2d == np.ones((3, 3)) / 6.0).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == noise_map_converted.in_2d).all()
        assert (
            imaging_data.noise_map.in_2d == np.sqrt(17.0) * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3)) / 6.0
        ).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__noise_map_from_image_and_background_noise_map__include_convert_from_adus(
        self
    ):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_from_image_and_background_noise_map=True,
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            gain=2.0,
            convert_from_adus=True,
            renormalize_psf=False,
        )

        image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))
        background_noise_map = aa.Array.from_array_2d(
            array_2d=4.0 * np.ones((3, 3)),
        )

        noise_map_converted = al.NoiseMap.from_image_and_background_noise_map(
            image=image,
            background_noise_map=background_noise_map,
            gain=2.0,
            exposure_time_map=imaging_data.exposure_time_map,
            convert_from_adus=True,
        )

        noise_map_converted = 2.0 * noise_map_converted / 6.0

        assert (imaging_data.image.in_2d == 2.0 * np.ones((3, 3)) / 6.0).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == noise_map_converted.in_2d).all()
        assert (
            imaging_data.noise_map.in_2d == np.sqrt(66.0) * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.background_noise_map.in_2d == 2.0 * 4.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == 2.0 * 5.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_sky_map.in_2d == 2.0 * 7.0 * np.ones((3, 3)) / 6.0
        ).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_background_noise_map_from_weight_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            convert_background_noise_map_from_weight_map=True,
        )

        weight_map = aa.Array.from_array_2d(array_2d=4.0 * np.ones((3, 3)))

        background_noise_map_converted = al.NoiseMap.from_weight_map(
            weight_map=weight_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_noise_map.in_2d
            == background_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_background_noise_map_from_inverse_noise_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            convert_background_noise_map_from_inverse_noise_map=True,
        )

        inverse_noise_map = aa.Array.from_array_2d(
            array_2d=4.0 * np.ones((3, 3)),
        )

        background_noise_map_converted = al.NoiseMap.from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_noise_map.in_2d
            == background_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__poisson_noise_map_from_image(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            poisson_noise_map_from_image=True,
        )

        image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))

        poisson_noise_map_converted = al.PoissonNoiseMap.from_image_and_exposure_time_map(
            image=image,
            exposure_time_map=imaging_data.exposure_time_map,
            gain=None,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.poisson_noise_map.in_2d
            == (np.sqrt(6.0) / (6.0)) * np.ones((3, 3))
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == poisson_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__poisson_noise_map_from_image__include_convert_from_electrons(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            poisson_noise_map_from_image=True,
            convert_from_electrons=True,
        )

        image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))

        poisson_noise_map_counts = al.PoissonNoiseMap.from_image_and_exposure_time_map(
            image=image,
            exposure_time_map=imaging_data.exposure_time_map,
            gain=None,
            convert_from_electrons=True,
        )

        poisson_noise_map_converted = poisson_noise_map_counts / 6.0

        assert (imaging_data.image.in_2d == np.ones((3, 3)) / 6.0).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3)) / 6.0).all()
        assert (
            imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (imaging_data.poisson_noise_map.in_2d == np.ones((3, 3)) / 6.0).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == poisson_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3)) / 6.0
        ).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__poisson_noise_map_from_image__include_convert_from_adus(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            poisson_noise_map_from_image=True,
            gain=2.0,
            convert_from_adus=True,
        )

        image = aa.Array.from_array_2d(array_2d=np.ones((3, 3)))

        poisson_noise_map_counts = al.PoissonNoiseMap.from_image_and_exposure_time_map(
            image=image,
            exposure_time_map=imaging_data.exposure_time_map,
            gain=2.0,
            convert_from_adus=True,
        )

        poisson_noise_map_converted = 2.0 * poisson_noise_map_counts / 6.0

        assert (imaging_data.image.in_2d == 2.0 * np.ones((3, 3)) / 6.0).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 2.0 * 3.0 * np.ones((3, 3)) / 6.0).all()
        assert (
            imaging_data.background_noise_map.in_2d == 2.0 * 4.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == np.sqrt(2.0 * np.ones((3, 3))) / 6.0
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == poisson_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_sky_map.in_2d == 2.0 * 7.0 * np.ones((3, 3)) / 6.0
        ).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_poisson_noise_map_from_weight_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            convert_poisson_noise_map_from_weight_map=True,
        )

        weight_map = aa.Array.from_array_2d(array_2d=5.0 * np.ones((3, 3)))

        poisson_noise_map_converted = al.NoiseMap.from_weight_map(
            weight_map=weight_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == poisson_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_poisson_noise_map_from_inverse_noise_map(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            convert_poisson_noise_map_from_inverse_noise_map=True,
        )

        inverse_noise_map = aa.Array.from_array_2d(
            array_2d=5.0 * np.ones((3, 3)),
        )

        poisson_noise_map_converted = al.NoiseMap.from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == poisson_noise_map_converted.in_2d
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__psf_renormalized_true__renormalized_psf(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=True,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert imaging_data.psf.in_2d == pytest.approx(
            (1.0 / 9.0) * np.ones((3, 3)), 1e-2
        )
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_image_from_electrons_using_exposure_time(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            convert_from_electrons=True,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3)) / 6.0).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3)) / 6.0).all()
        assert (
            imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3)) / 6.0
        ).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__convert_image_from_adus_using_exposure_time_and_gain(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
            gain=2.0,
            convert_from_adus=True,
        )

        assert (imaging_data.image.in_2d == 2.0 * np.ones((3, 3)) / 6.0).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 2.0 * 3.0 * np.ones((3, 3)) / 6.0).all()
        assert (
            imaging_data.background_noise_map.in_2d == 2.0 * 4.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (
            imaging_data.poisson_noise_map.in_2d == 2.0 * 5.0 * np.ones((3, 3)) / 6.0
        ).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (
            imaging_data.background_sky_map.in_2d == 2.0 * 7.0 * np.ones((3, 3)) / 6.0
        ).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1

    def test__no_noise_map_input__raises_imaging_exception(self):
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
            )

    def test__multiple_noise_map_options__raises_imaging_exception(self):
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
                noise_map_path=test_data_dir + "3x3_threes.fits",
                convert_noise_map_from_inverse_noise_map=True,
                convert_noise_map_from_weight_map=True,
            )

        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
                noise_map_path=test_data_dir + "3x3_threes.fits",
                convert_noise_map_from_inverse_noise_map=True,
                noise_map_from_image_and_background_noise_map=True,
            )

        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                psf_path=test_data_dir + "3x3_twos.fits",
                noise_map_path=test_data_dir + "3x3_threes.fits",
                noise_map_from_image_and_background_noise_map=True,
                convert_noise_map_from_weight_map=True,
            )

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                exposure_time_map_path=test_data_dir + "3x3_ones.fits",
                exposure_time_map_from_single_value=1.0,
            )

    def test__noise_map_from_image_and_background_noise_map_exceptions(self):
        # need background noise_map map - raise error if not present
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                exposure_time_map_from_single_value=1.0,
                noise_map_from_image_and_background_noise_map=True,
            )

        # Dont need gain if datas is in electrons
        al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            exposure_time_map_from_single_value=1.0,
            noise_map_from_image_and_background_noise_map=True,
            convert_from_electrons=True,
        )

        # Need gain if datas is in adus
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                background_noise_map_path=test_data_dir + "3x3_fours.fits",
                noise_map_from_image_and_background_noise_map=True,
                convert_from_adus=True,
            )

        # No error if datas already in adus
        al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            exposure_time_map_from_single_value=1.0,
            noise_map_from_image_and_background_noise_map=True,
            gain=1.0,
            convert_from_adus=True,
        )

    def test__poisson_noise_map_from_image_exceptions(self):
        # Dont need gain if datas is in e/s
        al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            noise_map_path=test_data_dir + "3x3_threes.fits",
            exposure_time_map_from_single_value=1.0,
            poisson_noise_map_from_image=True,
        )

        # No exposure time - not load
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                poisson_noise_map_from_image=True,
                convert_from_electrons=True,
            )

        # Need gain if datas in adus
        with pytest.raises(exc.DataException):
            al.load_imaging_data_from_fits(
                image_path=test_data_dir + "3x3_ones.fits",
                psf_path=test_data_dir + "3x3_ones.fits",
                pixel_scale=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                background_noise_map_path=test_data_dir + "3x3_fours.fits",
                exposure_time_map_from_single_value=1.0,
                poisson_noise_map_from_image=True,
                convert_from_adus=True,
            )

    def test__output_all_arrays(self):
        imaging_data = al.load_imaging_data_from_fits(
            image_path=test_data_dir + "3x3_ones.fits",
            pixel_scale=0.1,
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            background_noise_map_path=test_data_dir + "3x3_fours.fits",
            poisson_noise_map_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3x3_sixes.fits",
            background_sky_map_path=test_data_dir + "3x3_sevens.fits",
            renormalize_psf=False,
        )

        output_data_dir = "{}/../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        al.output_imaging_data_to_fits(
            imaging_data=imaging_data,
            image_path=output_data_dir + "image.fits",
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            background_noise_map_path=output_data_dir + "background_noise_map.fits",
            poisson_noise_map_path=output_data_dir + "poisson_noise_map.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            background_sky_map_path=output_data_dir + "background_sky_map.fits",
        )

        imaging_data = al.load_imaging_data_from_fits(
            image_path=output_data_dir + "image.fits",
            pixel_scale=0.1,
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            background_noise_map_path=output_data_dir + "background_noise_map.fits",
            poisson_noise_map_path=output_data_dir + "poisson_noise_map.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            background_sky_map_path=output_data_dir + "background_sky_map.fits",
            renormalize_psf=False,
        )

        assert (imaging_data.image.in_2d == np.ones((3, 3))).all()
        assert (imaging_data.psf.in_2d == 2.0 * np.ones((3, 3))).all()
        assert (imaging_data.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_noise_map.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (imaging_data.poisson_noise_map.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (imaging_data.exposure_time_map.in_2d == 6.0 * np.ones((3, 3))).all()
        assert (imaging_data.background_sky_map.in_2d == 7.0 * np.ones((3, 3))).all()

        assert imaging_data.pixel_scale == 0.1
        assert imaging_data.psf.mask.geometry.pixel_scale == 0.1
        assert imaging_data.noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.poisson_noise_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.exposure_time_map.mask.geometry.pixel_scale == 0.1
        assert imaging_data.background_sky_map.mask.geometry.pixel_scale == 0.1
