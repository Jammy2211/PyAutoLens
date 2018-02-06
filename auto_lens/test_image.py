from __future__ import division, print_function
import pytest
import numpy as np
import image
import os

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestData(object):
    class TestSetup(object):
        def test__init__input_data_3x3__all_attributes_correct(self):
            test_data = image.Image(array=np.ones((3, 3)), pixel_scale=0.1)

            assert (test_data == np.ones((3, 3))).all()
            assert test_data.pixel_scale == 0.1
            assert test_data.shape == (3, 3)
            assert test_data.central_pixel_coordinates == (1.0, 1.0)
            assert test_data.shape_arc_seconds == pytest.approx((0.3, 0.3))

        def test__init__input_data_4x3__all_attributes_correct(self):
            test_data = image.Image(array=np.ones((4, 3)), pixel_scale=0.1)

            assert (test_data == np.ones((4, 3))).all()
            assert test_data.pixel_scale == 0.1
            assert test_data.shape == (4, 3)
            assert test_data.central_pixel_coordinates == (1.5, 1.0)
            assert test_data.shape_arc_seconds == pytest.approx((0.4, 0.3))

    class TestTrimData(object):
        class TestOddToOdd(object):
            def test__trimmed_5x5_to_3x3(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(3, 3))

                assert (data == np.array([[1.0, 1.0, 1.0],
                                                [1.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0]])).all()

                assert data.shape == (3, 3)
                assert data.central_pixel_coordinates == (1.0, 1.0)
                assert data.shape_arc_seconds == pytest.approx((0.3, 0.3), 1e-3)

            def test__trimmed_7x7_to_3x3(self):
                data = np.ones((7, 7))
                data[3, 3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(3, 3))

                assert (data == np.array([[1.0, 1.0, 1.0],
                                                [1.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0]])).all()

                assert data.shape == (3, 3)
                assert data.central_pixel_coordinates == (1.0, 1.0)
                assert data.shape_arc_seconds == pytest.approx((0.3, 0.3), 1e-3)

            def test__trimmed_11x11_to_5x5(self):
                data = np.ones((11, 11))
                data[5, 5] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(5, 5))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (5, 5)
                assert data.central_pixel_coordinates == (2.0, 2.0)
                assert data.shape_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

        class TestOddToEven(object):
            def test__trimmed_5x5_to_2x2__goes_to_3x3_to_keep_symmetry(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(2, 2))

                assert (data == np.array([[1.0, 1.0, 1.0],
                                                [1.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0]])).all()

                assert data.shape == (3, 3)
                assert data.central_pixel_coordinates == (1.0, 1.0)
                assert data.shape_arc_seconds == pytest.approx((0.3, 0.3), 1e-3)

            def test__trimmed_5x5_to_4x4__goes_to_5x5_to_keep_symmetry(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(4, 4))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (5, 5)
                assert data.central_pixel_coordinates == (2.0, 2.0)
                assert data.shape_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

            def test__trimmed_11x11_to_4x4__goes_to_5x5_to_keep_symmetry(self):
                data = np.ones((11, 11))
                data[5, 5] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(4, 4))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (5, 5)
                assert data.central_pixel_coordinates == (2.0, 2.0)
                assert data.shape_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

        class TestEvenToEven(object):
            def test__trimmed_4x4_to_2x2(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(2, 2))

                assert (data == np.array([[2.0, 2.0],
                                                [2.0, 2.0]])).all()

                assert data.shape == (2, 2)
                assert data.central_pixel_coordinates == (0.5, 0.5)
                assert data.shape_arc_seconds == pytest.approx((0.2, 0.2), 1e-3)

            def test__trimmed_6x6_to_4x4(self):
                data = np.ones((6, 6))
                data[2:4, 2:4] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(4, 4))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (4, 4)
                assert data.central_pixel_coordinates == (1.5, 1.5)
                assert data.shape_arc_seconds == pytest.approx((0.4, 0.4), 1e-3)

            def test__trimmed_12x12_to_6x6(self):
                data = np.ones((12, 12))
                data[5:7, 5:7] = 2.0
                data[4, 4] = 9.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(6, 6))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (6, 6)
                assert data.central_pixel_coordinates == (2.5, 2.5)
                assert data.shape_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

        class TestEvenToOdd(object):
            def test__trimmed_4x4_to_3x3__goes_to_4x4_to_keep_symmetry(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(3, 3))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (4, 4)
                assert data.central_pixel_coordinates == (1.5, 1.5)
                assert data.shape_arc_seconds == pytest.approx((0.4, 0.4), 1e-3)

            def test__trimmed_6x6_to_3x3_goes_to_4x4_to_keep_symmetry(self):
                data = np.ones((6, 6))
                data[2:4, 2:4] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(3, 3))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (4, 4)
                assert data.central_pixel_coordinates == (1.5, 1.5)
                assert data.shape_arc_seconds == pytest.approx((0.4, 0.4), 1e-3)

            def test__trimmed_12x12_to_5x5__goes_to_6x6_to_keep_symmetry(self):
                data = np.ones((12, 12))
                data[5:7, 5:7] = 2.0
                data[4, 4] = 9.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(5, 5))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.shape == (6, 6)
                assert data.central_pixel_coordinates == (2.5, 2.5)
                assert data.shape_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

        class TestRectangle(object):
            def test__trimmed_5x4_to_3x2(self):
                data = np.ones((5, 4))
                data[2, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(3, 2))

                assert (data == np.array([[1.0, 1.0],
                                                [2.0, 2.0],
                                                [1.0, 1.0]])).all()

                assert data.shape == (3, 2)
                assert data.central_pixel_coordinates == (1, 0.5)
                assert data.shape_arc_seconds == pytest.approx((0.3, 0.2), 1e-3)

            def test__trimmed_4x5_to_2x3(self):
                data = np.ones((4, 5))
                data[1:3, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(2, 3))

                assert (data == np.array([[1.0, 2.0, 1.0],
                                                [1.0, 2.0, 1.0]])).all()

                assert data.shape == (2, 3)
                assert data.central_pixel_coordinates == (0.5, 1.0)
                assert data.shape_arc_seconds == pytest.approx((0.2, 0.3), 1e-3)

            def test__trimmed_5x4_to_4x3__goes_to_5x4_to_keep_symmetry(self):
                data = np.ones((5, 4))
                data[2, 1:3] = 2.0
                data[4, 3] = 9.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.trim(data, pixel_dimensions=(4, 3))

                assert (data == np.array([[1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0],
                                                [1.0, 2.0, 2.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0, 9.0]])).all()

                assert data.shape == (5, 4)
                assert data.central_pixel_coordinates == (2, 1.5)
                assert data.shape_arc_seconds == pytest.approx((0.5, 0.4), 1e-3)

        class TestBadInput(object):
            def test__x_size_bigger_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert image.trim(data, pixel_dimensions=(8, 3))

            def test__y_size_bigger_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert image.trim(data, pixel_dimensions=(3, 8))

    class TestPadData(object):
        class TestType(object):
            def test__image(self):
                data = image.Image(np.ones((3, 3)), pixel_scale=0.1)
                assert image.trim(data, pixel_dimensions=(1, 1)).__class__ == image.Image

            def test__psf(self):
                data = image.PSF.from_array(np.ones((3, 3)))
                assert image.trim(data, pixel_dimensions=(1, 1)).__class__ == image.PSF

            def test__noise(self):
                data = image.Noise.from_array(np.ones((3, 3)))
                assert image.trim(data, pixel_dimensions=(1, 1)).__class__ == image.Noise

            def test__array(self):
                data = np.ones((3, 3))
                assert image.trim(data, pixel_dimensions=(1, 1)).__class__ == np.ndarray

        class TestOddToOdd(object):
            def test__padded_3x3_to_5x5(self):
                data = np.ones((3, 3))
                data[1, 1] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(5, 5))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (5, 5)
                assert data.central_pixel_coordinates == (2.0, 2.0)
                assert data.shape_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

            def test__padded_5x5_to_9x9(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(9, 9))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (9, 9)
                assert data.central_pixel_coordinates == (4.0, 4.0)
                assert data.shape_arc_seconds == pytest.approx((0.9, 0.9), 1e-3)

        class TestOddToEven(object):
            def test__padded_3x3_to_4x4__goes_to_5x5_to_keep_symmetry(self):
                data = np.ones((3, 3))
                data[1, 1] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(4, 4))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (5, 5)
                assert data.central_pixel_coordinates == (2.0, 2.0)
                assert data.shape_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

            def test__padded_5x5_to_8x8__goes_to_9x9_to_keep_symmetry(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(8, 8))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (9, 9)
                assert data.central_pixel_coordinates == (4.0, 4.0)
                assert data.shape_arc_seconds == pytest.approx((0.9, 0.9), 1e-3)

        class TestEvenToEven(object):
            def test__padded_4x4_to_6x6(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(6, 6))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (6, 6)
                assert data.central_pixel_coordinates == (2.5, 2.5)
                assert data.shape_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

            def test__padded_4x4_to_8x8(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(8, 8))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (8, 8)
                assert data.central_pixel_coordinates == (3.5, 3.5)
                assert data.shape_arc_seconds == pytest.approx((0.8, 0.8), 1e-3)

        class TestEvenToOdd(object):
            def test__padded_4x4_to_5x5__goes_to_6x6_to_keep_symmetry(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(5, 5))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (6, 6)
                assert data.central_pixel_coordinates == (2.5, 2.5)
                assert data.shape_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

            def test__padded_4x4_to_7x7__goes_to_8x8_to_keep_symmetry(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(7, 7))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (8, 8)
                assert data.central_pixel_coordinates == (3.5, 3.5)
                assert data.shape_arc_seconds == pytest.approx((0.8, 0.8), 1e-3)

        class TestRectangle(object):
            def test__padded_5x4_to_7x6(self):
                data = np.ones((5, 4))
                data[2, 1:3] = 2.0

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(7, 6))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (7, 6)
                assert data.central_pixel_coordinates == (3, 2.5)
                assert data.shape_arc_seconds == pytest.approx((0.7, 0.6), 1e-3)

            def test__padded_2x3_to_6x7(self):
                data = np.ones((2, 3))
                data[0:2, 1] = 2.0
                data[1, 2] = 9

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(6, 7))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (6, 7)
                assert data.central_pixel_coordinates == (2.5, 3.0)
                assert data.shape_arc_seconds == pytest.approx((0.6, 0.7), 1e-3)

            def test__padded_2x3_to_5x6__goes_to_6x7_to_keep_symmetry(self):
                data = np.ones((2, 3))
                data[0:2, 1] = 2.0
                data[1, 2] = 9

                data = image.Image(data, pixel_scale=0.1)
                data = image.pad(data, pixel_dimensions=(5, 6))

                assert (data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.shape == (6, 7)
                assert data.central_pixel_coordinates == (2.5, 3.0)
                assert data.shape_arc_seconds == pytest.approx((0.6, 0.7), 1e-3)

        class TestBadInput(object):
            def test__x_size_smaller_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert image.trim(data, pixel_dimensions=(3, 8))

            def test__y_size_smaller_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Image(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert image.trim(data, pixel_dimensions=(8, 3))


class TestImage(object):
    class TestSetup(object):
        def test__init__input_image_3x3__all_attributes_correct(self):
            test_image = image.Image(array=np.ones((3, 3)), pixel_scale=0.1)

            assert (test_image == np.ones((3, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.shape == (3, 3)
            assert test_image.central_pixel_coordinates == (1.0, 1.0)
            assert test_image.shape_arc_seconds == pytest.approx((0.3, 0.3))

        def test__init__input_image_4x3__all_attributes_correct(self):
            test_image = image.Image(array=np.ones((4, 3)), pixel_scale=0.1)

            assert (test_image == np.ones((4, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.shape == (4, 3)
            assert test_image.central_pixel_coordinates == (1.5, 1.0)
            assert test_image.shape_arc_seconds == pytest.approx((0.4, 0.3))

        def test__from_fits__input_image_3x3__all_attributes_correct(self):
            test_image = image.Image.from_fits('3x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)

            assert (test_image == np.ones((3, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.shape == (3, 3)
            assert test_image.central_pixel_coordinates == (1.0, 1.0)
            assert test_image.shape_arc_seconds == pytest.approx((0.3, 0.3))

        def test__from_fits__input_image_4x3__all_attributes_correct(self):
            test_image = image.Image.from_fits('4x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)

            assert (test_image == np.ones((4, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.shape == (4, 3)
            assert test_image.central_pixel_coordinates == (1.5, 1.0)
            assert test_image.shape_arc_seconds == pytest.approx((0.4, 0.3))

    class TestSetSky(object):
        def test__via_edges__input_all_ones__sky_bg_level_1(self):
            test_image = image.Image(array=np.ones((3, 3)), pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):
            image_data = np.array([[1, 1, 1],
                                   [1, 100, 1],
                                   [1, 1, 1]])

            test_image = image.Image(array=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):
            image_data = np.array([[1, 1, 1],
                                   [1, 100, 1],
                                   [1, 100, 1],
                                   [1, 1, 1]])

            test_image = image.Image(array=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image_data = np.array([[1, 1, 1, 1],
                                   [1, 100, 100, 1],
                                   [1, 100, 100, 1],
                                   [1, 1, 1, 1]])

            test_image = image.Image(array=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):
            image_data = np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 100, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]])

            test_image = image.Image(array=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=2)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__6x5_image_two_edges__correct_values(self):
            image_data = np.array([[0, 1, 2, 3, 4],
                                   [5, 6, 7, 8, 9],
                                   [10, 11, 100, 12, 13],
                                   [14, 15, 100, 16, 17],
                                   [18, 19, 20, 21, 22],
                                   [23, 24, 25, 26, 27]])

            test_image = image.Image(array=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=2)

            assert test_image.sky_background_level == np.mean(np.arange(28))
            assert test_image.sky_background_noise == np.std(np.arange(28))

        def test__via_edges__7x7_image_three_edges__correct_values(self):
            image_data = np.array([[0, 1, 2, 3, 4, 5, 6],
                                   [7, 8, 9, 10, 11, 12, 13],
                                   [14, 15, 16, 17, 18, 19, 20],
                                   [21, 22, 23, 100, 24, 25, 26],
                                   [27, 28, 29, 30, 31, 32, 33],
                                   [34, 35, 36, 37, 38, 39, 40],
                                   [41, 42, 43, 44, 45, 46, 47]])

            test_image = image.Image(array=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=3)

            assert test_image.sky_background_level == np.mean(np.arange(48))
            assert test_image.sky_background_noise == np.std(np.arange(48))


class TestPSF(object):
    class TestSetup(object):
        def test__init__input_psf_3x3__all_attributes_correct(self):
            test_psf = image.PSF.from_array(array=np.ones((3, 3)), renormalize=False)

            assert (test_psf == np.ones((3, 3))).all()

        def test__init__input_psf_4x3__all_attributes_correct(self):
            test_psf = image.PSF.from_array(array=np.ones((4, 3)), renormalize=False)

            assert (test_psf == np.ones((4, 3))).all()

        def test__from_fits__input_psf_3x3__all_attributes_correct(self):
            test_psf = image.PSF.from_fits('3x3_ones.fits', hdu=0, renormalize=False, path=test_data_dir)

            assert (test_psf == np.ones((3, 3))).all()

        def test__from_fits__input_psf_4x3__all_attributes_correct(self):
            test_psf = image.PSF.from_fits('4x3_ones.fits', hdu=0, renormalize=False, path=test_data_dir)

            assert (test_psf == np.ones((4, 3))).all()

    class TestNormalize(object):
        def test__input_is_already_normalized__no_change(self):
            psf_data = np.ones((3, 3)) / 9.0

            assert image.normalize(psf_data) == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization__correctly_normalized(self):
            psf_data = np.ones((3, 3))

            assert image.normalize(psf_data) == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__input_is_below_normalization__correctly_normalized(self):
            psf_data = np.ones((3, 3)) / 90.0

            assert image.normalize(psf_data) == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__input_different_psf__correctly_normalized(self):
            psf_data = np.ones((4, 4))
            psf_data[1:3, 1:3] = 2.0

            normalization_factor = 2.0 * 4.0 + 12

            assert image.normalize(psf_data) == pytest.approx(psf_data / normalization_factor, 1e-3)


class TestNoise(object):
    class TestSetup(object):
        def test__init__input_noise_3x3__all_attributes_correct(self):
            test_noise = image.Noise.from_array(array=np.ones((3, 3)))

            assert (test_noise == np.ones((3, 3))).all()

        def test__init__input_noise_4x3__all_attributes_correct(self):
            test_noise = image.Noise.from_array(array=np.ones((4, 3)))

            assert (test_noise == np.ones((4, 3))).all()

        def test__from_fits__input_noise_3x3__all_attributes_correct(self):
            test_noise = image.Noise.from_fits('3x3_ones.fits', hdu=0, path=test_data_dir)

            assert (test_noise == np.ones((3, 3))).all()

        def test__from_fits__input_noise_4x3__all_attributes_correct(self):
            test_noise = image.Noise.from_fits('4x3_ones.fits', hdu=0, path=test_data_dir)

            assert (test_noise == np.ones((4, 3))).all()


class TestMask(object):
    class TestPixelScale(object):
        def test__central_pixel(self):
            assert image.central_pixel((3, 3)) == (1.0, 1.0)

        def test__shape(self):
            assert image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=5).shape == (3, 3)
            assert image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=0.5, radius=5).shape == (6, 6)
            assert image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=0.2, radius=5).shape == (15, 15)

        def test__odd_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=0.5, radius=0.5)
            assert (mask == np.array([[False, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, True, True, False, False],
                                      [False, False, True, True, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False]])).all()

    class TestCentre(object):
        def test__simple_shift_back(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5, centre=(-1, 0))
            assert mask.shape == (3, 3)
            assert (mask == np.array([[False, True, False],
                                      [False, False, False],
                                      [False, False, False]])).all()

        def test__simple_shift_forward(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5, centre=(0, 1))
            assert mask.shape == (3, 3)
            assert (mask == np.array([[False, False, False],
                                      [False, False, True],
                                      [False, False, False]])).all()

        def test__diagonal_shift(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5, centre=(1, 1))
            assert (mask == np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, True]])).all()

    class TestCircular(object):
        def test__input_big_mask__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=5)
            assert mask.shape == (3, 3)
            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__odd_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5)
            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=1)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__odd_x_odd_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__even_x_odd_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius=1.50001)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_even_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius=0.72)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, False],
                                      [False, True, True, False],
                                      [False, False, False, False]])).all()

        def test__even_x_even_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius=1.7)

            assert (mask == np.array([[False, True, True, False],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True]])).all()

    class TestAnnular(object):
        def test__odd_x_odd_mask_inner_radius_zero_outer_radius_small__correct_mask(self):
            mask = image.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius=0, outer_radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_inner_radius_small_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius=0.5, outer_radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
            mask = image.Mask.annular(arc_second_dimensions=(4, 3), pixel_scale=1, inner_radius=0.51, outer_radius=1.51)

            assert (mask == np.array([[False, True, False],
                                      [True, False, True],
                                      [True, False, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(arc_second_dimensions=(4, 3), pixel_scale=1, inner_radius=1.51, outer_radius=3)

            assert (mask == np.array([[True, False, True],
                                      [False, False, False],
                                      [False, False, False],
                                      [True, False, True]])).all()

        def test__even_x_even_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
            mask = image.Mask.annular(arc_second_dimensions=(4, 4), pixel_scale=1, inner_radius=0.81, outer_radius=2)

            assert (mask == np.array([[False, True, True, False],
                                      [True, False, False, True],
                                      [True, False, False, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(arc_second_dimensions=(4, 4), pixel_scale=1, inner_radius=1.71, outer_radius=3)

            assert (mask == np.array([[True, False, False, True],
                                      [False, False, False, False],
                                      [False, False, False, False],
                                      [True, False, False, True]])).all()

    class TestBlurringRegion(object):

        def test__size_is_3x3_small_mask(self):

            mask = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True, True, True],
                                                [True, False, True],
                                                [True, True, True]])).all()

        def test__size_is_3x3__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, False, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size_is_5x5__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(5, 5))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, False, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size_is_5x3__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(5, 3))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, False, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size_is_3x5__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(3, 5))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, False, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size_is_3x3__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True,   True,  True, False,  True,  True,  True],
                                                 [True,  False,  True, False,  True, False,  True],
                                                 [True,   True,  True, False,  True,  True,  True],
                                                 [False, False, False, False, False, False, False],
                                                 [True,   True,  True, False,  True,  True,  True],
                                                 [True,  False,  True, False,  True, False,  True],
                                                 [True,   True,  True, False,  True,  True,  True]])).all()

        def test__size_is_5x5__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(5, 5))

            assert (blurring_region == np.array([[True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True]])).all()

        def test__size_is_5x3__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(5, 3))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False, False, False],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [False, False, False, False, False, False, False, False, False],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [False, False, False, False, False, False, False, False, False]])).all()

        def test__size_is_3x5__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(3, 5))

            assert (blurring_region == np.array([[False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,  False,  True, False,  True, False,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,  False,  True, False,  True, False,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False]])).all()

        def test__size_is_3x3__even_sized_image(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True,   True,  True, False,  True,  True,  True, False],
                                                 [True,  False,  True, False,  True, False,  True, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [True,  False,  True, False,  True, False,  True, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [False, False, False, False, False, False, False, False]])).all()

        def test__size_is_5x5__even_sized_image(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(5, 5))

            assert (blurring_region == np.array([[False, False, False,  False, False, False, False, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [False, False, False, True,  True,  True,  True, True],
                                                 [False, False, False, True,  True,  True,  True, True],
                                                 [False, False, False, True,  True, False,  True, True],
                                                 [False, False, False, True,  True,  True,  True, True],
                                                 [False, False, False,  True,  True,  True,  True, True]])).all()

        def test__size_is_5x5__multiple_points__mask_extends_beyond_border_so_raises_mask_exception(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False]])

            with pytest.raises(image.MaskException):
                blurring_region = image.Mask.blurring_region(mask, blurring_region_size=(5, 5))

        def test__setup_via_image(self):

            image1 = image.Image(array=np.zeros((3,3)), pixel_scale=0.1)
            
            mask = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]])
            
            blurring_region = image1.blurring_region(mask=mask, psf_kernel_size=(3,3))
            
            assert (blurring_region == np.array([[True, True, True],
                                                [True, False, True],
                                                [True, True, True]])).all()

    class TestBorderPixels(object):

        def test__7x7_mask_one_central_pixel__is_entire_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = image.Mask.border_pixels(mask)

            assert (border_pixels == np.array([0])).all()

        def test__7x7_mask_nine_central_pixels__is_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = image.Mask.border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

        def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = image.Mask.border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

        def test__7x7_mask_add_edge_pixels__also_in_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True,  True,  True,  True, True, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = image.Mask.border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18])).all()

        def test__7x7_mask_big_square(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, True, True, True,  True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = image.Mask.border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 30, 31, 32, 33, 34])).all()

    class TestSparseClusteringImagePixelList(object):

        def test__7x7_circle_mask__five_central_pixels__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.array([0, 1, 2, 3, 4])).all()

        def test__7x7_circle_mask__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.arange(21)).all()
            
        def test__7x7_rectangle_mask__sparse_grid_size_1(self):

            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.arange(49)).all()

        def test__7x7_circle_mask__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([4, 6, 14, 16])).all()

        def test__8x8_sporadic_mask__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, True, True],
                             [False, False, True,  True,  True, False, True, True],
                             [False, True, True,  True,  True, True, True, True],
                             [False, True, True,  True,  True, True, True, True],
                             [False, True, True,  True,  True, True, True, True],
                             [False, False, True,  True, True, False, True, True],
                             [False, False, False, False, False, False, True, True],
                             [False, False, True, True, True, False, True, True]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([0, 8, 10, 12, 22, 24, 26, 33])).all()

        def test__7x7_circle_mask_trues_on_even_values__sparse_grid_size_2(self):

            mask = np.array([[True,  False, True, False, True, False, True],
                             [False, False, False, False, False, False, False],
                             [True,  False, True, False, True, False, True],
                             [False, False, False, False, False, False, False],
                             [True,  False, True, False, True, False, True],
                             [False, False, False, False, False, False, False],
                             [True,  False, True, False, True, False, True]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.arange(16)).all()

        def test__7x7_circle_mask__sparse_grid_size_3(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.array([10])).all()

        def test__7x7_circle_mask_more_points_added__sparse_grid_size_3(self):

            mask = np.array([[True, False, False, True, False, True, True],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, True],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, True]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.array([0, 1, 3, 14, 17, 26])).all()

        def test__7x7_mask_trues_on_values_which_divide_by_3__sparse_grid_size_3(self):

            mask = np.array([[True,  False, False, True, False, False, True],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True, False, False,  True, False, False,  True],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True,  False, False, True, False, False, True]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.arange(9)).all()

        def test__8x8_mask_trues_on_values_which_divide_by_3_and_other_values__sparse_grid_size_3(self):

            mask = np.array([[True,  False, True, True, False, False, True],
                             [False, False, False, False, False, False, False],
                             [False, False, True, True, True, False, False],
                             [True, False, False,  True, False, False,  True],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True,  True, True, True, True, True, True]])

            sparse_list = image.Mask.sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.array([0, 2, 3, 7, 8, 9, 10, 13, 16])).all()

    # class TestSubPixelsToSparsePixels(object):
    #
    #     def test__7x7_mask__one_sparse_pixel__2x2_sub_grid_in_it(self):

