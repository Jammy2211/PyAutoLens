from __future__ import division, print_function
import pytest
import numpy as np
import image
import os

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestData(object):
    class TestSetup(object):
        def test__init__input_data_3x3__all_attributes_correct(self):
            test_data = image.Data(data=np.ones((3, 3)), pixel_scale=0.1)

            assert (test_data.data == np.ones((3, 3))).all()
            assert test_data.pixel_scale == 0.1
            assert test_data.dimensions == (3, 3)
            assert test_data.central_pixels == (1.0, 1.0)
            assert test_data.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__init__input_data_4x3__all_attributes_correct(self):
            test_data = image.Data(data=np.ones((4, 3)), pixel_scale=0.1)

            assert (test_data.data == np.ones((4, 3))).all()
            assert test_data.pixel_scale == 0.1
            assert test_data.dimensions == (4, 3)
            assert test_data.central_pixels == (1.5, 1.0)
            assert test_data.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

    class TestTrimData(object):
        class TestOddToOdd(object):
            def test__trim_5x5_to_3x3(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(3, 3))

                assert (data.data == np.array([[1.0, 1.0, 1.0],
                                               [1.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (3, 3)
                assert data.central_pixels == (1.0, 1.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.3, 0.3), 1e-3)

            def test__trim_7x7_to_3x3(self):
                data = np.ones((7, 7))
                data[3, 3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(3, 3))

                assert (data.data == np.array([[1.0, 1.0, 1.0],
                                               [1.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (3, 3)
                assert data.central_pixels == (1.0, 1.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.3, 0.3), 1e-3)

            def test__trim_11x11_to_5x5(self):
                data = np.ones((11, 11))
                data[5, 5] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(5, 5))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (5, 5)
                assert data.central_pixels == (2.0, 2.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

        class TestOddToEven(object):
            def test__trim_5x5_to_2x2__goes_to_3x3_to_keep_symmetry(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(2, 2))

                assert (data.data == np.array([[1.0, 1.0, 1.0],
                                               [1.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (3, 3)
                assert data.central_pixels == (1.0, 1.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.3, 0.3), 1e-3)

            def test__trim_5x5_to_4x4__goes_to_5x5_to_keep_symmetry(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(4, 4))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (5, 5)
                assert data.central_pixels == (2.0, 2.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

            def test__trim_11x11_to_4x4__goes_to_5x5_to_keep_symmetry(self):
                data = np.ones((11, 11))
                data[5, 5] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(4, 4))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (5, 5)
                assert data.central_pixels == (2.0, 2.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

        class TestEvenToEven(object):
            def test__trim_4x4_to_2x2(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(2, 2))

                assert (data.data == np.array([[2.0, 2.0],
                                               [2.0, 2.0]])).all()

                assert data.dimensions == (2, 2)
                assert data.central_pixels == (0.5, 0.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.2, 0.2), 1e-3)

            def test__trim_6x6_to_4x4(self):
                data = np.ones((6, 6))
                data[2:4, 2:4] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(4, 4))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (4, 4)
                assert data.central_pixels == (1.5, 1.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.4, 0.4), 1e-3)

            def test__trim_12x12_to_6x6(self):
                data = np.ones((12, 12))
                data[5:7, 5:7] = 2.0
                data[4, 4] = 9.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(6, 6))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (6, 6)
                assert data.central_pixels == (2.5, 2.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

        class TestEvenToOdd(object):
            def test__trim_4x4_to_3x3__goes_to_4x4_to_keep_symmetry(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(3, 3))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (4, 4)
                assert data.central_pixels == (1.5, 1.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.4, 0.4), 1e-3)

            def test__trim_6x6_to_3x3_goes_to_4x4_to_keep_symmetry(self):
                data = np.ones((6, 6))
                data[2:4, 2:4] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(3, 3))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (4, 4)
                assert data.central_pixels == (1.5, 1.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.4, 0.4), 1e-3)

            def test__trim_12x12_to_5x5__goes_to_6x6_to_keep_symmetry(self):
                data = np.ones((12, 12))
                data[5:7, 5:7] = 2.0
                data[4, 4] = 9.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(5, 5))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

                assert data.dimensions == (6, 6)
                assert data.central_pixels == (2.5, 2.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

        class TestRectangle(object):
            def test__trim_5x4_to_3x2(self):
                data = np.ones((5, 4))
                data[2, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(3, 2))

                assert (data.data == np.array([[1.0, 1.0],
                                               [2.0, 2.0],
                                               [1.0, 1.0]])).all()

                assert data.dimensions == (3, 2)
                assert data.central_pixels == (1, 0.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.3, 0.2), 1e-3)

            def test__trim_4x5_to_2x3(self):
                data = np.ones((4, 5))
                data[1:3, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(2, 3))

                assert (data.data == np.array([[1.0, 2.0, 1.0],
                                               [1.0, 2.0, 1.0]])).all()

                assert data.dimensions == (2, 3)
                assert data.central_pixels == (0.5, 1.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.2, 0.3), 1e-3)

            def test__trim_5x4_to_4x3__goes_to_5x4_to_keep_symmetry(self):
                data = np.ones((5, 4))
                data[2, 1:3] = 2.0
                data[4, 3] = 9.0

                data = image.Data(data, pixel_scale=0.1)
                data.trim_data(new_dimensions=(4, 3))

                assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 2.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0],
                                               [1.0, 1.0, 1.0, 9.0]])).all()

                assert data.dimensions == (5, 4)
                assert data.central_pixels == (2, 1.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.5, 0.4), 1e-3)

        class TestBadInput(object):
            def test__x_size_bigger_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert data.trim_data(new_dimensions=(8, 3))

            def test__y_size_bigger_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert data.trim_data(new_dimensions=(3, 8))

    class TestPadData(object):
        class TestOddToOdd(object):
            def test__pad_3x3_to_5x5(self):
                data = np.ones((3, 3))
                data[1, 1] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(5, 5))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (5, 5)
                assert data.central_pixels == (2.0, 2.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

            def test__pad_5x5_to_9x9(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(9, 9))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (9, 9)
                assert data.central_pixels == (4.0, 4.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.9, 0.9), 1e-3)

        class TestOddToEven(object):
            def test__pad_3x3_to_4x4__goes_to_5x5_to_keep_symmetry(self):
                data = np.ones((3, 3))
                data[1, 1] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(4, 4))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (5, 5)
                assert data.central_pixels == (2.0, 2.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.5, 0.5), 1e-3)

            def test__pad_5x5_to_8x8__goes_to_9x9_to_keep_symmetry(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(8, 8))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (9, 9)
                assert data.central_pixels == (4.0, 4.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.9, 0.9), 1e-3)

        class TestEvenToEven(object):
            def test__pad_4x4_to_6x6(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(6, 6))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (6, 6)
                assert data.central_pixels == (2.5, 2.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

            def test__pad_4x4_to_8x8(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(8, 8))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (8, 8)
                assert data.central_pixels == (3.5, 3.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.8, 0.8), 1e-3)

        class TestEvenToOdd(object):
            def test__pad_4x4_to_5x5__goes_to_6x6_to_keep_symmetry(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(5, 5))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (6, 6)
                assert data.central_pixels == (2.5, 2.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.6, 0.6), 1e-3)

            def test__pad_4x4_to_7x7__goes_to_8x8_to_keep_symmetry(self):
                data = np.ones((4, 4))
                data[1:3, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(7, 7))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (8, 8)
                assert data.central_pixels == (3.5, 3.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.8, 0.8), 1e-3)

        class TestRectangle(object):
            def test__pad_5x4_to_7x6(self):
                data = np.ones((5, 4))
                data[2, 1:3] = 2.0

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(7, 6))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (7, 6)
                assert data.central_pixels == (3, 2.5)
                assert data.dimensions_arc_seconds == pytest.approx((0.7, 0.6), 1e-3)

            def test__pad_2x3_to_6x7(self):
                data = np.ones((2, 3))
                data[0:2, 1] = 2.0
                data[1, 2] = 9

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(6, 7))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (6, 7)
                assert data.central_pixels == (2.5, 3.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.6, 0.7), 1e-3)

            def test__pad_2x3_to_5x6__goes_to_6x7_to_keep_symmetry(self):
                data = np.ones((2, 3))
                data[0:2, 1] = 2.0
                data[1, 2] = 9

                data = image.Data(data, pixel_scale=0.1)
                data.pad_data(new_dimensions=(5, 6))

                assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

                assert data.dimensions == (6, 7)
                assert data.central_pixels == (2.5, 3.0)
                assert data.dimensions_arc_seconds == pytest.approx((0.6, 0.7), 1e-3)

        class TestBadInput(object):
            def test__x_size_smaller_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert data.trim_data(new_dimensions=(3, 8))

            def test__y_size_smaller_than_data__raises_error(self):
                data = np.ones((5, 5))
                data[2, 2] = 2.0

                data = image.Data(data, pixel_scale=0.1)

                with pytest.raises(ValueError):
                    assert data.trim_data(new_dimensions=(8, 3))


class TestImage(object):
    class TestSetup(object):
        def test__init__input_image_3x3__all_attributes_correct(self):
            test_image = image.Image(image=np.ones((3, 3)), pixel_scale=0.1)

            assert (test_image.data == np.ones((3, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.dimensions == (3, 3)
            assert test_image.central_pixels == (1.0, 1.0)
            assert test_image.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__init__input_image_4x3__all_attributes_correct(self):
            test_image = image.Image(image=np.ones((4, 3)), pixel_scale=0.1)

            assert (test_image.data == np.ones((4, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.dimensions == (4, 3)
            assert test_image.central_pixels == (1.5, 1.0)
            assert test_image.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

        def test__from_fits__input_image_3x3__all_attributes_correct(self):
            test_image = image.Image.from_fits('3x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)

            assert (test_image.data == np.ones((3, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.dimensions == (3, 3)
            assert test_image.central_pixels == (1.0, 1.0)
            assert test_image.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__from_fits__input_image_4x3__all_attributes_correct(self):
            test_image = image.Image.from_fits('4x3_ones.fits', hdu=0, pixel_scale=0.1, path=test_data_dir)

            assert (test_image.data == np.ones((4, 3))).all()
            assert test_image.pixel_scale == 0.1
            assert test_image.dimensions == (4, 3)
            assert test_image.central_pixels == (1.5, 1.0)
            assert test_image.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

    class TestSetSky(object):
        def test__via_edges__input_all_ones__sky_bg_level_1(self):
            test_image = image.Image(image=np.ones((3, 3)), pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):
            image_data = np.array([[1, 1, 1],
                                   [1, 100, 1],
                                   [1, 1, 1]])

            test_image = image.Image(image=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):
            image_data = np.array([[1, 1, 1],
                                   [1, 100, 1],
                                   [1, 100, 1],
                                   [1, 1, 1]])

            test_image = image.Image(image=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image_data = np.array([[1, 1, 1, 1],
                                   [1, 100, 100, 1],
                                   [1, 100, 100, 1],
                                   [1, 1, 1, 1]])

            test_image = image.Image(image=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=1)

            assert test_image.sky_background_level == 1.0
            assert test_image.sky_background_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):
            image_data = np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 100, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]])

            test_image = image.Image(image=image_data, pixel_scale=0.1)
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

            test_image = image.Image(image=image_data, pixel_scale=0.1)
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

            test_image = image.Image(image=image_data, pixel_scale=0.1)
            test_image.set_sky_via_edges(no_edges=3)

            assert test_image.sky_background_level == np.mean(np.arange(48))
            assert test_image.sky_background_noise == np.std(np.arange(48))


class TestPSF(object):
    class TestSetup(object):
        def test__init__input_psf_3x3__all_attributes_correct(self):
            test_psf = image.PSF(psf=np.ones((3, 3)), pixel_scale=0.1, renormalize=False)

            assert (test_psf.data == np.ones((3, 3))).all()
            assert test_psf.pixel_scale == 0.1
            assert test_psf.dimensions == (3, 3)
            assert test_psf.central_pixels == (1.0, 1.0)
            assert test_psf.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__init__input_psf_4x3__all_attributes_correct(self):
            test_psf = image.PSF(psf=np.ones((4, 3)), pixel_scale=0.1, renormalize=False)

            assert (test_psf.data == np.ones((4, 3))).all()
            assert test_psf.pixel_scale == 0.1
            assert test_psf.dimensions == (4, 3)
            assert test_psf.central_pixels == (1.5, 1.0)
            assert test_psf.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

        def test__from_fits__input_psf_3x3__all_attributes_correct(self):
            test_psf = image.PSF.from_fits('3x3_ones.fits', hdu=0, pixel_scale=0.1, renormalize=False,
                                           path=test_data_dir)

            assert (test_psf.data == np.ones((3, 3))).all()
            assert test_psf.pixel_scale == 0.1
            assert test_psf.dimensions == (3, 3)
            assert test_psf.central_pixels == (1.0, 1.0)
            assert test_psf.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__from_fits__input_psf_4x3__all_attributes_correct(self):
            test_psf = image.PSF.from_fits('4x3_ones.fits', hdu=0, pixel_scale=0.1, renormalize=False,
                                           path=test_data_dir)

            assert (test_psf.data == np.ones((4, 3))).all()
            assert test_psf.pixel_scale == 0.1
            assert test_psf.dimensions == (4, 3)
            assert test_psf.central_pixels == (1.5, 1.0)
            assert test_psf.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

        def test__input_image_3x3__setup_from_image(self):
            test_image = image.Image(image=np.ones((3, 3)), pixel_scale=0.1)

            test_psf = test_image.load_psf(file_name='3x3_ones.fits', hdu=0, renormalize=False, path=test_data_dir)

            assert (test_psf.data == np.ones((3, 3))).all()
            assert test_psf.pixel_scale == 0.1
            assert test_psf.dimensions == (3, 3)
            assert test_psf.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__input_image_4x3__setup_from_image(self):
            test_image = image.Image(image=np.ones((3, 3)), pixel_scale=0.1)

            test_psf = test_image.load_psf(file_name='4x3_ones.fits', hdu=0, renormalize=False, path=test_data_dir)

            assert (test_psf.data == np.ones((4, 3))).all()
            assert test_psf.pixel_scale == 0.1
            assert test_psf.dimensions == (4, 3)
            assert test_psf.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

    class TestRenormalize(object):
        def test__input_is_already_normalized__no_change(self):
            psf_data = np.ones((3, 3)) / 9.0

            test_psf = image.PSF(psf=psf_data, pixel_scale=0.1, renormalize=False)
            test_psf.renormalize()

            assert test_psf.data == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization__correctly_normalized(self):
            psf_data = np.ones((3, 3))

            test_psf = image.PSF(psf=psf_data, pixel_scale=0.1, renormalize=False)
            test_psf.renormalize()

            assert test_psf.data == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__input_is_below_normalization__correctly_normalized(self):
            psf_data = np.ones((3, 3)) / 90.0

            test_psf = image.PSF(psf=psf_data, pixel_scale=0.1)
            test_psf.renormalize()

            assert test_psf.data == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__input_different_psf__correctly_normalized(self):
            psf_data = np.ones((4, 4))
            psf_data[1:3, 1:3] = 2.0

            normalization_factor = 2.0 * 4.0 + 12

            test_psf = image.PSF(psf=psf_data, pixel_scale=0.1, renormalize=False)
            test_psf.renormalize()

            assert test_psf.data == pytest.approx(psf_data / normalization_factor, 1e-3)

        def test__works_in_class_constructor(self):
            psf_data = np.ones((4, 4))
            psf_data[1:3, 1:3] = 2.0

            normalization_factor = 2.0 * 4.0 + 12

            test_psf = image.PSF(psf=psf_data, pixel_scale=0.1)

            assert test_psf.data == pytest.approx(psf_data / normalization_factor, 1e-3)


class TestNoise(object):
    class TestSetup(object):
        def test__init__input_noise_3x3__all_attributes_correct(self):
            test_noise = image.Noise(noise=np.ones((3, 3)), pixel_scale=0.1)

            assert (test_noise.data == np.ones((3, 3))).all()
            assert test_noise.pixel_scale == 0.1
            assert test_noise.dimensions == (3, 3)
            assert test_noise.central_pixels == (1.0, 1.0)
            assert test_noise.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__init__input_noise_4x3__all_attributes_correct(self):
            test_noise = image.Noise(noise=np.ones((4, 3)), pixel_scale=0.1)

            assert (test_noise.data == np.ones((4, 3))).all()
            assert test_noise.pixel_scale == 0.1
            assert test_noise.dimensions == (4, 3)
            assert test_noise.central_pixels == (1.5, 1.0)
            assert test_noise.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

        def test__from_fits__input_noise_3x3__all_attributes_correct(self):
            test_noise = image.Noise.from_fits('3x3_ones.fits', hdu=0, pixel_scale=0.1,
                                               path=test_data_dir)

            assert (test_noise.data == np.ones((3, 3))).all()
            assert test_noise.pixel_scale == 0.1
            assert test_noise.dimensions == (3, 3)
            assert test_noise.central_pixels == (1.0, 1.0)
            assert test_noise.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__from_fits__input_noise_4x3__all_attributes_correct(self):
            test_noise = image.Noise.from_fits('4x3_ones.fits', hdu=0, pixel_scale=0.1,
                                               path=test_data_dir)

            assert (test_noise.data == np.ones((4, 3))).all()
            assert test_noise.pixel_scale == 0.1
            assert test_noise.dimensions == (4, 3)
            assert test_noise.central_pixels == (1.5, 1.0)
            assert test_noise.dimensions_arc_seconds == pytest.approx((0.4, 0.3))

        def test__input_image_3x3__setup_from_image(self):
            test_image = image.Image(image=np.ones((3, 3)), pixel_scale=0.1)

            test_noise = test_image.load_noise(file_name='3x3_ones.fits', hdu=0,
                                               path=test_data_dir)

            assert (test_noise.data == np.ones((3, 3))).all()
            assert test_noise.pixel_scale == 0.1
            assert test_noise.dimensions == (3, 3)
            assert test_noise.dimensions_arc_seconds == pytest.approx((0.3, 0.3))

        def test__input_image_4x3__setup_from_image(self):
            test_image = image.Image(image=np.ones((3, 3)), pixel_scale=0.1)

            test_noise = test_image.load_noise(file_name='4x3_ones.fits', hdu=0,
                                               path=test_data_dir)

            assert (test_noise.data == np.ones((4, 3))).all()
            assert test_noise.pixel_scale == 0.1
            assert test_noise.dimensions == (4, 3)
            assert test_noise.dimensions_arc_seconds == pytest.approx((0.4, 0.3))


class TestMask(object):
    class TestPixelScale(object):
        def test_central_pixel(self):
            assert image.Mask.central_pixel((3, 3), 0.5) == (2.5, 2.5)

        def test__shape(self):
            assert image.Mask.circular(dimensions=(3, 3), pixel_scale=1, radius=5).shape == (3, 3)
            assert image.Mask.circular(dimensions=(3, 3), pixel_scale=0.5, radius=5).shape == (6, 6)
            assert image.Mask.circular(dimensions=(3, 3), pixel_scale=0.2, radius=5).shape == (15, 15)

        def test__odd_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=0.5, radius=0.5)
            print(mask)
            assert (mask == np.array([[False, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, True, True, False, False],
                                      [False, False, True, True, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False]])).all()

    class TestCircular(object):
        def test__input_big_mask__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=1, radius=5)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__odd_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=1, radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=1, radius=1)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__odd_x_odd_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(dimensions=(3, 3), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 3), pixel_scale=1, radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__even_x_odd_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 3), pixel_scale=1, radius=1.50001)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 3), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_even_mask_input_radius_small__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 4), pixel_scale=1, radius=0.72)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, False],
                                      [False, True, True, False],
                                      [False, False, False, False]])).all()

        def test__even_x_even_mask_input_radius_medium__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 4), pixel_scale=1, radius=1.7)

            assert (mask == np.array([[False, True, True, False],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_input_radius_large__correct_mask(self):
            mask = image.Mask.circular(dimensions=(4, 4), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True]])).all()

    class TestAnnulus(object):
        def test__odd_x_odd_mask_inner_radius_zero_outer_radius_small__correct_mask(self):
            mask = image.Mask.annular(dimensions=(3, 3), pixel_scale=1, inner_radius=0, outer_radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_inner_radius_small_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(dimensions=(3, 3), pixel_scale=1, inner_radius=0.5, outer_radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 3), pixel_scale=1, inner_radius=0.51, outer_radius=1.51)

            assert (mask == np.array([[False, True, False],
                                      [True, False, True],
                                      [True, False, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 3), pixel_scale=1, inner_radius=1.51, outer_radius=3)

            assert (mask == np.array([[True, False, True],
                                      [False, False, False],
                                      [False, False, False],
                                      [True, False, True]])).all()

        def test__even_x_even_mask_inner_radius_small_outer_radius_medium__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 4), pixel_scale=1, inner_radius=0.81, outer_radius=2)

            assert (mask == np.array([[False, True, True, False],
                                      [True, False, False, True],
                                      [True, False, False, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_inner_radius_medium_outer_radius_large__correct_mask(self):
            mask = image.Mask.annular(dimensions=(4, 4), pixel_scale=1, inner_radius=1.71, outer_radius=3)

            assert (mask == np.array([[True, False, False, True],
                                      [False, False, False, False],
                                      [False, False, False, False],
                                      [True, False, False, True]])).all()
