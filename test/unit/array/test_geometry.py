
import autolens as al
from autolens import exc

import numpy as np
import pytest


class TestPixelScale:

    def test__zero_or_negative_pixel_scale__raises_exception(self):
        with pytest.raises(exc.GeometryException):
            al.Geometry(shape=(2,2), pixel_scales=(0.0, 0.0), sub_size=1)

        with pytest.raises(exc.GeometryException):
            al.Geometry(shape=(2,2), pixel_scales=(-0.5, 0.0), sub_size=1)


class TestCentralPixel:

    def test__depends_on_shape_pixel_scale_and_origin(self):
        geometry = al.Geometry(shape=(3,3), pixel_scales=(0.1, 0.1), sub_size=1)
        assert geometry.central_pixel_coordinates == (1, 1)

        geometry = al.Geometry(shape=(4,4), pixel_scales=(0.1, 0.1), sub_size=1)
        assert geometry.central_pixel_coordinates == (1.5, 1.5)

        geometry = al.Geometry(
            shape=(5,3), pixel_scales=(0.1, 0.1), sub_size=1, origin=(1.0, 2.0)
        )
        assert geometry.central_pixel_coordinates == (2.0, 1.0)

        geometry = al.Geometry(
            shape=(3,3), pixel_scales=(2.0, 1.0), sub_size=1
        )
        assert geometry.central_pixel_coordinates == (1, 1)

        geometry = al.Geometry(
            shape=(4,4), pixel_scales=(2.0, 1.0), sub_size=1,
        )
        assert geometry.central_pixel_coordinates == (1.5, 1.5)

        geometry = al.Geometry(
            shape=(5,3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(1.0, 2.0)
        )
        assert geometry.central_pixel_coordinates == (2, 1)


class TestGrids:

    def test__square_pixel_grid__grid_2d__compare_to_array_util(self):
        
        grid_2d_util = al.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
        )

        geometry = al.Geometry(shape=(4,7), pixel_scales=(0.56, 0.56), sub_size=1)

        assert geometry.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        geometry = al.Geometry(shape=(3,3), pixel_scales=(1.0, 1.0), sub_size=1)

        assert (
            geometry.grid_2d
            == np.array(
                [
                    [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
                ]
            )
        ).all()

    def test__square_pixel_grid__grid_1d__compare_to_array_util(self):

        grid_1d_util = al.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
        )

        geometry = al.Geometry(shape=(4,7), pixel_scales=(0.56, 0.56), sub_size=1)

        assert geometry.grid_1d == pytest.approx(grid_1d_util, 1e-4)

    def test__square_pixel_grid__nonzero_centres__compure_to_array_util(self):
        grid_2d_util = al.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(1.0, 3.0)
        )

        geometry = al.Geometry(
            shape=(4,7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(1.0, 3.0)
        )

        assert geometry.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        grid_1d_util = al.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(-1.0, -4.0)
        )

        geometry = al.Geometry(
            shape=(4,7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(-1.0, -4.0)
        )

        assert geometry.grid_1d == pytest.approx(grid_1d_util, 1e-4)

    def test__rectangular_pixel_grid__grid_2d__compare_to_array_util(self):
        grid_2d_util = al.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
        )

        geometry = al.Geometry(
            shape=(4,7), sub_size=1, pixel_scales=(0.8, 0.56)
        )

        assert geometry.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        geometry = al.Geometry(
            shape=(3,3), sub_size=1, pixel_scales=(1.0, 2.0)
        )

        assert (
            geometry.grid_2d
            == np.array(
                [
                    [[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]],
                    [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
                    [[-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]],
                ]
            )
        ).all()

    def test__rectangular_pixel_grid__grid_1d__compare_to_array_util(self):
        grid_1d_util = al.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
        )

        geometry = al.Geometry(
            shape=(4,7), sub_size=1, pixel_scales=(0.8, 0.56)
        )

        assert geometry.grid_1d == pytest.approx(grid_1d_util, 1e-4)

    def test__rectangular_pixel_grid__nonzero_centres__compure_to_array_util(self):
        grid_2d_util = al.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1, origin=(1.0, 2.0)
        )

        geometry = al.Geometry(
            shape=(4,7), sub_size=1, pixel_scales=(0.8, 0.56), origin=(1.0, 2.0)
        )

        assert geometry.grid_2d == pytest.approx(grid_2d_util, 1e-4)

        grid_1d_util = al.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1, origin=(-1.0, -4.0)
        )

        geometry = al.Geometry(
            shape=(4,7), pixel_scales=(0.8, 0.56), sub_size=1, origin=(-1.0, -4.0)
        )

        assert geometry.grid_1d == pytest.approx(grid_1d_util, 1e-4)


class TestConversion:
    def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_centres(
        self
    ):
        geometry = al.Geometry(shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.0, -1.0)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.0, 1.0)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-1.0, -1.0)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-1.0, 1.0)
        ) == (1, 1)

        geometry = al.Geometry(shape=(3, 3), pixel_scales=(3.0, 3.0), sub_size=1)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.0, -3.0)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.0, 0.0)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.0, 3.0)
        ) == (0, 2)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, -3.0)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 0.0)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 3.0)
        ) == (1, 2)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-3.0, -3.0)
        ) == (2, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-3.0, 0.0)
        ) == (2, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-3.0, 3.0)
        ) == (2, 2)

    def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_corners(
        self
    ):
        geometry = al.Geometry(shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.99, -1.99)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.99, -0.01)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.01, -1.99)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.01, -0.01)
        ) == (0, 0)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(2.01, 0.01)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(2.01, 1.99)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.01, 0.01)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.01, 1.99)
        ) == (0, 1)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.01, -1.99)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.01, -0.01)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-1.99, -1.99)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-1.99, -0.01)
        ) == (1, 0)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.01, 0.01)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.01, 1.99)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-1.99, 0.01)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-1.99, 1.99)
        ) == (1, 1)

    def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_centres__nonzero_centre(
        self
    ):
        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 1.0), sub_size=1
        )

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(2.0, 0.0)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(2.0, 2.0)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 0.0)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 2.0)
        ) == (1, 1)

        geometry = al.Geometry(
            shape=(3, 3), pixel_scales=(3.0, 3.0), sub_size=1, origin=(3.0, 3.0)
        )

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(6.0, 0.0)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(6.0, 3.0)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(6.0, 6.0)
        ) == (0, 2)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.0, 0.0)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.0, 3.0)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.0, 6.0)
        ) == (1, 2)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 0.0)
        ) == (2, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 3.0)
        ) == (2, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.0, 6.0)
        ) == (2, 2)

    def test__arc_second_coordinates_to_pixel_coordinates__arcsec_are_pixel_corners__nonzero_centre(
        self
    ):
        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1, origin=(1.0, 1.0)
        )

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(2.99, -0.99)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(2.99, 0.99)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.01, -0.99)
        ) == (0, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.01, 0.99)
        ) == (0, 0)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.01, 1.01)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(3.01, 2.99)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.01, 1.01)
        ) == (0, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(1.01, 2.99)
        ) == (0, 1)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.99, -0.99)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.99, 0.99)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.99, -0.99)
        ) == (1, 0)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.99, 0.99)
        ) == (1, 0)

        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.99, 1.01)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(0.99, 2.99)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.99, 1.01)
        ) == (1, 1)
        assert geometry.arc_second_coordinates_to_pixel_coordinates(
            arc_second_coordinates=(-0.99, 2.99)
        ) == (1, 1)

    def test__square_pixel_grid__1d_arc_second_grid_to_1d_pixel_centred_grid__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        geometry = al.Geometry(shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1)

        grid_pixels = geometry.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)

        assert (grid_pixels == grid_pixels_util).all()

    def test__square_pixel_grid__1d_arc_second_grid_to_1d_pixel_indexes_grid__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])

        grid_pixel_indexes_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        geometry = al.Geometry(shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1)

        grid_pixel_indexes = geometry.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec=grid_arcsec
        )

        assert (grid_pixel_indexes == grid_pixel_indexes_util).all()

    def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_centred_grid__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(7.0, 2.0),
        )

        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(7.0, 2.0), sub_size=1
        )

        grid_pixels = geometry.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)

        assert (grid_pixels == grid_pixels_util).all()

    def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_indexes_grid__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 4.0)
        )

        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(2.0, 4.0), sub_size=1,
        )

        grid_pixels = geometry.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec=grid_arcsec
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__rectangular_pixel_grid__1d_arc_second_grid_to_1d_pixel_grid__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixels_1d(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 4.0)
        )

        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(2.0, 4.0), sub_size=1
        )

        grid_pixels = geometry.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)

        assert (grid_pixels == grid_pixels_util).all()

    def test__square_pixel_grid__1d_pixel_grid_to_1d_pixel_centred_grid__same_as_grid_util(
        self
    ):
        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_pixels_util = al.grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=grid_pixels, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        geometry = al.Geometry(shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1)

        grid_pixels = geometry.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

        assert (grid_pixels == grid_pixels_util).all()

    def test__square_pixel_grid__1d_pixel_grid_to_1d_pixel_grid__same_as_grid_util(
        self
    ):
        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_pixels_util = al.grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=grid_pixels, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        geometry = al.Geometry(shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1)

        grid_pixels = geometry.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

        assert (grid_pixels == grid_pixels_util).all()

    def test__square_pixel_grid__grids_with_nonzero_centres__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1, origin=(1.0, 2.0)
        )

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixels_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = geometry.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = geometry.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec=grid_arcsec
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = geometry.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_arcsec_util = al.grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=grid_pixels,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )

        grid_arcsec = geometry.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

        assert (grid_arcsec == grid_arcsec_util).all()

    def test__rectangular_pixel_grid__grids_with_nonzero_centres__same_as_grid_util(
        self
    ):
        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        geometry = al.Geometry(
            shape=(2, 2), pixel_scales=(2.0, 1.0), sub_size=1, origin=(1.0, 2.0)
        )

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixels_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = geometry.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = geometry.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec=grid_arcsec
        )
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = geometry.grid_arcsec_to_grid_pixel_centres(grid_arcsec=grid_arcsec)
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_arcsec_util = al.grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=grid_pixels,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )

        grid_arcsec = geometry.grid_pixels_to_grid_arcsec(grid_pixels=grid_pixels)

        assert (grid_arcsec == grid_arcsec_util).all()
        

class TestTicks:
    def test__square_pixel_grid__yticks(self):
        geometry = al.Geometry(shape=(3,3), pixel_scales=(1.0, 1.0), sub_size=1)
        assert geometry.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        geometry = al.Geometry(shape=(3,3), pixel_scales=(0.5, 0.5), sub_size=1)
        assert geometry.yticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        geometry = al.Geometry(shape=(6,3), pixel_scales=(1.0, 1.0), sub_size=1)
        assert geometry.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

        geometry = al.Geometry(shape=(3,1), pixel_scales=(1.0, 1.0), sub_size=1)
        assert geometry.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

    def test__square_pixel_grid__xticks(self):
        geometry = al.Geometry(shape=(3,3), pixel_scales=(1.0, 1.0), sub_size=1)
        assert geometry.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        geometry = al.Geometry(shape=(3,3), pixel_scales=(0.5, 0.5), sub_size=1)
        assert geometry.xticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        geometry = al.Geometry(shape=(3,6), pixel_scales=(1.0, 1.0), sub_size=1)
        assert geometry.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

        geometry = al.Geometry(shape=(1,3), pixel_scales=(1.0, 1.0), sub_size=1)
        assert geometry.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

    def test__rectangular_pixel_grid__yticks(self):
        geometry = al.Geometry(
            shape=(3,3), pixel_scales=(1.0, 5.0), sub_size=1
        )
        assert geometry.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        geometry = al.Geometry(
            shape=(3,3), pixel_scales=(0.5, 5.0), sub_size=1
        )
        assert geometry.yticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        geometry = al.Geometry(
            shape=(6,3), pixel_scales=(1.0, 5.0), sub_size=1
        )
        assert geometry.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

        geometry = al.Geometry(
            shape=(3,6), pixel_scales=(1.0, 5.0), sub_size=1
        )
        assert geometry.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

    def test__rectangular_pixel_grid__xticks(self):
        geometry = al.Geometry(
            shape=(3,3), pixel_scales=(5.0, 1.0), sub_size=1
        )
        assert geometry.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        geometry = al.Geometry(
            shape=(3,3), pixel_scales=(5.0, 0.5), sub_size=1
        )
        assert geometry.xticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        geometry = al.Geometry(
            shape=(3,6), pixel_scales=(5.0, 1.0), sub_size=1
        )
        assert geometry.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

        geometry = al.Geometry(
            shape=(6,3), pixel_scales=(5.0, 1.0), sub_size=1
        )
        assert geometry.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)