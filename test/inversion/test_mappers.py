import numpy as np
import pytest

from autolens.imaging.util import mapping_util
from autolens.inversion import mappers
from autolens.inversion import pixelizations
from test.mock.mock_imaging import MockSubGrid, MockGridCollection


class MockGeometry(object):

    def __init__(self):

        self.pixel_scales = (1.0, 1.0)
        self.origin = (0.0, 0.0)


def grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers):
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two image_grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2

    image_pixels = grid.shape[0]

    image_to_pixelization = np.zeros((image_pixels,))

    for image_index, image_coordinate in enumerate(grid):
        distances = list(map(lambda centers: compute_squared_separation(image_coordinate, centers), pixel_centers))

        image_to_pixelization[image_index] = (np.argmin(distances))

    return image_to_pixelization


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


class TestMappingMatrix:

    def test__3_image_pixels__6_pixel_pixels__sub_grid_1x1(self, three_pixels):
        sub_to_pixelization = np.array([0, 1, 2])
        sub_to_image = np.array([0, 1, 2])

        grids = MockGridCollection(image=three_pixels, sub=MockSubGrid(three_pixels, sub_to_image,
                                                                       sub_grid_size=1))

        pix = mappers.Mapper(pixels=6, grids=grids, border=None, pixel_neighbors=None)

        mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pixelization, pix.pixels,
                                                                pix.grids.image.shape[0], pix.grids.sub.sub_to_image,
                                                                pix.grids.sub.sub_grid_fraction)

        assert (mapping_matrix == np.array([[1, 0, 0, 0, 0, 0],  # Image pixel 0 maps to pix pixel 0.
                                            [0, 1, 0, 0, 0, 0],  # Image pixel 1 maps to pix pixel 1.
                                            [0, 0, 1, 0, 0, 0]])).all()  # Image pixel 2 maps to pix pixel 2

    def test__5_image_pixels__8_pixel_pixels__sub_grid_1x1(self, five_pixels):
        sub_to_pixelization = np.array([0, 1, 2, 7, 6])
        sub_to_image = np.array([0, 1, 2, 3, 4])

        grids = MockGridCollection(image=five_pixels, sub=MockSubGrid(five_pixels, sub_to_image,
                                                                      sub_grid_size=1))

        pix = mappers.Mapper(pixels=8, grids=grids, border=None, pixel_neighbors=None)

        mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pixelization, pix.pixels,
                                                                pix.grids.image.shape[0], pix.grids.sub.sub_to_image,
                                                                pix.grids.sub.sub_grid_fraction)

        assert (mapping_matrix == np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],  # Image image_to_pixel 0 and 3 mappers to pix pixel 0.
             [0, 1, 0, 0, 0, 0, 0, 0],  # Image image_to_pixel 1 and 4 mappers to pix pixel 1.
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0]])).all()  # Image image_to_pixel 2 and 5 mappers to pix pixel 2

    def test__5_image_pixels__8_pixel_pixels__sub_grid_2x2__no_overlapping_pixels(self, five_pixels):
        sub_to_pixelization = np.array([0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2])
        sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

        grids = MockGridCollection(image=five_pixels, sub=MockSubGrid(five_pixels, sub_to_image,
                                                                      sub_grid_size=2))

        pix = mappers.Mapper(pixels=8, grids=grids, border=None, pixel_neighbors=None)

        mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pixelization, pix.pixels,
                                                                pix.grids.image.shape[0], pix.grids.sub.sub_to_image,
                                                                pix.grids.sub.sub_grid_fraction)

        assert (mapping_matrix == np.array(
            [[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
             [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0],
             [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
             [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
             [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25]])).all()

    def test__5_image_pixels__8_pixel_pixels__sub_grid_2x2__include_overlapping_pixels(self, five_pixels):
        sub_to_pixelization = np.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2])
        sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

        grids = MockGridCollection(image=five_pixels, sub=MockSubGrid(five_pixels, sub_to_image,
                                                                      sub_grid_size=2))

        pix = mappers.Mapper(pixels=8, grids=grids, border=None, pixel_neighbors=None)

        mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pixelization, pix.pixels,
                                                                pix.grids.image.shape[0], pix.grids.sub.sub_to_image,
                                                                pix.grids.sub.sub_grid_fraction)

        assert (mapping_matrix == np.array(
            [[0.75, 0.25, 0, 0, 0, 0, 0, 0],
             [0.5, 0.5, 0, 0, 0, 0, 0, 0],
             [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
             [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
             [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25]])).all()

    def test__3_image_pixels__6_pixel_pixels__sub_grid_4x4(self, three_pixels):
        sub_to_pixelization = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3])

        sub_to_image = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        grids = MockGridCollection(image=three_pixels, sub=MockSubGrid(three_pixels, sub_to_image,
                                                                       sub_grid_size=4))

        pix = mappers.Mapper(pixels=6, grids=grids, border=None, pixel_neighbors=None)

        mapping_matrix = pix.mapping_matrix_from_sub_to_pix_jit(sub_to_pixelization, pix.pixels,
                                                                pix.grids.image.shape[0], pix.grids.sub.sub_to_image,
                                                                pix.grids.sub.sub_grid_fraction)

        assert (mapping_matrix == np.array(
            [[0.75, 0.25, 0, 0, 0, 0],
             [0, 0, 1.0, 0, 0, 0],
             [0.1875, 0.1875, 0.1875, 0.1875, 0.125, 0.125]])).all()


class TestRectangularMapper:

    class TestImageAndSubToPixelization:

        def test__3x3_grid_of_pixel_grid__1_coordinate_per_square_pixel__in_centre_of_pixels(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pixelization_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, geometry=geometry, border=None,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert pix.pix_to_image == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
            assert pix.pix_to_sub == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

        def test__3x3_grid_of_pixel_grid__1_coordinate_per_square_pixel__near_edges_of_pixels(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pixelization_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                          [-0.32, -1.0], [-0.32, 0.32], [0.0, 1.0],
                                          [-0.34, -0.34], [-0.34, 0.325], [-1.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert pix.pix_to_image == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
            assert pix.pix_to_sub == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

        def test__3x3_grid_of_pixel_grid__add_multiple_grid_to_1_pixel_pixel(self):
            #                  _ _ _
            # -1.0 to -(1/3)  |_|_|_|
            # -(1/3) to (1/3) |_|_|_|
            #  (1/3) to 1.0   |_|_|_|

            pixelization_grid = np.array([[1.0, -1.0], [0.0, 0.0], [1.0, 1.0],
                                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                          [-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])).all()
            assert pix.pix_to_image == [[0], [], [2], [], [1, 3, 4, 5, 7], [], [6], [], [8]]
            assert pix.pix_to_sub == [[0], [], [2], [], [1, 3, 4, 5, 7], [], [6], [], [8]]

        def test__4x3_grid_of_pixel_grid__1_coordinate_in_each_pixel(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            # Boundaries for column pixel 0 -1.0 to -(1/3)
            # Boundaries for column pixel 1 -(1/3) to (1/3)
            # Boundaries for column pixel 2  (1/3) to 1.0

            # Bounadries for row pixel 0 -1.0 to -0.5
            # Bounadries for row pixel 1 -0.5 to 0.0
            # Bounadries for row pixel 2  0.0 to 0.5
            # Bounadries for row pixel 3  0.5 to 1.0

            pixelization_grid = np.array([[ 1.0, -1.0], [ 1.0,  0.0],
                                          [ 1.0, 1.0], [ 0.5, -1.0],
                                          [-0.5, 1.0], [-1.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(4, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=12, shape=(4, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 2, 3, 8, 11])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 3, 8, 11])).all()
            assert pix.pix_to_image == [[0], [1], [2], [3], [], [], [], [], [4], [], [], [5]]
            assert pix.pix_to_sub == [[0], [1], [2], [3], [], [], [], [], [4], [], [], [5]]

        def test__3x4_grid_of_pixel_grid__1_coordinate_in_each_pixel(self):
            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            # Boundaries for row pixel 0 -1.0 to -(1/3)
            # Boundaries for row pixel 1 -(1/3) to (1/3)
            # Boundaries for row pixel 2  (1/3) to 1.0

            # Bounadries for column pixel 0 -1.0 to -0.5
            # Bounadries for column pixel 1 -0.5 to 0.0
            # Bounadries for column pixel 2  0.0 to 0.5
            # Bounadries for column pixel 3  0.5 to 1.0

            pixelization_grid = np.array([[1.0, -1.0], [1.0, -0.49], [1.0, 0.01], [0.32, 0.01],
                                          [-0.34, -0.01], [-1.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(3, 4))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=12, shape=(3, 4), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 2, 6, 9, 11])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 6, 9, 11])).all()
            assert pix.pix_to_image == [[0], [1], [2], [], [], [], [3], [], [], [4], [], [5]]
            assert pix.pix_to_sub == [[0], [1], [2], [], [], [], [3], [], [], [4], [], [5]]

        def test__3x3_grid__change_arcsecond_dimensions_size__grid_adapts_accordingly(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            pixelization_grid = np.array([[1.5, -1.5], [1.0, 0.0], [1.0, 0.6], [-1.4, 0.0], [-1.5, 1.5]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 2, 7, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 7, 8])).all()
            assert pix.pix_to_image == [[0], [1], [2], [], [], [], [], [3], [4]]
            assert pix.pix_to_sub == [[0], [1], [2], [], [], [], [], [3], [4]]

        def test__3x3_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            pixelization_grid = np.array([[1.0, -1.5], [1.0, -0.49], [0.32, -1.5], [0.32, 0.51], [-1.0, 1.5]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 3, 5, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 3, 5, 8])).all()
            assert pix.pix_to_image == [[0], [1], [], [2], [], [3], [], [], [4]]
            assert pix.pix_to_sub == [[0], [1], [], [2], [], [3], [], [], [4]]

        def test__4x3_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            pixelization_grid = np.array([[1.0, -1.5], [1.0, -0.49], [0.49, -1.5], [-0.6, 0.0], [-1.0, 1.5]])

            pix = pixelizations.Rectangular(shape=(4, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=12, shape=(4, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 3, 10, 11])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 3, 10, 11])).all()
            assert pix.pix_to_image == [[0], [1], [], [2], [], [], [], [], [], [], [3], [4]]
            assert pix.pix_to_sub == [[0], [1], [], [2], [], [], [], [], [], [], [3], [4]]

        def test__3x4_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            pixelization_grid = np.array([[1.0, -1.5], [1.0, -0.49], [0.32, -1.5], [-0.34, 0.49], [-1.0, 1.5]])

            pix = pixelizations.Rectangular(shape=(3, 4))

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            pix = mappers.RectangularMapper(pixels=12, shape=(3, 4), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 4, 10, 11])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 4, 10, 11])).all()
            assert pix.pix_to_image == [[0], [1], [], [], [2], [], [], [], [], [], [3], [4]]
            assert pix.pix_to_sub == [[0], [1], [], [], [2], [], [], [], [], [], [3], [4]]

        def test__different_image_and_sub_grids(self):
            #                  _ _ _
            # -1.0 to -(1/3)  |_|_|_|
            # -(1/3) to (1/3) |_|_|_|
            #  (1/3) to 1.0   |_|_|_|

            pixelization_grid = np.array([[1.0, -1.0], [0.0, 0.0], [1.0, 1.0],
                                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                          [-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])

            pixelization_sub_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                              [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                              [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_sub_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert pix.pix_to_image == [[0], [], [2], [], [1, 3, 4, 5, 7], [], [6], [], [8]]
            assert pix.pix_to_sub == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

        def test__3x3_grid_of_pixel_grid___shift_coordinates_to_new_centre__centre_adjusts_based_on_grid(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pixelization_grid = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                                          [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                                          [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
            assert pix.pix_to_image == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
            assert pix.pix_to_sub == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

        def test__4x3_grid__non_symmetric_centre_shift(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            pixelization_grid = np.array([[3.0, -0.5], [3.0, 0.51], [2.49, -0.5], [1.4, 1.0], [1.0, 2.5]])

            pix = pixelizations.Rectangular(shape=(4, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGrid(pixelization_grid, sub_to_image=np.ones((1)),
                                                       sub_grid_size=1))

            pix = mappers.RectangularMapper(pixels=12, shape=(4, 3), grids=grids, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            assert (pix.image_to_pix == np.array([0, 1, 3, 10, 11])).all()
            assert (pix.sub_to_pix == np.array([0, 1, 3, 10, 11])).all()
            assert pix.pix_to_image == [[0], [1], [], [2], [], [], [], [], [], [], [3], [4]]
            assert pix.pix_to_sub == [[0], [1], [], [2], [], [], [], [], [], [], [3], [4]]

    class TestReconstructedPixelization:

        def test__3x3_pixelization__solution_vector_ascending(self):

            pixelization_grid = np.array([[2.0, -1.0], [2.0, 0.0], [2.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0]])

            pix = pixelizations.Rectangular(shape=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 3), grids=None, border=None, geometry=geometry,
                                            pixel_neighbors=None)

            recon_pix = pix.reconstructed_pixelization_from_solution_vector(solution_vector=np.array([1.0, 2.0, 3.0,
                                                                                                      4.0, 5.0, 6.0,
                                                                                                      7.0, 8.0, 9.0]))

            assert (recon_pix == np.array([[1.0, 2.0, 3.0],
                                           [4.0, 5.0, 6.0],
                                           [7.0, 8.0, 9.0]])).all()
            assert recon_pix.pixel_scales == pytest.approx((4./3., 2./3.), 1e-2)
            assert recon_pix.origin == (0.0, 0.0)

        def test__compare_to_imaging_util(self):

            pix = mappers.RectangularMapper(pixels=9, shape=(4, 3), grids=None, border=None, geometry=MockGeometry(),
                                            pixel_neighbors=None)
            solution = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0])
            recon_pix = pix.reconstructed_pixelization_from_solution_vector(solution_vector=solution)
            recon_pix_util = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=solution,
                                                                                                 shape=(4,3))
            assert (recon_pix == recon_pix_util).all()
            assert recon_pix.shape == (4,3)

            pix = mappers.RectangularMapper(pixels=9, shape=(3, 4), grids=None, border=None, geometry=MockGeometry(),
                                            pixel_neighbors=None)
            solution = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0])
            recon_pix = pix.reconstructed_pixelization_from_solution_vector(solution_vector=solution)
            recon_pix_util = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=solution,
                                                                                                 shape=(3,4))
            assert (recon_pix == recon_pix_util).all()
            assert recon_pix.shape == (3,4)


class TestVoronoiPixMapper:

    class TestImageToPixelizationViaNearestNeighborsForTesting:

        def test__grid_to_pixel_pixels_via_nearest_neighbour__case1__correct_pairs(self):
            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            image_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

            image_to_pixelization = grid_to_pixel_pixels_via_nearest_neighbour(image_grid, pixel_centers)

            assert image_to_pixelization[0] == 0
            assert image_to_pixelization[1] == 1
            assert image_to_pixelization[2] == 2
            assert image_to_pixelization[3] == 3

        def test__grid_to_pixel_pixels_via_nearest_neighbour___case2__correct_pairs(self):
            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            image_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1],
                                   [0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])

            image_to_pixelization = grid_to_pixel_pixels_via_nearest_neighbour(image_grid, pixel_centers)

            assert image_to_pixelization[0] == 0
            assert image_to_pixelization[1] == 1
            assert image_to_pixelization[2] == 2
            assert image_to_pixelization[3] == 3
            assert image_to_pixelization[4] == 3
            assert image_to_pixelization[5] == 2
            assert image_to_pixelization[6] == 1
            assert image_to_pixelization[7] == 0

        def test__grid_to_pixel_pixels_via_nearest_neighbour___case3__correct_pairs(self):
            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
            image_grid = np.array([[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49],
                                   [0.51, 0.51], [1.01, 1.01], [1.51, 1.51]])

            image_to_pixelization = grid_to_pixel_pixels_via_nearest_neighbour(image_grid, pixel_centers)

            assert image_to_pixelization[0] == 4
            assert image_to_pixelization[1] == 4
            assert image_to_pixelization[2] == 4
            assert image_to_pixelization[3] == 0
            assert image_to_pixelization[4] == 0
            assert image_to_pixelization[5] == 5

    class TestSubToPixelizationViaNearestNeighborsForTesting:

        def test__grid_to_pixel_pixels_via_nearest_neighbour__case1__correct_pairs(self):
            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            sub_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

            sub_to_pixelization = grid_to_pixel_pixels_via_nearest_neighbour(sub_grid, pixel_centers)

            assert sub_to_pixelization[0] == 0
            assert sub_to_pixelization[1] == 1
            assert sub_to_pixelization[2] == 2
            assert sub_to_pixelization[3] == 3

        def test__grid_to_pixel_pixels_via_nearest_neighbour___case2__correct_pairs(self):
            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            sub_grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1],
                                 [0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])

            sub_to_pixelization = grid_to_pixel_pixels_via_nearest_neighbour(sub_grid, pixel_centers)

            assert sub_to_pixelization[0] == 0
            assert sub_to_pixelization[1] == 1
            assert sub_to_pixelization[2] == 2
            assert sub_to_pixelization[3] == 3
            assert sub_to_pixelization[4] == 3
            assert sub_to_pixelization[5] == 2
            assert sub_to_pixelization[6] == 1
            assert sub_to_pixelization[7] == 0

        def test__grid_to_pixel_pixels_via_nearest_neighbour___case3__correct_pairs(self):
            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
            sub_grid = np.array([[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49],
                                 [0.51, 0.51], [1.01, 1.01], [1.51, 1.51]])

            sub_to_pixelization = grid_to_pixel_pixels_via_nearest_neighbour(sub_grid, pixel_centers)

            assert sub_to_pixelization[0] == 4
            assert sub_to_pixelization[1] == 4
            assert sub_to_pixelization[2] == 4
            assert sub_to_pixelization[3] == 0
            assert sub_to_pixelization[4] == 0
            assert sub_to_pixelization[5] == 5

    class TestImageToPixelization:

        def test__image_to_pixelization_of_mapper_matches_nearest_neighbor_calculation(self):
            pixel_centers = np.array([[-1.0, -1.0], [-0.9, 0.9],
                                      [1.0, -1.1], [1.2, 1.2]])

            pixelization_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            image_to_pixelization_nearest_neighbour = grid_to_pixel_pixels_via_nearest_neighbour(pixelization_grid,
                                                                                                 pixel_centers)

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            grids = MockGridCollection(image=pixelization_grid, sub=MockSubGrid(np.array([]), sub_to_image,
                                                                                sub_grid_size=1))

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(pixel_centers)
            pixel_neighbors = pix.neighbors_from_pixelization(pixels=6, ridge_points=voronoi.ridge_points)
            image_to_nearest_image_pix = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])

            mapper = mappers.VoronoiMapper(pixels=6, grids=grids, border=None, pixel_neighbors=pixel_neighbors,
                                           pixel_centers=pixel_centers, voronoi=voronoi,
                                           image_to_nearest_image_pix=image_to_nearest_image_pix)

            assert (mapper.image_to_pix == image_to_pixelization_nearest_neighbour).all()

    class TestSubToPixelization:

        def test__sub_to_pixelization_of_mapper_matches_nearest_neighbor_calculation(self):

            pixel_centers = np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1],
                                      [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]])

            pixelization_sub_grid = np.array([[0.05, 0.15], [0.15, 0.15], [0.05, 0.05], [0.15, 0.05],
                                              [1.05, 0.15], [1.15, 0.15], [1.05, 0.05], [1.15, 0.05],
                                              [2.05, 0.15], [2.15, 0.15], [2.05, 0.05], [2.15, 0.05],
                                              [0.05, 1.15], [0.15, 1.15], [0.05, 1.05], [0.15, 1.05],
                                              [1.05, 1.15], [1.15, 1.15], [1.05, 1.05], [1.15, 1.05],
                                              [2.05, 1.15], [2.15, 1.15], [2.05, 1.05], [2.15, 1.05]])

            sub_to_pixelization_nearest_neighbour = grid_to_pixel_pixels_via_nearest_neighbour(pixelization_sub_grid,
                                                                                               pixel_centers)

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
            grids = MockGridCollection(image=np.array([]), sub=MockSubGrid(pixelization_sub_grid, sub_to_image,
                                                                           sub_grid_size=1))

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(pixel_centers)
            pixel_neighbors = pix.neighbors_from_pixelization(pixels=6, ridge_points=voronoi.ridge_points)
            image_to_nearest_image_pix = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])

            mapper = mappers.VoronoiMapper(pixels=6, grids=grids, border=None, pixel_neighbors=pixel_neighbors,
                                           pixel_centers=pixel_centers, voronoi=voronoi,
                                           image_to_nearest_image_pix=image_to_nearest_image_pix)

            assert (mapper.sub_to_pix == sub_to_pixelization_nearest_neighbour).all()
