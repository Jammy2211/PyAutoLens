import numpy as np
import pytest

from autolens.imaging import mask
from autolens.inversion.util import pixelization_util
from autolens.inversion import pixelizations


class MockGeometry(object):

    def __init__(self, y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_pixel_scale = y_pixel_scale
        self.x_pixel_scale = x_pixel_scale
        self.origin = (0.0, 0.0)


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


class TestPixelizationGrid:

    def test__properties_consistent_with_pixelization_util(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=0.5)

        image_grid = mask.ImageGrid.from_mask(mask=ma)

        pix_grid = pixelizations.ImagePlanePixelizationGrid(pix_grid_shape=(10, 10), pixel_scales=(0.16, 0.16),
                                                            image_grid=image_grid)

        full_pix_grid_pixel_centres = image_grid.mask.grid_arc_seconds_to_grid_pixel_centres(pix_grid.full_pix_grid)
        total_pix_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        image_to_full_pix_util = pix_grid.grid_arc_seconds_to_grid_pixel_indexes(grid_arc_seconds=image_grid)

        pix_to_full_pix_util = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(
            total_pix_pixels=total_pix_pixels, mask=ma, full_pix_grid_pixel_centres=full_pix_grid_pixel_centres).astype('int')

        full_pix_to_pix_util = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                             full_pix_grid_pixel_centres=full_pix_grid_pixel_centres).astype('int')

        image_to_pix_util = pixelization_util.image_to_pix_from_pix_mappings(image_to_full_pix=image_to_full_pix_util,
                                                                             full_pix_to_pix=full_pix_to_pix_util)

        pix_grid_util = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=pix_grid.grid_1d,
                                                                      pix_to_full_pix=pix_to_full_pix_util)

        assert pix_grid.total_pix_pixels == total_pix_pixels
        assert (pix_grid.pix_to_full_pix == pix_to_full_pix_util).all()
        assert (pix_grid.full_pix_to_pix == full_pix_to_pix_util).all()
        assert (pix_grid.image_to_full_pix == image_to_full_pix_util).all()
        assert (pix_grid.image_to_pix == image_to_pix_util).all()
        assert (pix_grid.pix_grid == pix_grid_util).all()

    def test__pixelization_grid_overlaps_mask_perfectly__masked_pixels_in_masked_pixelization_grid(self):

            ma = mask.Mask(array=np.array([[True, False, True],
                                           [False, False, False],
                                           [True, False, True]]), pixel_scale=1.0)

            image_grid = mask.ImageGrid.from_mask(mask=ma)

            pix_grid = pixelizations.ImagePlanePixelizationGrid(pix_grid_shape=(3, 3), pixel_scales=(1.0, 1.0),
                                                                image_grid=image_grid)

            assert pix_grid.total_pix_pixels == 5
            assert (pix_grid.pix_to_full_pix == np.array([1, 3, 4, 5, 7])).all()
            assert (pix_grid.full_pix_to_pix == np.array([0, 0, 1, 1, 2, 3, 4, 4, 5])).all()
            assert (pix_grid.image_to_full_pix == np.array([1, 3, 4, 5, 7])).all()
            assert (pix_grid.image_to_pix == np.array([0, 1, 2, 3, 4])).all()
            assert (pix_grid.pix_grid == np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                   [-1.0, 0.0]])).all()

    def test__same_as_above_but_4x3_grid_and_mask(self):

            ma = mask.Mask(array=np.array([[True, False, True],
                                           [False, False, False],
                                           [False, False, False],
                                           [True, False, True]]), pixel_scale=1.0)

            image_grid = mask.ImageGrid.from_mask(mask=ma)

            pix_grid = pixelizations.ImagePlanePixelizationGrid(pix_grid_shape=(4, 3), pixel_scales=(1.0, 1.0),
                                                                image_grid=image_grid)

            assert pix_grid.total_pix_pixels == 8
            assert (pix_grid.pix_to_full_pix == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
            assert (pix_grid.full_pix_to_pix == np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8])).all()
            assert (pix_grid.image_to_full_pix == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
            assert (pix_grid.image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7])).all()
            assert (pix_grid.pix_grid == np.array([[1.5, 0.0],
                                                   [ 0.5, -1.0], [ 0.5, 0.0], [ 0.5, 1.0],
                                                   [-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0],
                                                   [-1.5, 0.0]])).all()

    def test__same_as_above_but_3x4_grid_and_mask(self):

            ma = mask.Mask(array=np.array([[True,  False,  True,  True],
                                           [False, False, False, False],
                                           [True,  False,  True,  True]]), pixel_scale=1.0)

            image_grid = mask.ImageGrid.from_mask(mask=ma)

            pix_grid = pixelizations.ImagePlanePixelizationGrid(pix_grid_shape=(3, 4), pixel_scales=(1.0, 1.0),
                                                                image_grid=image_grid)

            assert pix_grid.total_pix_pixels == 6
            assert (pix_grid.pix_to_full_pix == np.array([1, 4, 5, 6, 7, 9])).all()
            assert (pix_grid.full_pix_to_pix == np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6])).all()
            assert (pix_grid.image_to_full_pix == np.array([1, 4, 5, 6, 7, 9])).all()
            assert (pix_grid.image_to_pix == np.array([0, 1, 2, 3, 4, 5])).all()
            assert (pix_grid.pix_grid == np.array([             [1.0, -0.5],
                                                   [0.0, -1.5], [0.0, -0.5], [0.0, 0.5], [0.0, 1.5],
                                                               [-1.0, -0.5]])).all()


class TestImagePlanePixelization:

    def test__pixelization_image_grid_from_image_grid__sets_up_with_correct_shape_and_pixel_scales(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        image_grid = mask.ImageGrid.from_mask(mask=ma)

        adaptive_image_grid = pixelizations.ImagePlanePixelization(pix_grid_shape=(3, 3))

        pix_grid = adaptive_image_grid.image_plane_pix_grid_from_image_grid(image_grid=image_grid)

        assert pix_grid.shape == (3,3)
        assert pix_grid.pixel_scales == (1.0, 1.0)
        assert pix_grid.total_pix_pixels == 9
        assert (pix_grid.pix_to_full_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.full_pix_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.image_to_full_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.pix_grid ==  np.array([[1.0, - 1.0], [1.0, 0.0], [1.0, 1.0],
                                                [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])).all()
        assert pix_grid.image_grid == pytest.approx(image_grid, 1e-4)

    def test__same_as_above__but_4x3_image(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        image_grid = mask.ImageGrid.from_mask(mask=ma)

        adaptive_image_grid = pixelizations.ImagePlanePixelization(pix_grid_shape=(4, 3))

        pix_grid = adaptive_image_grid.image_plane_pix_grid_from_image_grid(image_grid=image_grid)

        assert pix_grid.total_pix_pixels == 8
        assert (pix_grid.pix_to_full_pix == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
        assert (pix_grid.full_pix_to_pix == np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8])).all()
        assert (pix_grid.image_to_full_pix == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
        assert (pix_grid.image_to_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7])).all()
        assert (pix_grid.pix_grid == np.array([[1.5, 0.0],
                                               [0.5, -1.0], [0.5, 0.0], [0.5, 1.0],
                                               [-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0],
                                               [-1.5, 0.0]])).all()

    def test__same_as_above__but_3x4_image(self):

        ma = mask.Mask(array=np.array([[True, False, True, True],
                                       [False, False, False, False],
                                       [True, False, True, True]]), pixel_scale=1.0)

        image_grid = mask.ImageGrid.from_mask(mask=ma)

        adaptive_image_grid = pixelizations.ImagePlanePixelization(pix_grid_shape=(3, 4))

        pix_grid = adaptive_image_grid.image_plane_pix_grid_from_image_grid(image_grid=image_grid)

        assert pix_grid.total_pix_pixels == 6
        assert (pix_grid.pix_to_full_pix == np.array([1, 4, 5, 6, 7, 9])).all()
        assert (pix_grid.full_pix_to_pix == np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6])).all()
        assert (pix_grid.image_to_full_pix == np.array([1, 4, 5, 6, 7, 9])).all()
        assert (pix_grid.image_to_pix == np.array([0, 1, 2, 3, 4, 5])).all()
        assert (pix_grid.pix_grid == np.array([[1.0, -0.5],
                                               [0.0, -1.5], [0.0, -0.5], [0.0, 0.5], [0.0, 1.5],
                                               [-1.0, -0.5]])).all()



class TestRectangular:

    class TestConstructor:

        def test__number_of_pixels_and_regularization_set_up_correctly(self):
            pix = pixelizations.Rectangular(shape=(3, 3))

            assert pix.shape == (3, 3)
            assert pix.pixels == 9

    class TestGeometry:

        def test__3x3_grid__buffer_is_small__grid_give_min_minus_1_max_1__sets_up_geometry_correctly(self):

            pix = pixelizations.Rectangular(shape=(3, 3))

            pixelization_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.shape == (3,3)
            assert geometry.pixel_scales == pytest.approx((2./3., 2./3.), 1e-2)

        def test__3x3_grid__same_as_above_change_buffer(self):
            pix = pixelizations.Rectangular(shape=(3, 3))

            pixelization_grid = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-4)

            assert geometry.shape == (3,3)
            assert geometry.pixel_scales == pytest.approx((2./3., 2./3.), 1e-2)

        def test__5x4_grid__buffer_is_small(self):

            pix = pixelizations.Rectangular(shape=(5, 4))

            pixelization_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.shape == (5,4)
            assert geometry.pixel_scales == pytest.approx((2./5., 2./4.), 1e-2)

        def test__3x3_grid__larger_range_of_grid(self):

            pix = pixelizations.Rectangular(shape=(3, 3))

            pixelization_grid = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.shape == (3,3)
            assert geometry.pixel_scales == pytest.approx((6./3., 6./3.), 1e-2)

    class TestPixelCentres:

        def test__3x3_grid__pixel_centres(self):

            pix = pixelizations.Rectangular(shape=(3, 3))

            pixelization_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.pixel_centres == pytest.approx(np.array([[2./3., -2./3.], [2./3., 0.0], [2./3., 2./3.],
                                                                     [ 0.0, -2./3.], [ 0.0, 0.0], [ 0.0, 2./3.],
                                                                     [-2./3., -2./3.], [-2./3., 0.0], [-2./3., 2./3.]]))

        def test__4x3_grid__pixel_centres(self):

            pix = pixelizations.Rectangular(shape=(4, 3))

            pixelization_grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                          [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                          [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.pixel_centres == pytest.approx(np.array([[0.75, -2./3.], [0.75, 0.0], [0.75, 2./3.],
                                                                     [0.25, -2./3.], [0.25, 0.0], [0.25, 2./3.],
                                                                     [-0.25, -2./3.], [-0.25, 0.0], [-0.25, 2./3.],
                                                                     [-0.75, -2./3.], [-0.75, 0.0],[-0.75, 2./ 3.],]))

    class TestPixelNeighbors:

        def test__compute_pixel_neighbors__3x3_grid(self):
            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            pix = pixelizations.Rectangular(shape=(3, 3))

            pixel_neighbors = pix.neighbors_from_pixelization()

            assert pixel_neighbors[0] == [1, 3]
            assert pixel_neighbors[1] == [0, 2, 4]
            assert pixel_neighbors[2] == [1, 5]
            assert pixel_neighbors[3] == [0, 4, 6]
            assert pixel_neighbors[4] == [1, 3, 5, 7]
            assert pixel_neighbors[5] == [2, 4, 8]
            assert pixel_neighbors[6] == [3, 7]
            assert pixel_neighbors[7] == [4, 6, 8]
            assert pixel_neighbors[8] == [5, 7]

        def test__compute_pixel_neighbors__3x4_grid(self):
            # |0|1| 2| 3|
            # |4|5| 6| 7|
            # |8|9|10|11|

            pix = pixelizations.Rectangular(shape=(3, 4))

            pixel_neighbors = pix.neighbors_from_pixelization()

            assert pixel_neighbors[0] == [1, 4]
            assert pixel_neighbors[1] == [0, 2, 5]
            assert pixel_neighbors[2] == [1, 3, 6]
            assert pixel_neighbors[3] == [2, 7]
            assert pixel_neighbors[4] == [0, 5, 8]
            assert pixel_neighbors[5] == [1, 4, 6, 9]
            assert pixel_neighbors[6] == [2, 5, 7, 10]
            assert pixel_neighbors[7] == [3, 6, 11]
            assert pixel_neighbors[8] == [4, 9]
            assert pixel_neighbors[9] == [5, 8, 10]
            assert pixel_neighbors[10] == [6, 9, 11]
            assert pixel_neighbors[11] == [7, 10]

        def test__compute_pixel_neighbors__4x3_grid(self):
            # |0| 1| 2|
            # |3| 4| 5|
            # |6| 7| 8|
            # |9|10|11|

            pix = pixelizations.Rectangular(shape=(4, 3))

            pixel_neighbors = pix.neighbors_from_pixelization()

            assert pixel_neighbors[0] == [1, 3]
            assert pixel_neighbors[1] == [0, 2, 4]
            assert pixel_neighbors[2] == [1, 5]
            assert pixel_neighbors[3] == [0, 4, 6]
            assert pixel_neighbors[4] == [1, 3, 5, 7]
            assert pixel_neighbors[5] == [2, 4, 8]
            assert pixel_neighbors[6] == [3, 7, 9]
            assert pixel_neighbors[7] == [4, 6, 8, 10]
            assert pixel_neighbors[8] == [5, 7, 11]
            assert pixel_neighbors[9] == [6, 10]
            assert pixel_neighbors[10] == [7, 9, 11]
            assert pixel_neighbors[11] == [8, 10]

        def test__compute_pixel_neighbors__4x4_grid(self):
            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pix = pixelizations.Rectangular(shape=(4, 4))

            pixel_neighbors = pix.neighbors_from_pixelization()

            assert pixel_neighbors[0] == [1, 4]
            assert pixel_neighbors[1] == [0, 2, 5]
            assert pixel_neighbors[2] == [1, 3, 6]
            assert pixel_neighbors[3] == [2, 7]
            assert pixel_neighbors[4] == [0, 5, 8]
            assert pixel_neighbors[5] == [1, 4, 6, 9]
            assert pixel_neighbors[6] == [2, 5, 7, 10]
            assert pixel_neighbors[7] == [3, 6, 11]
            assert pixel_neighbors[8] == [4, 9, 12]
            assert pixel_neighbors[9] == [5, 8, 10, 13]
            assert pixel_neighbors[10] == [6, 9, 11, 14]
            assert pixel_neighbors[11] == [7, 10, 15]
            assert pixel_neighbors[12] == [8, 13]
            assert pixel_neighbors[13] == [9, 12, 14]
            assert pixel_neighbors[14] == [10, 13, 15]
            assert pixel_neighbors[15] == [11, 14]


class TestVoronoi:

    class TestVoronoiGrid:

        def test__points_in_x_cross_shape__sets_up_diamond_voronoi_vertices(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0, 1.] in voronoi.vertices
            assert [-1., 0.] in voronoi.vertices
            assert [1., 0.] in voronoi.vertices
            assert [0., -1.] in voronoi.vertices

        def test__9_points_in_square___sets_up_square_of_voronoi_vertices(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                               [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                               [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0.5, 1.5] in voronoi.vertices
            assert [1.5, 0.5] in voronoi.vertices
            assert [0.5, 0.5] in voronoi.vertices
            assert [1.5, 1.5] in voronoi.vertices

        def test__points_in_x_cross_shape__sets_up_pairs_of_voronoi_cells(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[1.0, -1.0], [1.0, 1.0],
                                      [0.0, 0.0],
                               [-1.0, -1.0], [-1.0, 1.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 8

            assert [2, 0] in voronoi.ridge_points or [0, 2] in voronoi.ridge_points
            assert [2, 1] in voronoi.ridge_points or [1, 2] in voronoi.ridge_points
            assert [2, 3] in voronoi.ridge_points or [3, 2] in voronoi.ridge_points
            assert [2, 4] in voronoi.ridge_points or [4, 2] in voronoi.ridge_points
            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [0.3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 1] in voronoi.ridge_points or [1, 4] in voronoi.ridge_points

        def test__9_points_in_square___sets_up_pairs_of_voronoi_cells(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                               [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                               [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 12

            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [1, 2] in voronoi.ridge_points or [2, 1] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 5] in voronoi.ridge_points or [5, 4] in voronoi.ridge_points
            assert [6, 7] in voronoi.ridge_points or [7, 6] in voronoi.ridge_points
            assert [7, 8] in voronoi.ridge_points or [8, 7] in voronoi.ridge_points

            assert [0, 3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [1, 4] in voronoi.ridge_points or [4, 1] in voronoi.ridge_points
            assert [4, 7] in voronoi.ridge_points or [7, 4] in voronoi.ridge_points
            assert [2, 5] in voronoi.ridge_points or [5, 2] in voronoi.ridge_points
            assert [5, 8] in voronoi.ridge_points or [8, 5] in voronoi.ridge_points
            assert [3, 6] in voronoi.ridge_points or [6, 3] in voronoi.ridge_points

    class TestNeighbors:

        def test__points_in_x_cross_shape__neighbors_of_each_pixel_correct(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[1.0, -1.0], [1.0, 1.0],
                                     [0.0, 0.0],
                               [-1.0, -1.0], [-1.0, 1.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)
            neighbors = pix.neighbors_from_pixelization(pixels=5, ridge_points=voronoi.ridge_points)

            assert set(neighbors[0]) == {2, 1, 3}
            assert set(neighbors[1]) == {2, 0, 4}
            assert set(neighbors[2]) == {0, 1, 3, 4}
            assert set(neighbors[3]) == {2, 0, 4}
            assert set(neighbors[4]) == {2, 1, 3}

        def test__9_points_in_square___neighbors_of_each_pixel_correct(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                               [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                               [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)
            neighbors = pix.neighbors_from_pixelization(pixels=9, ridge_points=voronoi.ridge_points)

            assert set(neighbors[0]) == {1, 3}
            assert set(neighbors[1]) == {0, 2, 4}
            assert set(neighbors[2]) == {1, 5}
            assert set(neighbors[3]) == {0, 4, 6}
            assert set(neighbors[4]) == {1, 3, 5, 7}
            assert set(neighbors[5]) == {2, 4, 8}
            assert set(neighbors[6]) == {3, 7}
            assert set(neighbors[7]) == {4, 6, 8}
            assert set(neighbors[8]) == {5, 7}


class TestAmorphous:

    class TestKMeans:

        def test__simple_points__sets_up_two_clusters(self):

            cluster_grid = np.array([[1.99, 0.99], [2.0, 1.0], [2.01, 1.01],
                                     [0.99, 1.99], [1.0, 2.0], [1.01, 2.01]])

            pix = pixelizations.Amorphous(image_grid_shape=(1,2))

            pixel_centers, pix_to_image = pix.kmeans_cluster(pixels=2, cluster_grid=cluster_grid)

            assert [2.0, 2.0] in pixel_centers
            assert [1.0, 1.0] in pixel_centers

            assert list(pix_to_image).count(0) == 3
            assert list(pix_to_image).count(1) == 3

        def test__simple_points__sets_up_three_clusters(self):
            cluster_grid = np.array([[1.99, -0.99], [2.0, -1.0], [2.01, -1.01],
                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                     [-0.99, 1.99], [-1.0, 2.0], [-1.01, 2.01]])

            pix = pixelizations.Amorphous(image_grid_shape=(1,3))

            pixel_centers, pix_to_image = pix.kmeans_cluster(pixels=3, cluster_grid=cluster_grid)

            assert [2.0, 2.0] in pixel_centers
            assert [1.0, 1.0] in pixel_centers
            assert [-1.0, -1.0] in pixel_centers

            assert list(pix_to_image).count(0) == 3
            assert list(pix_to_image).count(1) == 3
            assert list(pix_to_image).count(2) == 3

        def test__simple_points__sets_up_three_clusters_more_points_in_third_cluster(self):

            cluster_grid = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],

                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],

                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                     [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelizations.Amorphous(image_grid_shape=(1,3))

            pixel_centers, pix_to_image = pix.kmeans_cluster(pixels=3, cluster_grid=cluster_grid)

            pixel_centers = list(map(lambda x: pytest.approx(list(x), 1e-3), pixel_centers))

            assert [2.0, 2.0] in pixel_centers
            assert [1.0, 1.0] in pixel_centers
            assert [-1.0, -1.0] in pixel_centers

            assert list(pix_to_image).count(0) == 3 or 6 or 12
            assert list(pix_to_image).count(1) == 3 or 6 or 12
            assert list(pix_to_image).count(2) == 3 or 6 or 12

            assert list(pix_to_image).count(0) != list(pix_to_image).count(1) != list(pix_to_image).count(2)