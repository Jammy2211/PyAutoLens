import numpy as np
import pytest
import scipy.spatial

from autolens.data.array import grids, mask
from autolens.model.inversion import pixelizations
from autolens.model.inversion.util import pixelization_util
from autolens.model.inversion import regularization
from autolens.model.galaxy import galaxy as g


class TestImagePlanePixelization:

    def test__pixelization_regular_grid_from_regular_grid__sets_up_with_correct_shape_and_pixel_scales(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        adaptive_image_grid = pixelizations.ImagePlanePixelization(shape=(3, 3))

        pix_grid = adaptive_image_grid.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        assert pix_grid.shape == (3,3)
        assert pix_grid.pixel_scales == (1.0, 1.0)
        assert pix_grid.total_sparse_pixels == 9
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (pix_grid.sparse_grid == np.array([[1.0, - 1.0], [1.0, 0.0], [1.0, 1.0],
                                                  [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                  [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])).all()
        assert pix_grid.regular_grid == pytest.approx(regular_grid, 1e-4)

    def test__same_as_above__but_4x3_image(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        adaptive_image_grid = pixelizations.ImagePlanePixelization(shape=(4, 3))

        pix_grid = adaptive_image_grid.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        assert pix_grid.total_sparse_pixels == 8
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7])).all()
        assert (pix_grid.sparse_grid == np.array([[1.5, 0.0],
                                                  [0.5, -1.0], [0.5, 0.0], [0.5, 1.0],
                                                  [-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0],
                                                  [-1.5, 0.0]])).all()

    def test__same_as_above__but_3x4_image(self):

        ma = mask.Mask(array=np.array([[True, False, True, True],
                                       [False, False, False, False],
                                       [True, False, True, True]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        adaptive_image_grid = pixelizations.ImagePlanePixelization(shape=(3, 4))

        pix_grid = adaptive_image_grid.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        assert pix_grid.total_sparse_pixels == 6
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 4, 5, 6, 7, 9])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 4, 5, 6, 7, 9])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5])).all()
        assert (pix_grid.sparse_grid == np.array([[1.0, -0.5],
                                                  [0.0, -1.5], [0.0, -0.5], [0.0, 0.5], [0.0, 1.5],
                                                  [-1.0, -0.5]])).all()

    def test__setup_pixelization__galaxies_have_no_pixelization__returns_normal_grids(self):

        ma = mask.Mask(np.array([[False, False, False],
                                 [False, False, False],
                                 [False, False, False]]), pixel_scale=1.0)

        data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                 psf_shape=(1, 1))

        galaxy = g.Galaxy()

        image_plane_pix_grids = \
            pixelizations.setup_image_plane_pixelization_grid_from_galaxies_and_grids(galaxies=[galaxy, galaxy],
                                                                                      grids=data_grids)

        assert image_plane_pix_grids == data_grids

    def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grids(self):

        ma = mask.Mask(np.array([[False, False, False],
                                 [False, False, False],
                                 [False, False, False]]), pixel_scale=1.0)

        data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                 psf_shape=(1, 1))

        galaxy = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3,3)),
                          regularization=regularization.Constant())


        image_plane_pix_grids = \
            pixelizations.setup_image_plane_pixelization_grid_from_galaxies_and_grids(galaxies=[galaxy, galaxy],
                                                                                      grids=data_grids)

        assert image_plane_pix_grids == data_grids

    def test__setup_pixelization__galaxy_has_pixelization__returns_grids_with_pix_grid(self):
        
        ma = mask.Mask(np.array([[False, False, False],
                                 [False, False, False],
                                 [False, True, False]]), pixel_scale=1.0)

        data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1, 
                                                                                 psf_shape=(1, 1))

        galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                          regularization=regularization.Constant())

        image_plane_pix_grids = \
            pixelizations.setup_image_plane_pixelization_grid_from_galaxies_and_grids(galaxies=[galaxy, galaxy],
                                                                                      grids=data_grids)

        assert (image_plane_pix_grids.regular == data_grids.regular).all()
        assert (image_plane_pix_grids.sub == data_grids.sub).all()
        assert (image_plane_pix_grids.blurring == data_grids.blurring).all()
        assert (image_plane_pix_grids.pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                       [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                       [-1.0, -1.0],             [-1.0, 1.0]])).all()


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
            assert (geometry.pixel_neighbors[0] == [1, 3, -1, -1]).all()
            assert (geometry.pixel_neighbors[1] == [0, 2, 4, -1]).all()
            assert (geometry.pixel_neighbors[2] == [1, 5, -1, -1]).all()
            assert (geometry.pixel_neighbors[3] == [0, 4, 6, -1]).all()
            assert (geometry.pixel_neighbors[4] == [1, 3, 5, 7]).all()
            assert (geometry.pixel_neighbors[5] == [2, 4, 8, -1]).all()
            assert (geometry.pixel_neighbors[6] == [3, 7, -1, -1]).all()
            assert (geometry.pixel_neighbors[7] == [4, 6, 8, -1]).all()
            assert (geometry.pixel_neighbors[8] == [5, 7, -1, -1]).all()

            assert (geometry.pixel_neighbors_size == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()

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

        def test__compare_to_pixelization_util(self):
            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pix = pixelizations.Rectangular(shape=(7, 5))

            pixel_neighbors, pixel_neighbors_size = pix.neighbors_from_pixelization()
            pixel_neighbors_util, pixel_neighbors_size_util = \
                pixelization_util.rectangular_neighbors_from_shape(shape=(7, 5))

            assert (pixel_neighbors == pixel_neighbors_util).all()
            assert (pixel_neighbors_size == pixel_neighbors_size_util).all()


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

        def test__compare_to_pixelization_util(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[3.0, 0.0], [2.0, 1.0], [2.0, 2.0],
                               [8.0, 3.0], [1.0, 3.0], [1.0, 9.0],
                               [6.0, 31.0], [0.0, 2.0], [3.0, 5.0]])

            pix = pixelizations.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)
            pixel_neighbors, pixel_neighbors_size = pix.neighbors_from_pixelization(pixels=9,
                                                                                    ridge_points=voronoi.ridge_points)

            voronoi = scipy.spatial.Voronoi(points, qhull_options='Qbb Qc Qx Qm')
            pixel_neighbors_util, pixel_neighbors_size_util = \
                pixelization_util.voronoi_neighbors_from_pixels_and_ridge_points(pixels=9,
                                              ridge_points=np.array(voronoi.ridge_points))

            assert (pixel_neighbors == pixel_neighbors_util).all()
            assert (pixel_neighbors_size == pixel_neighbors_size_util).all()


class TestAdaptiveMagnification:

    class TestConstructor:

        def test__number_of_pixels_and_regularization_set_up_correctly(self):

            pix = pixelizations.AdaptiveMagnification(shape=(3, 3))

            assert pix.shape == (3, 3)


class TestAmorphous:

    class TestKMeans:

        def test__simple_points__sets_up_two_clusters(self):

            cluster_grid = np.array([[1.99, 0.99], [2.0, 1.0], [2.01, 1.01],
                                     [0.99, 1.99], [1.0, 2.0], [1.01, 2.01]])

            pix = pixelizations.Amorphous(pix_grid_shape=(1, 2))

            pixel_centers, pix_to_image = pix.kmeans_cluster(pixels=2, cluster_grid=cluster_grid)

            assert [2.0, 2.0] in pixel_centers
            assert [1.0, 1.0] in pixel_centers

            assert list(pix_to_image).count(0) == 3
            assert list(pix_to_image).count(1) == 3

        def test__simple_points__sets_up_three_clusters(self):
            cluster_grid = np.array([[1.99, -0.99], [2.0, -1.0], [2.01, -1.01],
                                     [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                     [-0.99, 1.99], [-1.0, 2.0], [-1.01, 2.01]])

            pix = pixelizations.Amorphous(pix_grid_shape=(1, 3))

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

            pix = pixelizations.Amorphous(pix_grid_shape=(1, 3))

            pixel_centers, pix_to_image = pix.kmeans_cluster(pixels=3, cluster_grid=cluster_grid)

            pixel_centers = list(map(lambda x: pytest.approx(list(x), 1e-3), pixel_centers))

            assert [2.0, 2.0] in pixel_centers
            assert [1.0, 1.0] in pixel_centers
            assert [-1.0, -1.0] in pixel_centers

            assert list(pix_to_image).count(0) == 3 or 6 or 12
            assert list(pix_to_image).count(1) == 3 or 6 or 12
            assert list(pix_to_image).count(2) == 3 or 6 or 12

            assert list(pix_to_image).count(0) != list(pix_to_image).count(1) != list(pix_to_image).count(2)