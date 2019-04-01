import numpy as np
import pytest

from autolens.data.array import grids, mask as msk
from autolens.model.inversion import mappers as pm
from autolens.model.inversion import pixelizations, regularization
from test.unit.mock.mock_imaging import MockSubGrid, MockGridStack


class TestRectangular:

    def test__5_simple_grid__no_sub_grid(self):
        # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.
        regular_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        sub_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])

        sub_to_regular = np.array([0, 1, 2, 3, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid=sub_grid,
                                                           sub_to_regular=sub_to_regular, sub_grid_size=1))

        # There is no sub-grid, so our sub_grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix ==
                np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                          [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                          [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()

    def test__15_grid__no_sub_grid(self):
        # Source-plane comprises 15 grid, so 15 masked_image pixels traced to the pix-plane.

        regular_grid = np.array([[0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                      [0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                      [0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                      [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                      [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1]])

        # There is no sub-grid, so our sub_grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)
        sub_grid = np.array([[0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                          [0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                          [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                          [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                          [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1]])

        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular,
                                                   sub_grid_size=1))

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.2, 2.2), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()

    def test__5_simple_grid__include_sub_grid(self):
        # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.
        regular_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
        # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
        # happen for a real lensing calculation. This is to make a mapping_matrix matrix which explicitly tests the
        # sub-grid.
        sub_grid = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                          [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                          [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0],
                                          [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]])

        sub_to_regular = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=2))

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (mapper.mapping_matrix == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()

    def test__grid__requires_border_relocation(self):

        regular_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        sub_grid = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [-2.0, -2.0]])

        border = grids.RegularGridBorder(arr=np.array([0, 1, 3, 4]))

        sub_to_regular = np.array([0, 1, 2, 3, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular,
                                                   sub_grid_size=1))

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack, border)

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (mapper.mapping_matrix == np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()


class TestImagePlanePixelization:

    def test__3x3_simple_grid__create_using_regular_grid(self):

        regular_grid = np.array([[1.0, - 1.0], [1.0, 0.0], [1.0, 1.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                               [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

        mask = msk.Mask(array=np.array([[False, False, False],
                                        [False, False, False],
                                        [False, False, False]]), pixel_scale=1.0)

        sub_grid = np.array([[1.0, - 1.0], [1.0, 0.0], [1.0, 1.0],
                             [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                             [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        regular_grid = grids.RegularGrid(arr=regular_grid, mask=mask)
        sub_grid = MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1)

        pix = pixelizations.AdaptiveMagnification(shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        grid_stack = MockGridStack(regular=regular_grid, sub=sub_grid, pix=image_plane_pix.sparse_grid,
                                   regular_to_nearest_pix=image_plane_pix.regular_to_sparse)

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == image_plane_pix.sparse_grid).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()

    def test__3x3_simple_grid__include_mask__create_using_regular_grid(self):

        regular_grid = np.array([             [1.0, 0.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                            [-1.0, 0.0]] )

        mask = msk.Mask(array=np.array([[True, False, True],
                                        [False, False, False],
                                        [True, False, True]]), pixel_scale=1.0)

        sub_grid = np.array([              [1.0, 0.0],
                                   [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                [-1.0, 0.0]])
        sub_to_regular = np.array([0, 1, 2, 3, 4])

        regular_grid = grids.RegularGrid(arr=regular_grid, mask=mask)
        sub_grid = MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1)

        pix = pixelizations.AdaptiveMagnification(shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        grid_stack = MockGridStack(regular=regular_grid, sub=sub_grid, pix=image_plane_pix.sparse_grid,
                                   regular_to_nearest_pix=image_plane_pix.regular_to_sparse)

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == image_plane_pix.sparse_grid).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__3x3_simple_grid__include_mask_and_sub_grid__create_using_regular_grid(self):

        regular_grid = np.array([             [1.0, 0.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                            [-1.0, 0.0]] )

        mask = msk.Mask(array=np.array([[True, False, True],
                                        [False, False, False],
                                        [True, False, True]]), pixel_scale=1.0)

        sub_grid = np.array([[1.01, 0.0], [1.01, 0.0], [1.01, 0.0], [0.01, 0.0],
                                  [0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.01, 0.0],
                                  [0.01, 0.0], [0.01, 0.0], [0.01, 0.0], [0.01, 0.0],
                                  [0.0, 1.01], [0.0, 1.01], [0.0, 1.01], [0.01, 0.0],
                                  [-1.01, 0.0], [-1.01, 0.0], [-1.01, 0.0], [0.01, 0.0]])

        sub_to_regular = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

        regular_grid = grids.RegularGrid(arr=regular_grid, mask=mask)
        sub_grid = MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=2)

        pix = pixelizations.AdaptiveMagnification(shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        grid_stack = MockGridStack(regular=regular_grid, sub=sub_grid, pix=image_plane_pix.sparse_grid,
                                   regular_to_nearest_pix=image_plane_pix.regular_to_sparse)

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.02, 2.01), 1.0e-4)
        assert (mapper.geometry.pixel_centres == image_plane_pix.sparse_grid).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.005), 1.0e-4)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[0.75, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.75, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 2.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.75, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.75]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__3x3_simple_grid__include_mask_with_offset_centre__create_using_regular_grid(self):

        regular_grid = np.array([          [2.0, 1.0],
                               [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                                           [0.0, 1.0]])

        mask = msk.Mask(array=np.array([[True, True, True, False, True],
                                        [True, True, False, False, False],
                                        [True, True, True, False, True],
                                        [True, True, True, True, True],
                                        [True, True, True, True, True]]), pixel_scale=1.0)

        sub_grid =np.array([          [2.0, 1.0],
                          [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                                      [0.0, 1.0]])
        sub_to_regular = np.array([0, 1, 2, 3, 4])

        regular_grid = grids.RegularGrid(arr=regular_grid, mask=mask)
        sub_grid = MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1)

        pix = pixelizations.AdaptiveMagnification(shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        grid_stack = MockGridStack(regular=regular_grid, sub=sub_grid, pix=image_plane_pix.sparse_grid,
                                   regular_to_nearest_pix=image_plane_pix.regular_to_sparse)

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == image_plane_pix.sparse_grid).all()
        assert mapper.geometry.origin == pytest.approx((1.0, 1.0), 1.0e-4)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()


class TestAdaptiveMagnification:

    def test__5_simple_grid__no_sub_grid(self):

        regular_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
        sub_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

        sub_to_regular = np.array([0, 1, 2, 3, 4])

        regular_to_sparse = np.array([0, 1, 2, 3, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1),
                                   pix=regular_grid,
                                   regular_to_nearest_pix=regular_to_sparse)

        pix = pixelizations.AdaptiveMagnification(shape=(5, 1))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == regular_grid).all()
        assert mapper.geometry.origin == (0.0, 0.0)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__3x3_simple_grid__include_mask_with_offset_centre__create_using_regular_grid(self):

        regular_grid = np.array([          [2.0, 1.0],
                               [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                                           [0.0, 1.0]])

        mask = msk.Mask(array=np.array([[True, True, True, False, True],
                                        [True, True, False, False, False],
                                        [True, True, True, False, True],
                                        [True, True, True, True, True],
                                        [True, True, True, True, True]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid(arr=regular_grid, mask=mask)

        sub_grid =np.array([          [2.0, 1.0],
                          [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                                      [0.0, 1.0]])

        sub_to_regular = np.array([0, 1, 2, 3, 4])

        regular_to_sparse = np.array([0, 1, 2, 3, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1),
                                   pix=regular_grid,
                                   regular_to_nearest_pix=regular_to_sparse)

        pix = pixelizations.AdaptiveMagnification(shape=(5, 1))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == regular_grid).all()
        assert mapper.geometry.origin == (1.0, 1.0)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()



        sub_to_regular = np.array([0, 1, 2, 3, 4])

        regular_grid = grids.RegularGrid(arr=regular_grid, mask=mask)
        sub_grid = MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1)

        pix = pixelizations.AdaptiveMagnification(shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_regular_grid(regular_grid=regular_grid)

        grid_stack = MockGridStack(regular=regular_grid, sub=sub_grid, pix=image_plane_pix.sparse_grid,
                                   regular_to_nearest_pix=image_plane_pix.regular_to_sparse)

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == image_plane_pix.sparse_grid).all()
        assert mapper.geometry.origin == pytest.approx((1.0, 1.0), 1.0e-4)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__15_grid__no_sub_grid(self):

        regular_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                      [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                      [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                      [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                      [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

        sub_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                          [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                          [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                          [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                          [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

        pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        regular_to_sparse = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1),
                                   pix=pixel_centers, regular_to_nearest_pix=regular_to_sparse)

        pix = pixelizations.AdaptiveMagnification(shape=(5, 1))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.2, 2.2), 1.0e-4)
        assert (mapper.geometry.pixel_centres == pixel_centers).all()
        assert mapper.geometry.origin == (0.0, 0.0)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__5_simple_grid__include_sub_grid__sets_up_correct_mapper(self):

        regular_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

        sub_grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                          [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                          [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                          [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

        sub_to_regular = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

        pixel_centers = regular_grid

        regular_to_sparse = np.array([0, 1, 2, 3, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular,
                                                   sub_grid_size=2),
                                   pix=pixel_centers,
                                   regular_to_nearest_pix=regular_to_sparse)

        pix = pixelizations.AdaptiveMagnification(shape=(5, 1))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=None)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == pixel_centers).all()
        assert mapper.geometry.origin == (0.0, 0.0)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                       [0.0, 0.75, 0.25, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.25, 0.75, 0.0],
                                                       [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__same_as_above_but_grid_requires_border_relocation(self):

        regular_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        sub_grid = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [-2.0, -2.0]])
        # These will all be relocated to the regular grid edge.
        pix_grid = np.array([[1.1, -1.1], [1.1, 1.1], [0.0, 0.0], [-1.1, -1.1], [-1.1, 1.1]])

        border = grids.RegularGridBorder(arr=np.array([0, 1, 3, 4]))

        sub_to_regular = np.array([0, 1, 2, 3, 4])

        regular_to_sparse = np.array([0, 1, 2, 3, 4])

        grid_stack = MockGridStack(regular=regular_grid,
                                   sub=MockSubGrid(sub_grid, sub_to_regular, sub_grid_size=1),
                                   pix=pix_grid, regular_to_nearest_pix=regular_to_sparse)

        pix = pixelizations.AdaptiveMagnification(shape=(5, 1))

        mapper = pix.mapper_from_grid_stack_and_border(grid_stack=grid_stack, border=border)

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.pixel_centres == pytest.approx(regular_grid, 1e-4)
        assert mapper.geometry.origin == (0.0, 0.0)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.geometry.pixel_neighbors,
                                                                               mapper.geometry.pixel_neighbors_size)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()