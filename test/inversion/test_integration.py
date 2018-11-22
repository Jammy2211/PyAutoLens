import numpy as np

from autolens.imaging import mask as msk
from autolens.inversion import mappers as pm
from autolens.inversion import pixelizations
from autolens.inversion import regularization
from test.mock.mock_imaging import MockSubGrid, MockGridCollection


class TestRectangular:

    def test__5_simple_grid__no_sub_grid(self):
        # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.
        pixelization_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        pixelization_sub_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])

        sub_to_image = np.array([0, 1, 2, 3, 4])

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(sub_grid=pixelization_sub_grid,
                                                   sub_to_image=sub_to_image, sub_grid_size=1))

        # There is no sub-grid, so our sub_grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

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

        pixelization_grid = np.array([[0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                      [0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                      [0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                      [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                      [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1]])

        # There is no sub-grid, so our sub_grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)
        pixelization_sub_grid = np.array([[0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                          [0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                          [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                          [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                          [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1]])

        sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image,
                                                   sub_grid_size=1))

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None)

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
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

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
        pixelization_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
        # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
        # happen for a real lensing calculation. This is to make a mapping_matrix matrix which explicitly tests the
        # sub-grid.
        pixelization_sub_grid = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                          [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                          [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0],
                                          [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]])

        sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image, sub_grid_size=2))

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None)

        assert (mapper.mapping_matrix == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

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

        pixelization_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        pixelization_sub_grid = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [-2.0, -2.0]])

        border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

        sub_to_image = np.array([0, 1, 2, 3, 4])

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image,
                                                   sub_grid_size=1))

        pix = pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grids_and_border(grids, border)

        assert (mapper.mapping_matrix == np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])).all()
        assert mapper.shape == (3, 3)

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

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

    def test__3x3_simple_grid__create_using_image_grid(self):
        image_grid = np.array([[1.0, - 1.0], [1.0, 0.0], [1.0, 1.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                               [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])

        mask = msk.Mask(array=np.array([[False, False, False],
                                        [False, False, False],
                                        [False, False, False]]), pixel_scale=1.0)

        image_sub_grid = np.array([[1.0, - 1.0], [1.0, 0.0], [1.0, 1.0],
                                   [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                   [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])
        sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        image_grid = msk.ImageGrid(arr=image_grid, mask=mask)
        image_sub_grid = MockSubGrid(image_sub_grid, sub_to_image, sub_grid_size=1)

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_image_grid(image_grid=image_grid)

        grids = MockGridCollection(image=image_grid, sub=image_sub_grid, pix=image_plane_pix.pix_grid)

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None,
                                                  image_to_nearest_image_pix=image_plane_pix.image_to_pix)

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
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                   [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()

    def test__3x3_simple_grid__include_mask__create_using_image_grid(self):

        image_grid = np.array([             [1.0, 0.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                            [-1.0, 0.0]] )

        mask = msk.Mask(array=np.array([[True, False, True],
                                        [False, False, False],
                                        [True, False, True]]), pixel_scale=1.0)

        image_sub_grid = np.array([              [1.0, 0.0],
                                   [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                [-1.0, 0.0]])
        sub_to_image = np.array([0, 1, 2, 3, 4])

        image_grid = msk.ImageGrid(arr=image_grid, mask=mask)
        image_sub_grid = MockSubGrid(image_sub_grid, sub_to_image, sub_grid_size=1)

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_image_grid(image_grid=image_grid)

        grids = MockGridCollection(image=image_grid, sub=image_sub_grid, pix=image_plane_pix.pix_grid)

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None,
                                                  image_to_nearest_image_pix=image_plane_pix.image_to_pix)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__3x3_simple_grid__include_mask_and_sub_grid__create_using_image_grid(self):

        image_grid = np.array([             [1.0, 0.0],
                               [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                            [-1.0, 0.0]] )

        mask = msk.Mask(array=np.array([[True, False, True],
                                        [False, False, False],
                                        [True, False, True]]), pixel_scale=1.0)

        image_sub_grid = np.array([[1.01, 0.0], [1.01, 0.0], [1.01, 0.0], [0.01, 0.0],
                                  [0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.01, 0.0],
                                  [0.01, 0.0], [0.01, 0.0], [0.01, 0.0], [0.01, 0.0],
                                  [0.0, 1.01], [0.0, 1.01], [0.0, 1.01], [0.01, 0.0],
                                  [-1.01, 0.0], [-1.01, 0.0], [-1.01, 0.0], [0.01, 0.0]])

        sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

        image_grid = msk.ImageGrid(arr=image_grid, mask=mask)
        image_sub_grid = MockSubGrid(image_sub_grid, sub_to_image, sub_grid_size=2)

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(3, 3))
        image_plane_pix = pix.image_plane_pix_grid_from_image_grid(image_grid=image_grid)

        grids = MockGridCollection(image=image_grid, sub=image_sub_grid, pix=image_plane_pix.pix_grid)

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None,
                                                  image_to_nearest_image_pix=image_plane_pix.image_to_pix)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[0.75, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.75, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 2.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.75, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.75]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

class TestAdaptiveMagnification:

    def test__5_simple_grid__no_sub_grid(self):

        pixelization_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
        pixelization_sub_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

        sub_to_image = np.array([0, 1, 2, 3, 4])

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image, sub_grid_size=1),
                                   pix=pixelization_grid)

        image_to_pix = np.array([0, 1, 2, 3, 4])

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(5, 1))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None, image_to_nearest_image_pix=image_to_pix)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__15_grid__no_sub_grid(self):
        pixelization_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                      [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                      [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                      [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                      [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])
        pixelization_sub_grid = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                          [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                          [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                          [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                          [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

        pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

        sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image, sub_grid_size=1),
                                   pix=pixel_centers)

        image_to_pix = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(5, 1))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None, image_to_nearest_image_pix=image_to_pix)

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
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__5_simple_grid__include_sub_grid__sets_up_correct_mapper(self):
        pixelization_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
        pixelization_sub_grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                          [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                          [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                          [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

        sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

        pixel_centers = pixelization_grid

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image,
                                                   sub_grid_size=2),
                                   pix=pixel_centers)

        image_to_pix = np.array([0, 1, 2, 3, 4])

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(5, 1))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=None, image_to_nearest_image_pix=image_to_pix)

        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                       [0.0, 0.75, 0.25, 0.0, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.25, 0.75, 0.0],
                                                       [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

    def test__same_as_above_but_grid_requires_border_relocation(self):

        pixelization_grid = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        pixelization_sub_grid = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [-2.0, -2.0]])

        border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

        sub_to_image = np.array([0, 1, 2, 3, 4])

        pixel_centers = pixelization_grid

        grids = MockGridCollection(image=pixelization_grid,
                                   sub=MockSubGrid(pixelization_sub_grid, sub_to_image, sub_grid_size=1),
                                   pix=pixel_centers)

        image_to_pix = np.array([0, 1, 2, 3, 4])

        pix = pixelizations.AdaptiveMagnification(pix_grid_shape=(5, 1))

        mapper = pix.mapper_from_grids_and_border(grids=grids, border=border, image_to_nearest_image_pix=image_to_pix)


        assert isinstance(mapper, pm.VoronoiMapper)

        assert (mapper.mapping_matrix == np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 0.0]])).all()

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(mapper.pixel_neighbors)

        assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                   [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                   [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                   [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                   [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()