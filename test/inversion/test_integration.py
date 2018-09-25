from test.mock.mock_mask import MockSubGridCoords, MockGridCollection, MockBorderCollection
from autolens.imaging import mask as msk
from autolens.inversion import regularization
from autolens.inversion import mappers as pm
from autolens.inversion import pixelizations

import pytest
import numpy as np


class TestPixelizationMapperAndRegularizationFromPixelization:


    class TestRectangular:

        def test__5_simple_grid__no_sub_grid(self):
            # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.
            pixelization_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pixelization_border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))

            pixelization_sub_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pixelization_sub_border = msk.SubGridBorder(arr=np.array([0, 1, 3, 4]), sub_grid_size=1)

            sub_to_image = np.array([0, 1, 2, 3, 4])

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image, sub_grid_size=1))

            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            # There is no sub-grid, so our sub_grid are just the masked_image grid (note the NumPy weighted_data structure
            # ensures this has no sub-gridding)

            pix = pixelizations.Rectangular(shape=(3, 3))

            pix_mapper = pix.mapper_from_grids_and_borders(grids, borders)

            assert (pix_mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()
            assert pix_mapper.shape == (3, 3)

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

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

            pixelization_grid = np.array([[-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                          [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                          [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                          [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                          [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]])

            pixelization_border = msk.ImageGridBorder(arr=np.array([2, 5, 11, 14]))

            # There is no sub-grid, so our sub_grid are just the masked_image grid (note the NumPy weighted_data structure
            # ensures this has no sub-gridding)
            pixelization_sub_grid = np.array([[-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                              [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                              [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                              [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                              [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]])

            pixelization_sub_border = msk.SubGridBorder(arr=np.array([2, 5, 11, 14]))

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image,
                                                             sub_grid_size=1))

            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            pix = pixelizations.Rectangular(shape=(3, 3))

            pix_mapper = pix.mapper_from_grids_and_borders(grids, borders)

            assert (pix_mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            assert pix_mapper.shape == (3, 3)

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

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
            pixelization_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pixelization_border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping_matrix matrix which explicitly tests the
            # sub-grid.
            pixelization_sub_grid = np.array([[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0],
                                              [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                              [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                              [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                              [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

            sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])
            pixelization_sub_border = msk.SubGridBorder(arr=np.array([0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]))

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image,
                                                             sub_grid_size=2))

            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            pix = pixelizations.Rectangular(shape=(3, 3))

            pix_mapper = pix.mapper_from_grids_and_borders(grids, borders)

            assert (pix_mapper.mapping_matrix == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()
            assert pix_mapper.shape == (3, 3)

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

            assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                       [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                       [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                       [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                       [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()

        def test__same_as_above_but_grid_requires_border_relocation(self):
            # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.
            pixelization_grid = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            pixelization_border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping_matrix matrix which explicitly tests the
            # sub-grid.
            pixelization_sub_grid = np.array([[-1.0, -1.0], [-2.0, -2.0], [-2.0, -2.0], [0.0, 0.0],
                                              [-1.0, 1.0], [-2.0, 2.0], [-2.0, 2.0], [0.0, 0.0],
                                              [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                              [1.0, -1.0], [2.0, -2.0], [2.0, -2.0], [0.0, 0.0],
                                              [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]])

            pixelization_sub_border = msk.SubGridBorder(arr=np.array([0, 4, 12, 16]))

            sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image,
                                                             sub_grid_size=2))

            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            pix = pixelizations.Rectangular(shape=(3, 3))

            pix_mapper = pix.mapper_from_grids_and_borders(grids, borders)

            assert (pix_mapper.mapping_matrix == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()
            assert pix_mapper.shape == (3, 3)

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

            assert (regularization_matrix == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                       [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                       [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                       [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                       [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()


    class TestCluster:

        def test__5_simple_grid__no_sub_grid(self):

            pixelization_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pixelization_sub_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

            pixelization_border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            pixelization_sub_border = msk.SubGridBorder(arr=np.array([0, 1, 3, 4]), sub_grid_size=1)

            sub_to_image = np.array([0, 1, 2, 3, 4])

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image, sub_grid_size=1))
            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            image_to_voronoi = np.array([0, 1, 2, 3, 4])

            pixel_centers = pixelization_grid

            pix = pixelizations.Cluster(pixels=5)

            pix_mapper = pix.mapper_from_grids_and_borders(grids=grids, borders=borders,
                                                           pixel_centers=pixel_centers,
                                                           image_to_voronoi=image_to_voronoi)

            assert isinstance(pix_mapper, pm.VoronoiMapper)

            assert (pix_mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

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

            pixelization_border = msk.ImageGridBorder(arr=np.array([2, 5, 11, 14]))
            pixelization_sub_border = msk.SubGridBorder(arr=np.array([2, 5, 11, 14]))

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image,
                                                                   sub_grid_size=1))
            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            image_to_voronoi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

            pix = pixelizations.Cluster(pixels=5)

            pix_mapper = pix.mapper_from_grids_and_borders(grids=grids, borders=borders,
                                                           pixel_centers=pixel_centers,
                                                           image_to_voronoi=image_to_voronoi)

            assert isinstance(pix_mapper, pm.VoronoiMapper)

            assert (pix_mapper.mapping_matrix == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
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

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

            assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

        def test__5_simple_grid__include_sub_grid__sets_up_correct_pix_mapper(self):

            pixelization_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pixelization_sub_grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                     [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

            pixelization_border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            pixelization_sub_border = msk.SubGridBorder(arr=np.array([0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]))

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            pixel_centers = pixelization_grid
            image_to_voronoi = np.array([0, 1, 2, 3, 4])

            pix = pixelizations.Cluster(pixels=5)

            pix_mapper = pix.mapper_from_grids_and_borders(grids=grids, borders=borders,
                                                           pixel_centers=pixel_centers,
                                                           image_to_voronoi=image_to_voronoi)

            assert isinstance(pix_mapper, pm.VoronoiMapper)

            assert (pix_mapper.mapping_matrix == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                      [0.0, 0.75, 0.25, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.25, 0.75, 0.0],
                                                      [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

            assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

        def test__same_as_above_but_grid_requires_border_relocation(self):

            pixelization_grid = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            pixelization_sub_grid = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0],
                                     [-1.0, 1.0], [-2.0, 2.0], [-2.0, 2.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [1.0, -1.0], [2.0, -2.0], [2.0, -2.0], [0.0, 0.0],
                                     [-1.0, -1.0], [-2.0, -2.0], [-2.0, -2.0], [0.0, 0.0]])

            pixelization_border = msk.ImageGridBorder(arr=np.array([0, 1, 3, 4]))
            pixelization_sub_border = msk.SubGridBorder(arr=np.array([0, 4, 12, 16]))

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            grids = MockGridCollection(image=pixelization_grid,
                                       sub=MockSubGridCoords(pixelization_sub_grid, sub_to_image,
                                                                   sub_grid_size=2))

            borders = MockBorderCollection(image=pixelization_border, sub=pixelization_sub_border)

            pixel_centers = pixelization_grid
            image_to_voronoi = np.array([0, 1, 2, 3, 4])

            pix = pixelizations.Cluster(pixels=5)

            pix_mapper = pix.mapper_from_grids_and_borders(grids=grids, borders=borders,
                                                           pixel_centers=pixel_centers,
                                                           image_to_voronoi=image_to_voronoi)

            assert isinstance(pix_mapper, pm.VoronoiMapper)

            assert (pix_mapper.mapping_matrix == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                      [0.0, 0.75, 0.25, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.25, 0.75, 0.0],
                                                      [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

            reg = regularization.Constant(regularization_coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pix_mapper.pixel_neighbors)

            assert (regularization_matrix == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                       [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                       [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                       [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                       [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()