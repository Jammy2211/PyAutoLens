import autolens as al
import numpy as np
import pytest

from test.unit.mock.data.mock_grids import MockPixelizationGrid


class TestRectangular:
    def test__5_simple_grid__no_sub_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=1.0, sub_size=1)

        # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.

        grid = al.Grid(
            arr=np.array(
                [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
            ),
            mask=mask,
            sub_size=1,
        )

        pixelization_grid = MockPixelizationGrid(arr=grid)

        # There is no sub-grid, so our grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)

        pix = al.pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid,
            pixelization_grid=pixelization_grid,
            inversion_uses_border=False,
            hyper_image=np.ones((2, 2)),
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()
        assert mapper.shape == (3, 3)
        assert (mapper.hyper_image == np.ones((2, 2))).all()

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            pixel_neighbors=mapper.geometry.pixel_neighbors,
            pixel_neighbors_size=mapper.geometry.pixel_neighbors_size,
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__15_grid__no_sub_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=1.0, sub_size=1)

        # There is no sub-grid, so our grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)
        grid = al.Grid(
            arr=np.array(
                [
                    [0.9, -0.9],
                    [1.0, -1.0],
                    [1.1, -1.1],
                    [0.9, 0.9],
                    [1.0, 1.0],
                    [1.1, 1.1],
                    [-0.01, 0.01],
                    [0.0, 0.0],
                    [0.01, 0.01],
                    [-0.9, -0.9],
                    [-1.0, -1.0],
                    [-1.1, -1.1],
                    [-0.9, 0.9],
                    [-1.0, 1.0],
                    [-1.1, 1.1],
                ]
            ),
            mask=mask,
            sub_size=1,
        )

        pixelization_grid = MockPixelizationGrid(arr=grid)

        pix = al.pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.2, 2.2), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()
        assert mapper.shape == (3, 3)

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__5_simple_grid__include_sub_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)

        # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
        # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
        # happen for a real lensing calculation. This is to make a mapping_matrix matrix which explicitly tests the
        # sub-grid.
        grid = al.Grid(
            arr=np.array(
                [
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, -1.0],
                    [-1.0, -1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            mask=mask,
            sub_size=2,
        )

        pixelization_grid = MockPixelizationGrid(arr=grid)

        pix = al.pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.75],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert mapper.shape == (3, 3)

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__grid__requires_border_relocation(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=1.0, sub_size=1)

        grid = al.Grid(
            arr=np.array(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
            ),
            mask=mask,
            sub_size=1,
        )

        pixelization_grid = MockPixelizationGrid(grid)

        pix = al.pixelizations.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=True
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert mapper.shape == (3, 3)

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()


class TestVoronoiMagnification:
    def test__3x3_simple_grid(self):

        mask = al.Mask(
            array=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        grid = np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )

        grid = al.Grid(arr=grid, mask=mask)

        pix = al.pixelizations.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=pix.shape, grid=grid.unlensed_unsubbed_1d
        )

        pixelization_grid = MockPixelizationGrid(
            arr=sparse_to_grid.sparse,
            mask_1d_index_to_nearest_pixelization_1d_index=sparse_to_grid.mask_1d_index_to_sparse_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid,
            pixelization_grid=pixelization_grid,
            inversion_uses_border=False,
            hyper_image=np.ones((2, 2)),
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)
        assert (mapper.hyper_image == np.ones((2, 2))).all()

        assert isinstance(mapper, al.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__3x3_simple_grid__include_mask(self):

        mask = al.Mask(
            array=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        grid = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        grid = al.Grid(arr=grid, mask=mask)

        pix = al.pixelizations.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=pix.shape, grid=grid.unlensed_unsubbed_1d
        )

        pixelization_grid = MockPixelizationGrid(
            arr=grid,
            mask_1d_index_to_nearest_pixelization_1d_index=sparse_to_grid.mask_1d_index_to_sparse_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert isinstance(mapper, al.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [3.00000001, -1.0, -1.0, -1.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                    [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                    [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, -1.0, -1.0, -1.0, 3.00000001],
                ]
            )
        ).all()

    def test__3x3_simple_grid__include_mask_and_sub_grid(self):

        mask = al.Mask(
            array=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scale=1.0,
            sub_size=2,
        )

        grid = np.array(
            [
                [1.01, 0.0],
                [1.01, 0.0],
                [1.01, 0.0],
                [0.01, 0.0],
                [0.0, -1.0],
                [0.0, -1.0],
                [0.0, -1.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.0, 1.01],
                [0.0, 1.01],
                [0.0, 1.01],
                [0.01, 0.0],
                [-1.01, 0.0],
                [-1.01, 0.0],
                [-1.01, 0.0],
                [0.01, 0.0],
            ]
        )

        grid = al.Grid(arr=grid, mask=mask)

        pix = al.pixelizations.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=pix.shape, grid=grid.unlensed_unsubbed_1d
        )

        pixelization_grid = MockPixelizationGrid(
            arr=sparse_to_grid.sparse,
            mask_1d_index_to_nearest_pixelization_1d_index=sparse_to_grid.mask_1d_index_to_sparse_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.02, 2.01), 1.0e-4)
        assert (mapper.geometry.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.005), 1.0e-4)

        assert isinstance(mapper, al.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [0.75, 0.0, 0.25, 0.0, 0.0],
                    [0.0, 0.75, 0.25, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.25, 0.75, 0.0],
                    [0.0, 0.0, 0.25, 0.0, 0.75],
                ]
            )
        ).all()

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [3.00000001, -1.0, -1.0, -1.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                    [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                    [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, -1.0, -1.0, -1.0, 3.00000001],
                ]
            )
        ).all()

    def test__3x3_simple_grid__include_mask_with_offset_centre(self):

        mask = al.Mask(
            array=np.array(
                [
                    [True, True, True, False, True],
                    [True, True, False, False, False],
                    [True, True, True, False, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        grid = np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]])

        grid = al.Grid(arr=grid, mask=mask, sub_size=1)

        pix = al.pixelizations.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=pix.shape, grid=grid.unlensed_unsubbed_1d
        )

        pixelization_grid = MockPixelizationGrid(
            arr=sparse_to_grid.sparse,
            mask_1d_index_to_nearest_pixelization_1d_index=sparse_to_grid.mask_1d_index_to_sparse_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.geometry.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((1.0, 1.0), 1.0e-4)

        assert isinstance(mapper, al.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()

        reg = al.regularization.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(
            mapper.geometry.pixel_neighbors, mapper.geometry.pixel_neighbors_size
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [3.00000001, -1.0, -1.0, -1.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                    [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                    [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, -1.0, -1.0, -1.0, 3.00000001],
                ]
            )
        ).all()
