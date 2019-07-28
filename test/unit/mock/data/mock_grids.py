import numpy as np

from autolens.data.array.util import grid_util

from autolens.data.array import grids

from test.unit.mock.data import mock_mask


class MockGrid(grids.Grid):
    def __new__(cls, mask, pixel_scale=1.0, *args, **kwargs):

        regular_grid = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask, pixel_scales=(pixel_scale, pixel_scale), sub_grid_size=1
        )

        obj = regular_grid.view(cls)
        obj.mask = mask
        obj.interpolator = None

        return obj

    def __init__(self, mask, pixel_scale=1.0):
        pass


class MockGrid(grids.Grid):
    def __new__(cls, mask, pixel_scale=1.0, sub_grid_size=2, *args, **kwargs):

        sub_grid = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask,
            pixel_scales=(pixel_scale, pixel_scale),
            sub_grid_size=sub_grid_size,
        )

        obj = sub_grid.view(cls)
        obj.mask = mask
        obj.sub_grid_size = sub_grid_size
        obj.sub_grid_length = int(obj.sub_grid_size ** 2.0)
        obj.sub_grid_fraction = 1.0 / obj.sub_grid_length
        obj.interpolator = None

        return obj

    def __init__(self, mask, pixel_scale=1.0, sub_grid_size=2):
        pass


class MockClusterGrid(grids.ClusterGrid):

    pass


class MockGridStack(grids.GridStack):
    def __init__(self, regular, sub, blurring, pixelization=None):

        super(MockGridStack, self).__init__(
            regular=regular, sub=sub, blurring=blurring, pixelization=pixelization
        )


class MockPixSubGrid(np.ndarray):
    def __new__(cls, sub_grid, *args, **kwargs):
        return sub_grid.view(cls)

    def __init__(self, sub_grid, sub_to_regular, sub_grid_size):
        # noinspection PyArgumentList
        super().__init__()
        self.sub_grid_coords = sub_grid
        self.sub_to_regular = sub_to_regular
        self.total_pixels = sub_to_regular.shape[0]
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length


class MockPixGridStack(object):
    def __init__(
        self, regular, sub, blurring=None, pix=None, regular_to_pixelization=None
    ):
        self.regular = grids.Grid(regular, mask=None)
        self.sub = sub
        self.blurring = (
            grids.Grid(blurring, mask=None) if blurring is not None else None
        )
        self.pixelization = (
            grids.PixelizationGrid(
                pix, regular_to_pixelization=regular_to_pixelization, mask=None
            )
            if pix is not None
            else np.array([[0.0, 0.0]])
        )


class MockBorders(grids.GridBorder):
    def __new__(cls, arr=np.array([0]), *args, **kwargs):
        """The borders of a regular grid, containing the pixel-index's of all masked pixels that are on the \
        mask's border (e.g. they are next to a *True* value in at least one of the surrounding 8 pixels and at one of \
        the exterior edge's of the mask).

        This is used to relocate demagnified pixel's in a grid to its border, so that they do not disrupt an \
        adaptive pixelization's inversion.

        Parameters
        -----------
        arr : ndarray
            A 1D array of the integer indexes of an *RegularGrid*'s borders pixels.
        """
        border = arr.view(cls)
        return border
