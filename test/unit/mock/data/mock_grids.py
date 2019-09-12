import numpy as np

import autolens as al


class MockGrid(al.Grid):
    def __new__(cls, mask, pixel_scale=1.0, sub_size=2, *args, **kwargs):
        grid = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(pixel_scale, pixel_scale), sub_size=sub_size
        )

        obj = grid.view(cls)
        obj.mask = mask
        obj.sub_size = sub_size
        obj.sub_length = int(obj.sub_size ** 2.0)
        obj.sub_fraction = 1.0 / obj.sub_length
        obj.interpolator = None

        return obj

    def __init__(self, mask, pixel_scale=1.0, sub_size=2):
        pass


class MockBinnedGrid(al.BinnedGrid):
    pass


class MockPixelizationGrid(np.ndarray):
    def __new__(
        cls,
        arr,
        mask_1d_index_to_nearest_pixelization_1d_index=None,
        sub_mask_1d_index_to_mask_1d_index=None,
        sub_size=1,
        *args,
        **kwargs
    ):
        """A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        mask_1d_index_to_nearest_pixelization_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """
        obj = arr.view(cls)
        obj.mask_1d_index_to_nearest_pixelization_1d_index = (
            mask_1d_index_to_nearest_pixelization_1d_index
        )
        obj.sub_mask_1d_index_to_mask_1d_index = sub_mask_1d_index_to_mask_1d_index
        obj.sub_size = sub_size
        obj.sub_length = int(sub_size ** 2.0)
        obj.sub_fraction = 1.0 / obj.sub_length
        obj.interpolator = None
        return obj

    @property
    def mapping(self):
        return self

    def relocated_grid_from_grid(self, grid):
        return grid
