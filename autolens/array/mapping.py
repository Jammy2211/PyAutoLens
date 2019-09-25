import numpy as np

from autolens.array import grids
from autolens.array.util import array_util
from autolens.array.mapping_util import (
    array_mapping_util,
    grid_mapping_util,
    mask_mapping_util,
)

from autolens.array import scaled_array

class Mapping(object):

    def __init__(self, mask):
        self.mask = mask

    @property
    def geometry(self):
        return self.mask.geometry

    @property
    def mask_1d_index_to_mask_2d_index(self):
        """A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates."""
        return mask_mapping_util.sub_mask_1d_index_to_sub_mask_2d_index_from_mask_and_sub_size(
            mask=self.mask, sub_size=1
        ).astype(
            "int"
        )

    @property
    def sub_mask_1d_index_to_sub_mask_2d_index(self):
        """A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates."""
        return mask_mapping_util.sub_mask_1d_index_to_sub_mask_2d_index_from_mask_and_sub_size(
            mask=self.mask, sub_size=self.geometry.sub_size
        ).astype(
            "int"
        )

    @property
    @array_util.Memoizer()
    def sub_mask_1d_index_to_mask_1d_index(self):
        """The mapping_util between every sub-pixel and its host pixel.

        For example:

        - sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd pixel.
        - sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th pixel.
        """
        return mask_mapping_util.sub_mask_1d_index_to_mask_1d_index_from_mask(
            mask=self.mask, sub_size=self.geometry.sub_size
        ).astype("int")

    def scaled_array_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return scaled_array.ScaledArray(sub_array_1d=array_1d, mask=mask)

    def scaled_array_from_array_2d(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """
        array_1d = array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=self.mask, sub_array_2d=array_2d, sub_size=1
        )
        return self.scaled_array_from_array_1d(array_1d=array_1d)

    def scaled_array_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return scaled_array.ScaledArray(sub_array_1d=sub_array_1d, mask=self.mask)

    def scaled_array_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        sub_array_1d = array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=self.mask, sub_size=self.geometry.sub_size
        )
        return self.scaled_array_from_sub_array_1d(sub_array_1d=sub_array_1d)

    def scaled_array_binned_from_sub_array_1d(self, sub_array_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """
        binned_array_1d = np.multiply(
            self.geometry.sub_fraction, sub_array_1d.reshape(-1, self.geometry.sub_length).sum(axis=1)
        )

        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return scaled_array.ScaledArray(sub_array_1d=binned_array_1d, mask=mask)

    def sub_array_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=sub_array_1d, mask=self.mask, sub_size=self.geometry.sub_size
        )

    def sub_array_2d_binned_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a hyper array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D hyper sub-array the dimensions.
        """
        binned_array_1d = np.multiply(
            self.geometry.sub_fraction, sub_array_1d.reshape(-1, self.geometry.sub_length).sum(axis=1)
        )
        return array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(sub_array_1d=binned_array_1d, mask=self.mask, sub_size=1)

    def grid_from_grid_1d(self, grid_1d):
        """ Map a 1D grid the same dimension as the grid to its original 2D grid.

        Values which were masked in the mapping_util to the 1D grid are returned as zeros.

        Parameters
        -----------
        grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D grid.
        """
        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return grids.Grid(sub_grid_1d=grid_1d, mask=mask)

    def grid_from_grid_2d(self, grid_2d):
        """For a 2D grid (e.g. an image, noise_map, etc.) map it to a masked 1D grid of valuees using this mask.

        Parameters
        ----------
        grid_2d : ndgrid | None | float
            The 2D grid to be mapped to a masked 1D grid.
        """
        grid_1d = grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            mask=self.mask, sub_grid_2d=grid_2d, sub_size=1
        )
        return self.grid_from_grid_1d(grid_1d=grid_1d)

    def grid_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grids.Grid(sub_grid_1d=sub_grid_1d, mask=self.mask)

    def grid_from_sub_grid_2d(self, sub_grid_2d):
        """ Map a 2D sub-grid to its masked 1D sub-grid.

        Values which are masked in the mapping_util to the 1D grid are returned as zeros.

        Parameters
        -----------
        su_grid_2d : ndgrid
            The 2D sub-grid which is mapped to its masked 1D sub-grid.
        """
        sub_grid_1d = grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=sub_grid_2d, mask=self.mask, sub_size=self.geometry.sub_size
        )
        return self.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

    def grid_binned_from_sub_grid_1d(self, sub_grid_1d):
        """For an input 1D sub-grid, map its values to a 1D grid of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            A 1D sub-grid of values (e.g. image, convergence, potential) which is mapped to
            a 1d grid.
        """

        grid_1d_y = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return grids.Grid(sub_grid_1d=np.stack((grid_1d_y, grid_1d_x), axis=-1), mask=mask)

    def sub_grid_2d_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=sub_grid_1d, mask=self.mask, sub_size=self.geometry.sub_size
        )

    def sub_grid_2d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid to its original masked 2D sub-grid and return it as
        a hyper grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub-grid of which is mapped to a 2D hyper sub-grid the dimensions.
        """

        grid_1d_y = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        binned_grid_1d = np.stack((grid_1d_y, grid_1d_x), axis=-1)

        return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(sub_grid_1d=binned_grid_1d, mask=self.mask, sub_size=1)