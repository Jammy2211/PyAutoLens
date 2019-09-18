import numpy as np
from functools import wraps

from autolens import exc
from autolens.array import grids
from autolens.array.util import array_util
from autolens.array.mapping_util import (
    array_mapping_util,
    grid_mapping_util,
    mask_mapping_util,
)
from autolens.array import scaled_array


def reshape_returned_array(func):
    @wraps(func)
    def wrapper(obj, return_in_2d=True, bypass_decorator=False, *args, **kwargs):
        """

        This wrapper decorates the _from_grid functions of profiles, which return 1D arrays of physical quantities \
        (e.g. image, convergences, potentials). Depending on the input variables, it determines whether the
        returned array is reshaped to 2D from 1D and if a sub-grid is input, it can bin the sub-gridded values to
        gridded values.

        Parameters
        ----------
        obj : autolens.model.geometry_profiles.Profile
            The profiles that owns the function
        grid : ndarray or Grid or Grid
            (y,x) in either cartesian or profiles coordinate system
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.

        Returns
        -------
            An array of a physical quantity that may be in 1D or 2D and binned up from a sub-grid.
        """

        if bypass_decorator:
            return func(obj)

        grid = kwargs["grid"] if "grid" in kwargs else None
        blurring_grid = kwargs["blurring_grid"] if "blurring_grid" in kwargs else None
        psf = kwargs["psf"] if "psf" in kwargs else None
        convolver = kwargs["convolver"] if "convolver" in kwargs else None

        return_masked = kwargs["return_masked"] if "return_masked" in kwargs else True

        if hasattr(obj, "mapping"):
            mapping = obj.mapping
        elif hasattr(grid, "mapping"):
            mapping = grid.mapping
        else:
            raise exc.MappingException(
                "Unable to find mapping object from the functions input object or any of its"
                "keyword arguments."
            )

        if grid is not None and psf is not None:
            array_from_func = func(obj, grid, psf, blurring_grid)
        elif grid is not None and convolver is not None:
            array_from_func = func(obj, grid, convolver, blurring_grid)
        elif grid is not None:
            array_from_func = func(obj, grid)
        else:
            array_from_func = func(obj)

        return reshaped_array_from_array_and_mapping(
            array=array_from_func,
            mapping=mapping,
            return_in_2d=return_in_2d,
            return_masked=return_masked,
        )

    return wrapper


def reshaped_array_from_array_and_mapping(array, mapping, return_in_2d, return_masked):

    if return_in_2d and not return_masked:
        return array

    if len(array.shape) == 2:
        array_1d = mapping.array_1d_from_array_2d(array_2d=array)
    else:
        array_1d = array

    if not return_in_2d:
        return array_1d
    else:
        return mapping.scaled_array_2d_from_array_1d(array_1d=array_1d)


def reshape_returned_sub_array(func):
    @wraps(func)
    def wrapper(object, grid, *args, **kwargs):
        """

        This wrapper decorates the _from_grid functions of profiles, which return 1D arrays of physical quantities \
        (e.g. image, convergences, potentials). Depending on the input variables, it determines whether the
        returned array is reshaped to 2D from 1D and if a sub-grid is input, it can bin the sub-gridded values to
        gridded values.

        Parameters
        ----------
        object : autolens.model.geometry_profiles.Profile
            The profiles that owns the function
        grid : ndarray or Grid or Grid
            (y,x) in either cartesian or profiles coordinate system
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.

        Returns
        -------
            An array of a physical quantity that may be in 1D or 2D and binned up from a sub-grid.
        """

        bypass_decorator = (
            kwargs["bypass_decorator"] if "bypass_decorator" in kwargs else False
        )

        if bypass_decorator:
            return func(object, grid)

        return_in_2d = kwargs["return_in_2d"] if "return_in_2d" in kwargs else True
        return_binned = kwargs["return_binned"] if "return_binned" in kwargs else True
        sub_array_from_func = func(object, grid)

        return reshaped_sub_array_from_sub_array_and_mapping(
            mapping=grid.mapping,
            sub_array=sub_array_from_func,
            return_in_2d=return_in_2d,
            return_binned=return_binned,
        )

    return wrapper


def reshaped_sub_array_from_sub_array_and_mapping(
    sub_array, mapping, return_in_2d, return_binned
):

    if len(sub_array.shape) == 2:
        sub_array_1d = mapping.sub_array_1d_with_sub_dimensions_from_sub_array_2d(
            sub_array_2d=sub_array
        )
    else:
        sub_array_1d = sub_array

    if not return_in_2d and not return_binned:
        return sub_array_1d

    elif not return_in_2d and return_binned:

        return mapping.array_1d_binned_from_sub_array_1d(sub_array_1d=sub_array_1d)

    elif return_in_2d and not return_binned:

        return mapping.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

    elif return_in_2d and return_binned:

        return mapping.scaled_array_2d_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )


def reshape_returned_grid(func):
    @wraps(func)
    def wrapper(object, grid, *args, **kwargs):
        """

        This wrapper decorates the _from_grid functions of profiles, which return 2D grids of physical quantities \
        (e.g. deflection angles). Depending on the input variables, it determines whether the
        returned grid is reshaped to 2D from 1D and if a sub-grid is input, it can bin the sub-gridded values to
        gridded values.

        Parameters
        ----------
        object : autolens.model.geometry_profiles.Profile
            The profiles that owns the function
        mapping : ndgrid or Grid or Grid
            (y,x) in either cartesian or profiles coordinate system

        Returns
        -------
            An grid of (y,x) coordinates that may be in 1D or 2D and binned up from a sub-grid.
        """

        bypass_decorator = (
            kwargs["bypass_decorator"] if "bypass_decorator" in kwargs else False
        )

        if bypass_decorator:
            return func(object, grid)

        return_in_2d = kwargs["return_in_2d"] if "return_in_2d" in kwargs else False
        return_binned = kwargs["return_binned"] if "return_binned" in kwargs else False

        mapping = grid.mapping

        grid_from_func = func(object, grid)

        if len(grid_from_func.shape) == 3:
            grid_y = reshaped_sub_array_from_sub_array_and_mapping(
                sub_array=grid_from_func[:, :, 0],
                mapping=mapping,
                return_in_2d=return_in_2d,
                return_binned=return_binned,
            )
            grid_x = reshaped_sub_array_from_sub_array_and_mapping(
                sub_array=grid_from_func[:, :, 1],
                mapping=mapping,
                return_in_2d=return_in_2d,
                return_binned=return_binned,
            )
        elif len(grid_from_func.shape) == 2:
            grid_y = reshaped_sub_array_from_sub_array_and_mapping(
                sub_array=grid_from_func[:, 0],
                mapping=mapping,
                return_in_2d=return_in_2d,
                return_binned=return_binned,
            )
            grid_x = reshaped_sub_array_from_sub_array_and_mapping(
                sub_array=grid_from_func[:, 1],
                mapping=mapping,
                return_in_2d=return_in_2d,
                return_binned=return_binned,
            )

        return grids.Grid(arr=np.stack((grid_y, grid_x), axis=-1), mask=grid.mask)

    return wrapper


class Mapping(object):
    def __init__(self, mask):
        self.mask = mask

    @property
    def shape(self):
        return self.mask.shape

    @property
    def sub_size(self):
        return self.mask.sub_size

    @property
    def pixel_scale(self):
        return self.mask.pixel_scale

    @property
    def origin(self):
        return self.mask.origin

    @property
    def mask_1d_index_tomask_index(self):
        """A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates."""
        return mask_mapping_util.sub_mask_1d_index_to_submask_index_from_mask_and_sub_size(
            mask=self.mask, sub_size=1
        ).astype(
            "int"
        )

    def array_1d_from_array_2d(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """
        if array_2d is None or isinstance(array_2d, float):
            return array_2d
        return array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=self.mask, sub_array_2d=array_2d, sub_size=1
        )

    def array_2d_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        return array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=self.mask, sub_size=1
        )

    def scaled_array_2d_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original masked 2D array and return it as a hyper \
        array.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array of which is mapped to a 2D hyper array.
        """
        return scaled_array.ScaledSquarePixelArray(
            array=self.array_2d_from_array_1d(array_1d=array_1d),
            pixel_scale=self.pixel_scale,
            origin=self.origin,
        )

    def grid_2d_from_grid_1d(self, grid_1d):
        """Map a 1D grid the same dimension as the grid to its original 2D grid.

        Values which were masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        grid_1d : ndarray
            The 1D grid which is mapped to its masked 2D array.
        """
        return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=self.mask, sub_size=1
        )

    def grid_1d_from_grid_2d(self, grid_2d):
        """ Map a 2D grid to its masked 1D grid..

        Values which are masked in the mapping_util to the 1D grid are returned as zeros.

        Parameters
        -----------
        grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D grid.
        """
        return grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=self.mask, sub_size=1
        )

    @property
    def sub_mask_1d_index_to_submask_index(self):
        """A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates."""
        return mask_mapping_util.sub_mask_1d_index_to_submask_index_from_mask_and_sub_size(
            mask=self.mask, sub_size=self.sub_size
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
            mask=self.mask, sub_size=self.sub_size
        ).astype("int")

    def sub_array_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=sub_array_1d, mask=self.mask, sub_size=self.sub_size
        )

    def scaled_array_2d_with_sub_dimensions_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a hyper array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D hyper sub-array the dimensions.
        """
        return scaled_array.ScaledSquarePixelArray(
            array=self.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d),
            pixel_scale=self.pixel_scale / self.sub_size,
            origin=self.origin,
        )

    def array_1d_binned_from_sub_array_1d(self, sub_array_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """

        sub_length = int(self.sub_size ** 2.0)
        sub_fraction = 1.0 / sub_length

        return np.multiply(
            sub_fraction, sub_array_1d.reshape(-1, sub_length).sum(axis=1)
        )

    def scaled_array_2d_binned_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a hyper array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D hyper sub-array the dimensions.
        """

        array_1d = self.array_1d_binned_from_sub_array_1d(sub_array_1d=sub_array_1d)

        return scaled_array.ScaledSquarePixelArray(
            array=self.array_2d_from_array_1d(array_1d=array_1d),
            pixel_scale=self.pixel_scale,
            origin=self.origin,
        )

    def sub_array_1d_with_sub_dimensions_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the mapping_util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        return array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=self.mask, sub_size=self.sub_size
        )

    def sub_grid_1d_with_sub_dimensions_from_sub_grid_2d(self, sub_grid_2d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_2d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """
        return grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=sub_grid_2d, mask=self.mask, sub_size=self.sub_size
        )

    def grid_1d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """

        grid_1d_y = self.array_1d_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.array_1d_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        return np.stack((grid_1d_y, grid_1d_x), axis=-1)

    def grid_2d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """

        grid_1d = self.grid_1d_binned_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

        return self.grid_2d_from_grid_1d(grid_1d=grid_1d)

    def sub_grid_2d_with_sub_dimensions_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D sub-grid.
        """
        return grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=sub_grid_1d, mask=self.mask, sub_size=self.sub_size
        )
