from functools import wraps
import numpy as np


def avg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Parameters
        ----------
        results : Sized
            A collection of numerical values or tuples
        Returns
        -------
            The logical average of that collection
        """
        results = func(*args, **kwargs)
        try:
            return sum(results) / len(results)
        except TypeError:
            sum_tuple = (0, 0)
            for t in results:
                sum_tuple = (sum_tuple[0] + t[0], sum_tuple[1] + t[1])
            return sum_tuple[0] / len(results), sum_tuple[1] / len(results)

    return wrapper


def subgrid(func):
    """
    Decorator to permit generic subgridding
    Parameters
    ----------
    func : function(coordinates) -> value OR (value, value)
        Function that takes coordinates and calculates some value
    Returns
    -------
    func: function(coordinates, pixel_scale, grid_size)
        Function that takes coordinates and pixel scale/grid_size required for subgridding
    """

    @wraps(func)
    def wrapper(self, coordinates, pixel_scale=0.1, grid_size=1):
        """

        Parameters
        ----------
        self
        coordinates : (float, float)
            A coordinate pair
        pixel_scale : float
            The scale of a pixel
        grid_size : int
            The side length of the subgrid (i.e. there will be grid_size^2 pixels)
        Returns
        -------
        result : [value] or [(value, value)]
            A list of results
        """

        half = pixel_scale / 2
        step = pixel_scale / (grid_size + 1)
        results = []
        for x in range(grid_size):
            for y in range(grid_size):
                x1 = coordinates[0] - half + (x + 1) * step
                y1 = coordinates[1] - half + (y + 1) * step
                results.append(func(self, (x1, y1)))
        return results

    return wrapper


def iterative_subgrid(subgrid_func):
    """
    Decorator to iteratively increase the grid size until the difference between results reaches a defined threshold
    Parameters
    ----------
    subgrid_func : function(coordinates, pixel_scale, grid_size) -> value
        A function decorated with subgrid and average
    Returns
    -------
        A function that will iteratively increase grid size until a desired accuracy is reached
    """

    @wraps(subgrid_func)
    def wrapper(self, coordinates, pixel_scale=0.1, threshold=0.0001):
        """

        Parameters
        ----------
        self : Profile
            The instance that owns the function being wrapped
        coordinates : (float, float)
            x, y coordinates in image space
        pixel_scale : float
            The size of a pixel
        threshold : float
            The minimum difference between the result at two different grid sizes
        Returns
        -------
            The last result calculated once the difference between two results becomes lower than the threshold
        """
        last_result = None
        grid_size = 1
        while True:
            next_result = subgrid_func(self, coordinates, pixel_scale=pixel_scale, grid_size=grid_size)
            if last_result is not None and abs(next_result - last_result) < threshold:
                return next_result
            last_result = next_result
            grid_size += 1

    return wrapper


def array_function(func):
    """

    Parameters
    ----------
    func : function(coordinates)
            A function that takes coordinates and returns a value

    Returns
    -------
        A function that takes bounds, a pixel scale and mask and returns an array
    """

    @wraps(func)
    def wrapper(x_min=-5, y_min=-5, x_max=5, y_max=5, pixel_scale=0.1, mask=None):
        """

        Parameters
        ----------
        self : object
            Object that owns the function
        mask : Mask
            An object that has an is_masked method which returns True if (x, y) coordinates should be masked (i.e. not
            return a value)
        x_min : float
            The minimum x bound
        y_min : float
            The minimum y bound
        x_max : float
            The maximum x bound
        y_max : float
            The maximum y bound
        pixel_scale : float
            The arcsecond (") size of each pixel

        Returns
        -------
        array
            A 2D numpy array of values returned by the function at each coordinate
        """
        x_size = side_length(x_min, x_max, pixel_scale)
        y_size = side_length(y_min, y_max, pixel_scale)

        array = []

        for i in range(x_size):
            row = []
            for j in range(y_size):
                x = pixel_to_coordinate(x_min, pixel_scale, i)
                y = pixel_to_coordinate(y_min, pixel_scale, j)
                if mask is not None and mask.is_masked((x, y)):
                    row.append(None)
                else:
                    row.append(func((x, y)))
            array.append(row)
        # This conversion was to resolve a bug with putting tuples in the array. It might increase execution time.
        return np.array(array)

    return wrapper


def side_length(dim_min, dim_max, pixel_scale):
    return int((dim_max - dim_min) / pixel_scale)


def pixel_to_coordinate(dim_min, pixel_scale, pixel_coordinate):
    return dim_min + pixel_coordinate * pixel_scale
