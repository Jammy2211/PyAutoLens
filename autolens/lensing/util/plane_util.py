from functools import wraps
import numpy as np
from autolens.data.array import grids
from autolens.data.array.util import grid_util, mapping_util
from autolens.lensing import plane as pl

def sub_to_image_grid(func):
    """
    Wrap the function in a function that, if the grid_stack is a sub-grid (grid_stacks.SubGrid), rebins the computed \
    values te the regular-grid by taking the mean of each set of sub-gridded values.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
        A function that requires the sub-grid and galaxies.
    """

    @wraps(func)
    def wrapper(grid, galaxies, *args, **kwargs):
        """

        Parameters
        ----------
        grid : ndarray
            PlaneCoordinates in either cartesian or profiles coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        result = func(grid, galaxies, *args, *kwargs)

        if isinstance(grid, grids.SubGrid):
            return grid.sub_data_to_regular_data(result)
        else:
            return result

    return wrapper

@sub_to_image_grid
def intensities_of_galaxies_from_grid(grid, galaxies):
    return sum(map(lambda g: g.intensities_from_grid(grid), galaxies))

@sub_to_image_grid
def surface_density_of_galaxies_from_grid(grid, galaxies):
    return sum(map(lambda g: g.surface_density_from_grid(grid), galaxies))

@sub_to_image_grid
def potential_of_galaxies_from_grid(grid, galaxies):
    return sum(map(lambda g: g.potential_from_grid(grid), galaxies))

def deflections_of_galaxies_from_grid(grid, galaxies):
    deflections = sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))
    if isinstance(grid, grids.SubGrid):
        return np.asarray([grid.sub_data_to_regular_data(deflections[:, 0]),
                           grid.sub_data_to_regular_data(deflections[:, 1])]).T
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

def deflections_of_galaxies_from_sub_grid(sub_grid, galaxies):
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(sub_grid), galaxies))

def deflections_of_galaxies_from_grid_stack(grid_stack, galaxies):
    return grid_stack.apply_function(lambda grid: deflections_of_galaxies_from_sub_grid(grid, galaxies))

def plane_image_of_galaxies_from_grid(shape, grid, galaxies, buffer=1.0e-2):

    y_min = np.min(grid[:, 0]) - buffer
    y_max = np.max(grid[:, 0]) + buffer
    x_min = np.min(grid[:, 1]) - buffer
    x_max = np.max(grid[:, 1]) + buffer

    pixel_scales = (float((y_max - y_min) / shape[0]), float((x_max - x_min) / shape[1]))
    origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

    uniform_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full(shape=shape,
                                                                                                   fill_value=False),
                                                                                      pixel_scales=pixel_scales,
                                                                                      origin=origin)

    image_1d = sum([intensities_of_galaxies_from_grid(uniform_grid, [galaxy]) for galaxy in galaxies])

    image_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=image_1d, shape=shape)

    return pl.PlaneImage(array=image_2d, pixel_scales=pixel_scales, grid=grid, origin=origin)

def traced_collection_for_deflections(grid_stack, deflections):

    def subtract_scaled_deflections(grid, scaled_deflection):
        return np.subtract(grid, scaled_deflection)

    return grid_stack.map_function(subtract_scaled_deflections, deflections)

def ordered_redshifts_from_galaxies(galaxies):
    """Given a list of galaxies (with redshifts), return a list of the in ascending order.

    If two or more galaxies have the same redshift that redshift is not double counted.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

    # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
    # Using a list of class attributes so make a list of redshifts for now.

    galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, ordered_galaxies))
    return [redshift for i, redshift in enumerate(galaxy_redshifts) if redshift not in galaxy_redshifts[:i]]

def galaxies_in_redshift_ordered_lists_from_galaxies(galaxies, ordered_redshifts):
    """Given a list of galaxies (with redshifts), return a list of the galaxies where each entry contains a list \
    of galaxies at the same redshift in ascending redshift order.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

    galaxies_in_redshift_ordered_lists = []

    for (index, redshift) in enumerate(ordered_redshifts):

        galaxies_in_redshift_ordered_lists.append(list(map(lambda galaxy:
                                                            galaxy if galaxy.redshift == redshift else None,
                                                            ordered_galaxies)))

        galaxies_in_redshift_ordered_lists[index] = list(filter(None, galaxies_in_redshift_ordered_lists[index]))

    return galaxies_in_redshift_ordered_lists

def scaling_factor_between_redshifts_for_cosmology(z1, z2, z_final, cosmology):

    angular_diameter_distance_between_z1_z2 = cosmology.angular_diameter_distance_z1z2(z1=z1, z2=z2).to('kpc').value
    angular_diameter_distance_to_z_final = cosmology.angular_diameter_distance(z=z_final).to('kpc').value
    angular_diameter_distance_of_z2_to_earth = cosmology.angular_diameter_distance(z=z2).to('kpc').value
    angular_diameter_distance_between_z2_z_final = \
        cosmology.angular_diameter_distance_z1z2(z1=z1, z2=z_final).to('kpc').value

    return (angular_diameter_distance_between_z1_z2 * angular_diameter_distance_to_z_final) / \
           (angular_diameter_distance_of_z2_to_earth * angular_diameter_distance_between_z2_z_final)