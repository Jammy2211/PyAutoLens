from autolens import exc
from autolens.data.array.util import grid_util, mapping_util
from autolens.model.galaxy.util import galaxy_util
from autolens.lens import plane as pl

import numpy as np

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

    image_1d = sum([galaxy_util.intensities_of_galaxies_from_grid(grid=uniform_grid, galaxies=[galaxy])
                    for galaxy in galaxies])

    image_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=image_1d, shape=shape)

    return pl.PlaneImage(array=image_2d, pixel_scales=pixel_scales, grid=grid, origin=origin)

def ordered_plane_redshifts_from_galaxies(galaxies):
    """Given a list of galaxies (with redshifts), return a list of the redshifts in ascending order.

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

def ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(lens_redshifts, planes_between_lenses,
                                                                                 source_plane_redshift):
    """Given a set of lens plane redshifts, the source-plane redshift and the number of planes between each, setup the \
    plane redshifts using these values. A lens redshift corresponds to the 'main' lens galaxy(s),
    whereas the slices collect line-of-sight halos over a range of redshifts.

    The source-plane redshift is removed from the ordered plane redshifts that are returned, so that galaxies are not \
    planed at the source-plane redshift.

    For example, if the main plane redshifts are [1.0, 2.0], and the bin sizes are [1,3], the following redshift \
    slices for planes will be used:

    z=0.5
    z=1.0
    z=1.25
    z=1.5
    z=1.75
    z=2.0

    Parameters
    -----------
    lens_redshifts : [float]
        The redshifts of the main-planes (e.g. the lens galaxy), which determine where redshift intervals are placed.
    planes_between_lenses : [int]
        The number of slices between each main plane. The first entry in this list determines the number of slices \
        between Earth (redshift 0.0) and main plane 0, the next between main planes 0 and 1, etc.
    source_plane_redshift : float
        The redshift of the source-plane, which is input explicitly to ensure galaxies are not placed in the \
        source-plane.
    """

    # Check that the number of slices between lens planes is equal to the number of intervals between the lens planes.
    if len(lens_redshifts) != len(planes_between_lenses)-1:
        raise exc.RayTracingException('The number of lens_plane_redshifts input is not equal to the number of '
                                      'slices_between_lens_planes+1.')

    plane_redshifts = []

    # Add redshift 0.0 and the source plane redshifit to the lens plane redshifts, so that calculation below can use
    # them when dividing slices. These will be removed by the return function at the end from the plane redshifts.

    lens_redshifts.insert(0, 0.0)
    lens_redshifts.append(source_plane_redshift)

    for lens_plane_index in range(1, len(lens_redshifts)):

        previous_plane_redshift = lens_redshifts[lens_plane_index - 1]
        plane_redshift = lens_redshifts[lens_plane_index]
        slice_total = planes_between_lenses[lens_plane_index - 1]
        plane_redshifts += list(np.linspace(previous_plane_redshift, plane_redshift, slice_total+2))[1:]

    return plane_redshifts[0:-1]

def galaxies_in_redshift_ordered_planes_from_galaxies(galaxies, plane_redshifts):
    """Given a list of galaxies (with redshifts), return a list of the galaxies where each entry contains a list \
    of galaxies at the same redshift in ascending redshift order.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """

    galaxies_in_redshift_ordered_planes =  [[] for i in range(len(plane_redshifts))]

    for galaxy in galaxies:

        index = (np.abs(np.asarray(plane_redshifts) - galaxy.redshift)).argmin()

        galaxies_in_redshift_ordered_planes[index].append(galaxy)

    return galaxies_in_redshift_ordered_planes

def compute_deflections_at_next_plane(plane_index, total_planes):
    """This function determines whether the tracer should compute the deflections at the next plane.

    This is True if there is another plane after this plane, else it is False..

    Parameters
    -----------
    plane_index : int
        The index of the plane we are deciding if we should compute its deflections.
    total_planes : int
        The total number of planes."""

    if plane_index < total_planes - 1:
        return True
    elif plane_index == total_planes - 1:
        return False
    else:
        raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

def scaling_factor_between_redshifts_for_cosmology(z1, z2, z_final, cosmology):

    angular_diameter_distance_between_z1_z2 = cosmology.angular_diameter_distance_z1z2(z1=z1, z2=z2).to('kpc').value
    angular_diameter_distance_to_z_final = cosmology.angular_diameter_distance(z=z_final).to('kpc').value
    angular_diameter_distance_of_z2_to_earth = cosmology.angular_diameter_distance(z=z2).to('kpc').value
    angular_diameter_distance_between_z2_z_final = \
        cosmology.angular_diameter_distance_z1z2(z1=z1, z2=z_final).to('kpc').value

    return (angular_diameter_distance_between_z1_z2 * angular_diameter_distance_to_z_final) / \
           (angular_diameter_distance_of_z2_to_earth * angular_diameter_distance_between_z2_z_final)

def scaled_deflection_stack_from_plane_and_scaling_factor(plane, scaling_factor):
    """Given a plane and scaling factor, compute a set of scaled deflections.

    Parameters
    -----------
    plane : plane.Plane
        The plane whose deflection stack is scaled.
    scaling_factor : float
        The factor the deflection angles are scaled by, which is typically the scaling factor between redshifts for \
        multi-plane lensing.
    """

    def scale(grid):
        return np.multiply(scaling_factor, grid)

    if plane.deflection_stack is not None:
        return plane.deflection_stack.apply_function(scale)
    else:
        return None

def grid_stack_from_deflection_stack(grid_stack, deflection_stack):
    """For a deflection stack, comput a new grid stack but subtracting the deflections"""

    if deflection_stack is not None:
        def minus(grid, deflections):
            return grid - deflections

        return grid_stack.map_function(minus, deflection_stack)


def traced_collection_for_deflections(grid_stack, deflections):

    def subtract_scaled_deflections(grid, scaled_deflection):
        return np.subtract(grid, scaled_deflection)

    return grid_stack.map_function(subtract_scaled_deflections, deflections)