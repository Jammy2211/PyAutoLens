from autolens import exc

import numpy as np

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

def galaxies_in_redshift_ordered_planes_from_galaxies(galaxies, plane_redshifts):
    """Given a list of galaxies (with redshifts), return a list of the galaxies where each entry contains a list \
    of galaxies at the same redshift in ascending redshift order.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

    galaxies_in_redshift_ordered_planes = []

    for (plane_index, redshift) in enumerate(plane_redshifts):

        galaxies_in_redshift_ordered_planes.append(list(map(lambda galaxy:
                                                            galaxy if galaxy.redshift == redshift else None,
                                                            ordered_galaxies)))

        galaxies_in_redshift_ordered_planes[plane_index] = \
            list(filter(None, galaxies_in_redshift_ordered_planes[plane_index]))

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