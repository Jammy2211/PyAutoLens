from autoarray.structures import grids
from autolens import exc
from autolens.lens import plane as pl

import numpy as np


def plane_image_of_galaxies_from_grid(shape, grid, galaxies, buffer=1.0e-2):

    y_min = np.min(grid[:, 0]) - buffer
    y_max = np.max(grid[:, 0]) + buffer
    x_min = np.min(grid[:, 1]) - buffer
    x_max = np.max(grid[:, 1]) + buffer

    pixel_scales = (
        float((y_max - y_min) / shape[0]),
        float((x_max - x_min) / shape[1]),
    )
    origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

    uniform_grid = grids.Grid.uniform(
        shape_2d=shape, pixel_scales=pixel_scales, sub_size=1, origin=origin
    )

    image = sum(map(lambda g: g.profile_image_from_grid(grid=uniform_grid), galaxies))

    return pl.PlaneImage(array=image, grid=grid)


def ordered_plane_redshifts_from_galaxies(galaxies):
    """Given a list of galaxies (with redshifts), return a list of the redshifts in ascending order.

    If two or more galaxies have the same redshift that redshift is not double counted.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(
        galaxies, key=lambda galaxy: galaxy.redshift, reverse=False
    )

    # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
    # Using a list of class attributes so make a list of redshifts for now.

    galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, ordered_galaxies))
    return [
        redshift
        for i, redshift in enumerate(galaxy_redshifts)
        if redshift not in galaxy_redshifts[:i]
    ]


def ordered_plane_redshifts_from_lens_source_plane_redshifts_and_slice_sizes(
    lens_redshifts, planes_between_lenses, source_plane_redshift
):
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
    if len(lens_redshifts) != len(planes_between_lenses) - 1:
        raise exc.RayTracingException(
            "The number of lens_plane_redshifts input is not equal to the number of "
            "slices_between_lens_planes+1."
        )

    plane_redshifts = []

    # Add redshift 0.0 and the source plane redshifit to the lens plane redshifts, so that calculation below can use
    # them when dividing slices. These will be removed by the return function at the end from the plane redshifts.

    lens_redshifts.insert(0, 0.0)
    lens_redshifts.append(source_plane_redshift)

    for lens_plane_index in range(1, len(lens_redshifts)):

        previous_plane_redshift = lens_redshifts[lens_plane_index - 1]
        plane_redshift = lens_redshifts[lens_plane_index]
        slice_total = planes_between_lenses[lens_plane_index - 1]
        plane_redshifts += list(
            np.linspace(previous_plane_redshift, plane_redshift, slice_total + 2)
        )[1:]

    return plane_redshifts[0:-1]


def galaxies_in_redshift_ordered_planes_from_galaxies(galaxies, plane_redshifts):
    """Given a list of galaxies (with redshifts), return a list of the galaxies where each entry contains a list \
    of galaxies at the same redshift in ascending redshift order.

    Parameters
    -----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """

    galaxies_in_redshift_ordered_planes = [[] for i in range(len(plane_redshifts))]

    for galaxy in galaxies:

        index = (np.abs(np.asarray(plane_redshifts) - galaxy.redshift)).argmin()

        galaxies_in_redshift_ordered_planes[index].append(galaxy)

    return galaxies_in_redshift_ordered_planes
