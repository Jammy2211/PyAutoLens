from astropy import cosmology as cosmo
from typing import List

import autoarray as aa
import autogalaxy as ag


def traced_grid_2d_list_from(
    planes: List[ag.Plane],
    grid: aa.type.Grid2DLike,
    cosmology=cosmo.Planck15,
    plane_index_limit: int = None,
):
    """
    Performs multi-plane ray tracing on a 2D grid of Cartesian (y,x) coordinates using the mass profiles of galaxies
    contained within an input list of `Plane` objects.

    This function returns a list of 2D (y,x) grids, corresponding to each redshift in the input list of `Planes`. For
    example, if the `planes` list contains three `Plane` objects with `redshift`'s z0.5, z=1.0 and z=2.0, the returned
    list of traced grids will contain three entries corresponding to the input grid after ray-tracing to
    redshifts 0.5, 1.0 and 2.0.

    An input `AstroPy` cosmology object can change the cosmological model, which is used to compute the scaling
    factors between planes (which are derived from their redshifts and angular diameter distances). It is these
    scaling factors that account for multi-plane ray tracing effects.

    The calculation can be terminated early by inputting a `plane_index_limit`, whereby all planes whose integer
    indexes are above this value are omitted from the calculation and not included in the returned list of grids (the
    size of this list is reduced accordingly).

    For example, if `planes` has 3 `Plane` objects but `plane_index_limit=1`, the third plane (corresponding to
    index 2) will not be calculated. The `plane_index_limit` is often used to avoid uncessary ray tracing calculations
    of higher redshift planes whose galaxies do not have mass profile (and only have light profiles).

    Parameters
    ----------
    planes
        The list of planes whose galaxies and mass profiles are used to perform multi-plane ray-tracing.
    grid
        The 2D (y, x) coordinates on which multi-plane ray-tracing calculations are performed.
    cosmology
        The cosmology used for ray-tracing from which angular diameter distances between planes are computed.
    plane_index_limit
        The integer index of the last plane which is used to perform ray-tracing, all planes with an index above
        this value are omitted.

    Returns
    -------
    traced_grid_list
        A list of 2D (y,x) grids each of which are the input grid ray-traced to a redshift of the input list of planes.
    """

    traced_grid_list = []
    traced_deflection_list = []

    plane_redshifts = [plane.redshift for plane in planes]

    for (plane_index, plane) in enumerate(planes):

        scaled_grid = grid.copy()

        if plane_index > 0:
            for previous_plane_index in range(plane_index):
                scaling_factor = ag.util.cosmology.scaling_factor_between_redshifts_from(
                    redshift_0=plane_redshifts[previous_plane_index],
                    redshift_1=plane.redshift,
                    redshift_final=plane_redshifts[-1],
                    cosmology=cosmology,
                )

                scaled_deflections = (
                    scaling_factor * traced_deflection_list[previous_plane_index]
                )

                scaled_grid -= scaled_deflections

        traced_grid_list.append(scaled_grid)

        if plane_index_limit is not None:
            if plane_index == plane_index_limit:
                return traced_grid_list

        traced_deflection_list.append(plane.deflections_yx_2d_from(grid=scaled_grid))

    return traced_grid_list


def grid_2d_at_redshift_from(
    redshift: float,
    galaxies: List[ag.Galaxy],
    grid: aa.type.Grid2DLike,
    cosmology=cosmo.Planck15,
) -> aa.type.Grid2DLike:
    """
    Given a list of galaxies whose redshifts define a multi-plane lensing system and an input grid of (y,x) arc-second
    coordinates (e.g. an image-plane grid), ray-trace the grid to an input redshift in of the multi-plane system.

    This is performed using multi-plane ray-tracing and a list of galaxies which are converted into a list of planes
    at a set of redshift. The galaxy mass profiles are used to compute deflection angles. Any redshift can be input
    even if a plane does not exist there, including redshifts before the first plane of the lens system.

    An input `AstroPy` cosmology object can change the cosmological model, which is used to compute the scaling
    factors between planes (which are derived from their redshifts and angular diameter distances). It is these
    scaling factors that account for multi-plane ray tracing effects.

    For example, the input list `galaxies` may contained three `Galaxy` objects at redshifts z=0.5, z=1.0 and z=2.0.
    We can input an image-plane grid and request that its coordinates are ray-traced to a plane at z=1.75 in this
    multi-plane system. This will use the galaxy's at z=0.5 and z=1.0 to compute deflection angles, alongside
    accounting for multi-plane lensing effects via the angular diameter distances between the different galaxy
    redshifts.

    Parameters
    ----------
    redshift
        The redshift the input (image-plane) grid is traced too.
    galaxies
        A list of galaxies which make up a multi-plane strong lens ray-tracing system.
    grid
        The 2D (y, x) coordinates which is ray-traced to the input redshift.
    cosmology
        The cosmology used for ray-tracing from which angular diameter distances between planes are computed.
    """

    plane_redshifts = ag.util.plane.ordered_plane_redshifts_from(galaxies=galaxies)

    if redshift <= plane_redshifts[0]:
        return grid.copy()

    planes = ag.util.plane.planes_via_galaxies_from(galaxies=galaxies)

    plane_index_with_redshift = [
        plane_index
        for plane_index, plane in enumerate(planes)
        if plane.redshift == redshift
    ]

    if plane_index_with_redshift:

        traced_grid_list = traced_grid_2d_list_from(
            planes=planes, grid=grid, cosmology=cosmology
        )

        return traced_grid_list[plane_index_with_redshift[0]]

    for plane_index, plane_redshift in enumerate(plane_redshifts):

        if redshift < plane_redshift:
            plane_index_insert = plane_index

    planes.insert(plane_index_insert, ag.Plane(redshift=redshift, galaxies=[]))

    traced_grid_list = traced_grid_2d_list_from(
        planes=planes, grid=grid, cosmology=cosmology
    )

    return traced_grid_list[plane_index_insert]
